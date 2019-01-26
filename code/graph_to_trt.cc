#ifdef TRT
#include <algorithm>
#include <chrono>
#include "ops.h"
#include "cuda_helper.h"

#define TIMING_ITERATIONS 10

class SplitPlugin : public IPlugin {
public:
  SplitPlugin(int nOuts, int *channels_): nOuts(nOuts) {
    assert(nOuts <= MAX_NUM_OUTPUTS);
    for (int i = 0; i < nOuts; i++) {
      channels[i] = channels_[i];
    }
  }

  int getNbOutputs() const override {
    return nOuts;
  }

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
    assert(nbInputDims == 1);
    assert(inputs[0].nbDims == 3);
    int outChannelsSum = 0;
    for (int i = 0; i < nOuts; i++) {
      outChannelsSum += channels[i];
    }
    assert(inputs[0].d[0] == outChannelsSum);
    return Dims3{channels[index], inputs[0].d[1], inputs[0].d[2]};
  }

  void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    int maxBatchSize) override {

    assert(maxBatchSize == 1);
    h = inputDims[0].d[1];
    w = inputDims[0].d[2];
  }

  int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace,
    cudaStream_t stream) override {

    int runningChannels = 0;
    for (int i = 0; i < nOuts; i++) {
      auto inputSlice = static_cast<const uint32_t*>(inputs[0]) + runningChannels * h * w;
      checkCUDA(cudaMemcpyAsync(outputs[i], inputSlice, sizeof(uint32_t) * channels[i] * h * w,
        cudaMemcpyDeviceToDevice, stream));
      runningChannels += channels[i];
    }
    return 0;
  }

  int initialize() override { return 0; }
  void terminate() override {}
  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
  size_t getSerializationSize() override { return 0; }
  void serialize(void *buffer) override {}
private:
  int nOuts, h, w;
  int channels[MAX_NUM_OUTPUTS];
};

class Logger : public ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

static Logger gLogger;

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS); // %-400.400s
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

} gProfiler;

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    checkCUDA(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

void Graph::buildTRTNetworkHelper(INetworkDefinition *network, std::map<Edge, ITensor *, EdgeCompare>& outputs, Edge edge) {
  if (outputs.find(edge) != outputs.end()) {
    return;
  }

  std::set<Edge, EdgeCompare> inList = inEdges[edge.op];
  std::vector<ITensor *> inputs;
  for (auto it = inList.begin(); it != inList.end(); it++) {
    if (it->op.guid > 0) {
      buildTRTNetworkHelper(network, outputs, *it);
      inputs.push_back(outputs[*it]);
    }
  }
  if (inputs.size() == 0) {
    Tensor input = edge.op.ptr->inputs[0];
    if (input.numDim == 4) {
      // CNNs: NCHW
      assert(input.dim[0] == 1);
      char name[255];
      sprintf(name, "in%zd", edge.op.guid);
      ITensor *trt_input = network->addInput(name, DataType::kFLOAT,
        Dims3{input.dim[1], input.dim[2], input.dim[3]});
      outputs[edge] = trt_input;
      return;
    } else if (input.numDim == 3) {
      // RNNs: XNC
      assert(input.dim[1] == 1);
      char name[255];
      sprintf(name, "in%zd", edge.op.guid);
      ITensor *trt_input = network->addInput(name, DataType::kFLOAT,
        Dims3{input.dim[0], input.dim[1], input.dim[2]});
      outputs[edge] = trt_input;
      return;
    } else {
      assert(false);
    }
  }

  switch (edge.op.ptr->type) {
    case OpBase::OP_CONV2D:
    {
      assert(inputs.size() == 1);
      assert(inputs[0]->getDimensions().nbDims == 3);
      Conv2D* conv = (Conv2D*) edge.op.ptr;
      int inputC = inputs[0]->getDimensions().d[0];
      int numWeights = conv->kernelH * conv->kernelW * conv->outputC * inputC;
      auto trt_conv = network->addConvolution(*inputs[0], conv->outputC, DimsHW{conv->kernelH, conv->kernelW},
        (Weights) {DataType::kFLOAT, malloc(sizeof(uint32_t) * numWeights), numWeights}, // TODO memory leak
        (Weights) {DataType::kFLOAT, nullptr, 0});
      char name[255];
      sprintf(name, "conv%zd:%dx%d/%dx%d/%d/%d",
        edge.op.guid, conv->kernelH, conv->kernelW, conv->strideH, conv->strideW, inputC, conv->outputC);
      trt_conv->setName(name);
      trt_conv->setStride(DimsHW{conv->strideH, conv->strideW});
      trt_conv->setPadding(DimsHW{conv->padH, conv->padW});
      outputs[edge] = trt_conv->getOutput(0);
      break;
    }
    case OpBase::OP_POOL2D_MAX:
    case OpBase::OP_POOL2D_AVG:
    {
      assert(inputs.size() == 1);
      Pool2D* pool = (Pool2D*) edge.op.ptr;
      auto trt_pool = network->addPooling(*inputs[0],
        pool->type == OpBase::OP_POOL2D_MAX ? PoolingType::kMAX : PoolingType::kAVERAGE,
        DimsHW{pool->kernelH, pool->kernelW});
      trt_pool->setStride(DimsHW{pool->strideH, pool->strideW});
      trt_pool->setPadding(DimsHW{pool->padH, pool->padW});
      outputs[edge] = trt_pool->getOutput(0);
      break;
    }
    case OpBase::OP_RELU:
    {
      assert(inputs.size() == 1);
      outputs[edge] = network->addActivation(*inputs[0], ActivationType::kRELU)->getOutput(0);
      break;
    }
    case OpBase::OP_SIGMOID:
    {
      assert(inputs.size() == 1);
      outputs[edge] = network->addActivation(*inputs[0], ActivationType::kSIGMOID)->getOutput(0);
      break;
    }
    case OpBase::OP_BATCHNORM:
    {
      assert(inputs.size() == 1);
      float scale_param = 5.0f;
      float shift_param = 1.0f;
      outputs[edge] = network->addScale(*inputs[0], ScaleMode::kUNIFORM,
        (Weights) {DataType::kFLOAT, &shift_param, 1}, (Weights) {DataType::kFLOAT, &scale_param, 1},
        (Weights) {DataType::kFLOAT, nullptr, 0})->getOutput(0);
      break;
    }
    case OpBase::OP_SPLIT:
    {
      assert(inputs.size() == 1);
      Split *split = (Split *) edge.op.ptr;
      SplitPlugin *trt_split_plugin = new SplitPlugin(edge.op.ptr->numOutputs, split->channels); // TODO memory leak
      auto trt_split_layer = network->addPlugin(&inputs[0], 1, *trt_split_plugin);
      for (int i = 0; i < trt_split_layer->getNbOutputs(); i++) {
        outputs[(Edge) {i, edge.op}] = trt_split_layer->getOutput(i);
      }
      break;
    }
    case OpBase::OP_EW_ADD:
    case OpBase::OP_EW_MUL:
    {
      assert(inputs.size() == 2);
      outputs[edge] = network->addElementWise(*inputs[0], *inputs[1],
        edge.op.ptr->type == OpBase::OP_EW_ADD ? ElementWiseOperation::kSUM : ElementWiseOperation::kPROD)->getOutput(0);
      break;
    }
    case OpBase::OP_MATMUL:
    {
      assert(inputs.size() == 1);
      Matmul *matmul = (Matmul *) edge.op.ptr;
      int numParams = matmul->outputC;
      for (int i = 0; i < inputs[0]->getDimensions().nbDims; i++) {
        numParams *= inputs[0]->getDimensions().d[i];
      }
      auto trt_fc = network->addFullyConnected(*inputs[0], matmul->outputC,
        (Weights) {DataType::kFLOAT, malloc(sizeof(uint32_t) * numParams), numParams}, // TODO memory leak
        (Weights) {DataType::kFLOAT, nullptr, 0});
      if (matmul->actiMode != OpBase::AC_MODE_NONE) {
        ActivationType at = matmul->actiMode == OpBase::AC_MODE_RELU ? ActivationType::kRELU :
          matmul->actiMode == OpBase::AC_MODE_SIGMOID ? ActivationType::kSIGMOID : ActivationType::kTANH;
        outputs[edge] = network->addActivation(*trt_fc->getOutput(0), at)->getOutput(0);
      } else {
        outputs[edge] = trt_fc->getOutput(0);
      }
      break;
    }
    case OpBase::OP_NOOP:
    {
      assert(inputs.size() == 1);
      outputs[edge] = inputs[0];
      break;
    }
    case OpBase::OP_CONCAT:
    {
      assert(inputs.size() > 1);
      for (int i = 0; i < inputs.size(); i++) {
        assert(inputs[i]->getDimensions().nbDims == 3);
        if (i > 0) {
          assert(inputs[i]->getDimensions().d[1] == inputs[i - 1]->getDimensions().d[1] &&
            inputs[i]->getDimensions().d[2] == inputs[i - 1]->getDimensions().d[2]);
        }
      }
      outputs[edge] = network->addConcatenation(&inputs[0], inputs.size())->getOutput(0);
      break;
    }
    default:
      assert(false);
  }
}

void Graph::buildTRTNetwork(INetworkDefinition *network) {
  std::map<Edge, ITensor *, EdgeCompare> outputs;
  bool found_output = false;
  for (auto it = outEdges.begin(); it != outEdges.end(); it++) {
    if (it->second.size() == 0) {
      //assert(!found_output);
      assert(it->first.ptr->numOutputs == 1);
      found_output = true;
      Edge outEdge(0, it->first);
      buildTRTNetworkHelper(network, outputs, outEdge);
      network->markOutput(*outputs[outEdge]);
    }
  }
}

void runGraphTRT(Graph *graph) {
  IBuilder* builder = createInferBuilder(gLogger);
  INetworkDefinition* network = builder->createNetwork();
  graph->buildTRTNetwork(network);
  IRuntime* runtime = createInferRuntime(gLogger);

  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1 << 30);

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  network->destroy();
  builder->destroy();

  IExecutionContext* context = engine->createExecutionContext();
  context->setProfiler(&gProfiler);
  int batchSize = 1;

  int nbBindings = engine->getNbBindings();
  //assert(nbBindings == 2);

  std::vector<void*> buffers(nbBindings);

  for (int i = 0; i < nbBindings; ++i) {
    Dims dims = engine->getBindingDimensions(i);
    assert(dims.nbDims == 3);
    int64_t v = 1;
    for (int j = 0; j < dims.nbDims; j++) {
      v *= dims.d[j];
    }
    buffers[i] = safeCudaMalloc(sizeof(uint32_t) * v);
  }

  int numberRun = TIMING_ITERATIONS;
  float total = 0, ms;
  for (int run = 0; run < numberRun; run++) {
      auto t_start = std::chrono::high_resolution_clock::now();
      context->execute(batchSize, &buffers[0]);
      auto t_end = std::chrono::high_resolution_clock::now();
      ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
      total += ms;
  }

  total /= numberRun;
  std::cout << "Optimized Graph on TensorRT:" << std::endl;
  std::cout << "    Average over " << numberRun << " runs is " << total << " ms." << std::endl;

  for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx) {
    checkCUDA(cudaFree(buffers[bindingIdx]));
  }

  context->destroy();
  engine->destroy();
  //gProfiler.printLayerTimes();
}

#endif

/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _CNN_OPS_H_
#define _CNN_OPS_H_

#ifdef USE_CUDNN
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef TRT
#include "NvInfer.h"
#include "NvUtils.h"

using namespace nvinfer1;
#endif

#ifdef USE_MKL
#include "mkl_dnn.h"
#endif

#include <cassert>
#include <map>
#include <array>
#include <vector>
#include <set>
#include <list>
#include <iostream>
#include <fstream>
using namespace std;

#define MAX_DIM 4
#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6
#define BATCH_SIZE 1
#define MAX_TENSOR_SIZE 128 * 1024 * 1024 // 128MB
#define REPEAT_TIMES 8
#define CONCAT_KEY_LENGTH (MAX_NUM_INPUTS + 5)
#define SPLIT_KEY_LENGTH (MAX_NUM_OUTPUTS + 4)
const size_t WORK_SPACE_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2GB
typedef float DATATYPE;

class Model;
class OpBase;
struct Op {
  Op(void);
  bool operator==(const Op& b) const {
    if (guid != b.guid) return false;
    if (ptr != b.ptr) return false;
    return true;
  }
  bool operator<(const Op& b) const {
    if (guid != b.guid) return guid < b.guid;
    if (ptr != b.ptr) return ptr < b.ptr;
    return true;
  }
  size_t guid;
  OpBase* ptr;
};

struct Edge {
  Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx);
  int srcIdx, dstIdx;
  Op srcOp, dstOp;
};

struct EdgeCompare {
  bool operator()(const Edge& a, const Edge& b) const {
    if (!(a.srcOp == b.srcOp)) return a.srcOp < b.srcOp;
    if (!(a.dstOp == b.dstOp)) return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
    return false;
  };
};

struct OpCompare {
  bool operator()(const Op& a, const Op& b) const {
    if (a.guid != b.guid) return a.guid < b.guid;
    return a.ptr < b.ptr;
  };
};

struct Tensor {
  Tensor(void)
  : numDim(0), idx(0), op(), ptr(NULL) {}
  //bool operator==(const Tensor& b);
  int numDim, dim[MAX_DIM];
  int idx; // idx is used for Ops with multiple outputs (e.g., split)
  Op op;
  DATATYPE* ptr;
};

enum PMParameter {
  PM_OP_TYPE,   	// AnyOp
  PM_NUM_INPUTS,	// AnyOp
  PM_NUM_OUTPUTS,	// AnyOp
  PM_KERNEL_H,		// Conv2D, Pool2D
  PM_KERNEL_W,		// Conv2D, Pool2D
  PM_STRIDE_H,		// Conv2D, Pool2D
  PM_STRIDE_W,		// Conv2D, Pool2D
  PM_PAD_H,		// Conv2D, Pool2D
  PM_PAD_W,		// Conv2D, Pool2D
  PM_RELU,		// Conv2D, Pool2D
};

enum TNParameter {
  IN_0 = 100,
  IN_1 = 101,
  IN_2 = 102,
  IN_3 = 103,
  IN_4 = 104,
  IN_5 = 105,
  OU_0 = 200,
  OU_1 = 201,
  OU_2 = 202,
  OU_3 = 203,
  OU_4 = 204,
  OU_5 = 205,
};

enum DIMParameter {
  DIM_0 = 300,
  DIM_1 = 301,
  DIM_2 = 302,
  DIM_3 = 303,
  DIM_ND = 310,
};

class OpBase {
public:
  enum OpType {
    OP_NOOP,
    OP_ANY,
    OP_CONV2D,
    OP_LINEAR,
    OP_POOL2D_MAX,
    OP_POOL2D_AVG,
    OP_RELU,
    OP_SIGMOID,
    OP_BATCHNORM,
    OP_CONCAT,
    OP_SPLIT,
    // RNN operators
    OP_EW_ADD,
    OP_EW_MUL,
    OP_MATMUL,
    OP_SCALARMUL,
    OP_ENLARGECONV,
  };
  enum ActiMode {
    AC_MODE_NONE,
    AC_MODE_SIGMOID,
    AC_MODE_RELU,
    AC_MODE_TANH,
  };
  enum PaddingMode {
    PD_MODE_SAME,
    PD_MODE_VALID,
  };

  OpBase(Tensor input, Model* _model, OpType _type);
  OpBase(Tensor input0, Tensor input1, Model* _model, OpType _type);
  OpBase(int n, Tensor* inputs, Model* _model, OpType _type);
  virtual bool get_input_parameter(TNParameter, DIMParameter, int*);
  virtual bool get_parameter(PMParameter, int*) = 0;
  virtual void forward(void) = 0;
  virtual void map(void) = 0;
  virtual void unmap(void) = 0;
  virtual void collect_costs(float& exe_time, float& flops,
                             float& mem_acc, int& num_kernels) = 0;
public:
  Tensor inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
  int numInputs, numOutputs;
  OpType type;
  Model *model;
  float runtime;
};

class Graph {
public:
  Graph(Model *_model);
  void add_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx);
  bool has_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx);
  void export_to_file(std::string file_name);
  // This conv2d will create a weight tensor
  Tensor conv2d(Tensor _input, int _outputC, int _kernelH, int _kernelW,
                int _strideH, int _strideW, int _padH, int _padW,
                bool _relu = false);
  Tensor conv2d(Tensor _input, Tensor _weight, int _strideH, int _strideW,
                int _padH, int _padW, bool _relu = false);
  Tensor fc(Tensor _input, int _outputC,
            OpBase::ActiMode _actiMode = OpBase::AC_MODE_NONE);
  Tensor matmul(Tensor _input, Tensor _weight,
                OpBase::ActiMode _actiMode = OpBase::AC_MODE_NONE);
  Tensor pool2d_max(Tensor _input, int _kernelH, int _kernelW,
                    int _strideH, int _strideW, int _padH, int _padW,
                    bool _relu = false);
  Tensor pool2d_avg(Tensor _input, int _kernelH, int _kernelW,
                    int _strideH, int _strideW, int _padH, int _padW,
                    bool _relu = false);
  Tensor relu(Tensor _input, bool _inPlace = true);
  Tensor sigmoid(Tensor _input, bool _inPlace = true);
  Tensor batchnorm(Tensor _input);
  Tensor concat(int n, Tensor* _inputs);
  void split(Tensor _input, int c1, int c2, Tensor* outputs);
  void split(Tensor _input, int num, int* channels, Tensor* outputs);
  Tensor noop(Tensor _input);
  Tensor element(OpBase::OpType type, Tensor _t1, Tensor _t2);
  size_t num_in_edges(Op op);
  size_t num_out_edges(Op op);
  size_t hash(void);
  void print(void);
  bool check_correctness(void);
  float total_cost(void);
  float run(Model *model);
  void print_costs(void);
#ifdef TRT
  void buildTRTNetwork(INetworkDefinition *network);
private:
  void buildTRTNetworkHelper(INetworkDefinition *network, std::map<Edge, ITensor *, EdgeCompare>& outputs, Edge edge);
#endif
  void export_op(ofstream &file_stream, Op &op);
public:
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare> inEdges, outEdges;
  Model *model;
  float totalCost;
};


class Conv2D : public OpBase {
public:
  Conv2D(Model* _model, Tensor _input, Tensor _weight,
         int _strideH, int _strideW,
         int _padH, int _padW, bool _relu);
  ~Conv2D(void);
  void forward(void);
  void map(void);
  void unmap(void);
  bool get_parameter(PMParameter para, int*);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
#ifdef USE_CUDNN
  cudnnConvolutionFwdAlgo_t selectForwardAlgorithm(void);
#endif
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
#endif
#ifdef USE_MKL
  std::vector<dnnPrimitive_t> compList;
  std::vector<std::array<void*, dnnResourceNumber>> rsrcList;
  int fwdAlgo; // Placeholder, should never use this in mkl.
#endif
  int strideH, strideW, padH, padW;
  bool relu;
  void *biasPtr;
};

class Matmul : public OpBase {
public:
  Matmul(Model* _model, Tensor _input, Tensor _weight,
         ActiMode _actiMode);
  ~Matmul(void);
  void forward(void);
  void map(void);
  void unmap(void);
  bool get_parameter(PMParameter para, int*);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int outputC;
  ActiMode actiMode;
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
};

class Pool2D : public OpBase {
public:
  Pool2D(Model* _model, Tensor _input, OpType _type,
         int _kernelH, int _kernelW,
         int _strideH, int _strideW,
         int _padH, int _padW, bool _relu);
  ~Pool2D(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
#endif
#ifdef USE_MKL
  std::vector<dnnPrimitive_t> compList;
  std::vector<std::array<void*, dnnResourceNumber>> rsrcList;
#endif
  int kernelH, kernelW, strideH, strideW, padH, padW;
  bool relu;
};

class Activation : public OpBase {
public:
  Activation(Model* _model, Tensor _input, OpType _type, bool _inPlace);
  ~Activation(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t actiDesc;
#endif
  bool inPlace;
};

class BatchNorm : public OpBase {
public:
  BatchNorm(Model* _model, Tensor _input);
  ~BatchNorm(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
#endif
#ifdef USE_MKL
  dnnPrimitive_t comp;
  std::array<void*, dnnResourceNumber> rsrc;
#endif
  DATATYPE *biasPtr, *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar;
};

class Concat : public OpBase {
public:
  Concat(Model* _model, int n, Tensor* _inputs, bool* _needCopy);
  ~Concat(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  bool needCopy[MAX_NUM_INPUTS];
};

class Split : public OpBase {
public:
  Split(Model* _model, Tensor _input, int n, int* channels);
  ~Split(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
  int channels[MAX_NUM_OUTPUTS];
};

class NoOp : public OpBase {
public:
  NoOp(Model* _model, Tensor _input);
  ~NoOp(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
};

class Element : public OpBase {
public:
  Element(Model* _model, OpType _type, Tensor _t1, Tensor _t2);
  ~Element(void);
  bool get_parameter(PMParameter para, int*);
  void forward(void);
  void map(void);
  void unmap(void);
  void collect_costs(float& exe_time, float& flops, float& mem_acc, int& num_kernels);
public:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t inputTensor;
  cudnnOpTensorDescriptor_t opDesc;
#endif
};

// keys are (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,              
//           strideH, strideW, padH, padW, relu)
struct Conv2DKey {
  Conv2DKey(Tensor, Tensor, int, int, int, int, bool);
  int keys[12];
};

struct Conv2DCompare {
  bool operator()(const Conv2DKey& a, const Conv2DKey& b) const {
    for (int i = 0; i < 12; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (inputX, inputN, inputC, outputC, acti)
//
struct MatmulKey {
  MatmulKey(Tensor, Tensor, OpBase::ActiMode);
  int keys[4];
};

struct MatmulCompare {
  bool operator()(const MatmulKey& a, const MatmulKey& b) const {
    for (int i = 0; i < 4; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (inputN, inputC, inputH, inputW, kernelH, kernelW,              
//           strideH, strideW, padH, padW, relu, type)
struct Pool2DKey {
  Pool2DKey(Tensor, OpBase::OpType, int, int, int, int, int, int, bool);
  int keys[12];
};

struct Pool2DCompare {
  bool operator()(const Pool2DKey& a, const Pool2DKey& b) const {
    for (int i = 0; i < 12; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (inputN, inputC, inputH, inputW, type, inPlace)
struct ActivationKey {
  ActivationKey(Tensor, OpBase::OpType, bool);
  int keys[6];
};

struct ActivationCompare {
  bool operator()(const ActivationKey& a, const ActivationKey& b) const {
    for (int i = 0; i < 6; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, inputC, inputH, inputW)
struct BatchNormKey {
  BatchNormKey(Tensor);
  int keys[4];
};

struct BatchNormCompare {
  bool operator()(const BatchNormKey& a, const BatchNormKey& b) const {
    for (int i = 0; i < 4; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// keys are (n, inputN, inputH, inputW, bitmask(needCopy), inputC[0...n-1])
struct ConcatKey {
  ConcatKey(int, Tensor*, bool*);
  int keys[CONCAT_KEY_LENGTH];
};

struct ConcatCompare {
  bool operator()(const ConcatKey& a, const ConcatKey& b) const {
    for (int i = 0; i < CONCAT_KEY_LENGTH; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, input H, inputW, n, outputC[0...,n-1])
struct SplitKey {
  SplitKey(Tensor input, int n, int* channels);
  int keys[SPLIT_KEY_LENGTH];
};

struct SplitCompare {
  bool operator()(const SplitKey& a, const SplitKey& b) const {
    for (int i = 0; i < SPLIT_KEY_LENGTH; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, inputC, inputH, inputW)
struct NoopKey {
  NoopKey(Tensor input);
  int keys[4];
};

struct NoopCompare {
  bool operator()(const NoopKey& a, const NoopKey& b) const {
    for (int i = 0; i < 4; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

// key is (inputN, inputC, inputH, inputW, type)
struct ElementKey {
  ElementKey(Tensor t, OpBase::OpType type);
  int keys[5];
};

struct ElementCompare {
  bool operator()(const ElementKey& a, const ElementKey& b) const {
    for (int i = 0; i < 5; i++)
      if (a.keys[i] != b.keys[i])
        return a.keys[i] < b.keys[i];
    return false;
  };
};

class Model {
public:
  Model(bool);
  Op get_or_create_conv2d(Tensor _input, Tensor _weight,
                          int _strideH, int _strideW,
                          int _padH, int _padW, bool _relu);
  Op get_or_create_matmul(Tensor _input, Tensor _weight,
                          OpBase::ActiMode _actimode);
  Op get_or_create_pool2d(Tensor _input, OpBase::OpType _type,
                          int _kernelH, int _kernelW,
                          int _strideH, int _strideW,
                          int _padH, int _padW, bool _relu);
  Op get_or_create_activation(Tensor _input, OpBase::OpType _type,
                              bool _inPlace);
  Op get_or_create_batchnorm(Tensor _input);
  Op get_or_create_concat(int n, Tensor* _inputs, bool* _needCopy);
  Op get_or_create_split(Tensor _input, int n, int* channels);
  Op get_or_create_noop(Tensor _input);
  Op get_or_create_element(OpBase::OpType type, Tensor t1, Tensor t2);
  void measure_conv2d_cost(Conv2D*);
  void measure_matmul_cost(Matmul*);
  void measure_pool2d_cost(Pool2D*);
  void measure_activation_cost(Activation*);
  void measure_batchnorm_cost(BatchNorm*);
  void measure_concat_cost(Concat*);
  void measure_split_cost(Split*);
  void measure_element_cost(Element*);
  void* allocate_memory(size_t size);
  float measure_oplist_runtime(const std::vector<OpBase*>& list);
public:
  bool isTraining;
  size_t global_unique_id;
  size_t workSpaceSize;
  void* workSpace;
#ifdef USE_CUDNN
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  // Note that actiDesc is set when we construct Model since
  // all relus are identical.
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudaEvent_t startEvent, endEvent;
  // variables for batch norm
  cudnnTensorDescriptor_t scaleTensor;
  // variables for element wise
  cudnnOpTensorDescriptor_t opDesc;
#endif
  std::map<Conv2DKey, Conv2D*, Conv2DCompare> conv2d;
  std::map<MatmulKey, Matmul*, MatmulCompare> matmul;
  std::map<Pool2DKey, Pool2D*, Pool2DCompare> pool2d;
  std::map<ActivationKey, Activation*, ActivationCompare> activation;
  std::map<BatchNormKey, BatchNorm*, BatchNormCompare> batchnorm;
  std::map<ConcatKey, Concat*, ConcatCompare> concat;
  std::map<SplitKey, Split*, SplitCompare> split;
  std::map<NoopKey, NoOp*, NoopCompare> noop;
  std::map<ElementKey, Element*, ElementCompare> element;
  DATATYPE *inputPtr, *biasPtr, *outputPtr, *filterPtr;
  // variables for batch norm
  DATATYPE *scalePtr, *runningMean, *runningVar, *saveMean, *saveVar;
};

#endif

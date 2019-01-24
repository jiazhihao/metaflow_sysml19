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

#include "ops.h"
#include <iostream>
#include <fstream>
using namespace std;

/*
bool Op::operator==(const Op& b)
{
  if (guid != b.guid) return false;
  return (ptr == b.ptr);
}

bool Op::operator<(const Op& b)
{
  if (guid != b.guid) return guid < b.guid;
  return ptr < b.ptr;
}
*/

Op::Op(void)
{
  guid = 0;
  ptr = NULL;
}

Edge::Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

/*
bool Tensor::operator==(const Tensor& b)
{
  if (numDim != b.numDim) return false;
  for (int i = 0; i < numDim; i++)
    if (dim[i] != b.dim[i]) return false;
  if (idx != b.idx) return false;
  if (op.guid != b.op.guid) return false;
  return true;
}
*/

OpBase::OpBase(Tensor _input, Model* _model, OpType _type)
: model(_model), type(_type), numInputs(1), runtime(0.0f)
{
  inputs[0] = _input;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(Tensor _input0, Tensor _input1, Model* _model, OpType _type)
: model(_model), type(_type), numInputs(2), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(int n, Tensor* _inputs, Model* _model, OpType _type)
: model(_model), type(_type), numInputs(n), runtime(0.0f)
{
  assert(n <= MAX_NUM_INPUTS);
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

bool OpBase::get_input_parameter(TNParameter tnp, DIMParameter dim, int* value)
{
  int inputIdx = 0, dimIdx = 0;
  switch (tnp) {
    case IN_5:
      inputIdx++;
    case IN_4:
      inputIdx++;
    case IN_3:
      inputIdx++;
    case IN_2:
      inputIdx++;
    case IN_1:
      inputIdx++;
    case IN_0:
      break;
    default:
      return false;
  }
  if (inputIdx >= numInputs) return false;
  switch (dim) {
    case DIM_3:
      dimIdx ++;
    case DIM_2:
      dimIdx ++;
    case DIM_1:
      dimIdx ++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = inputs[inputIdx].numDim;
      return true;
    default:
      return false;
  }
  if (dimIdx >= inputs[inputIdx].numDim) return false;
  *value = inputs[inputIdx].dim[dimIdx];
  return true;
}

Graph::Graph(Model *_model)
: model(_model), totalCost(-1.0f)
{
  //size_t inputSize = sizeof(DATATYPE) * n * c * h * w;
  //checkCUDA(cudaMalloc(&input.ptr, inputSize));
  //printf("Initialize a graph\n");
}

void Graph::add_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx)
{
  if (inEdges.find(dstOp) == inEdges.end()) {
    inEdges[dstOp];
  }
  if (outEdges.find(srcOp) == outEdges.end()) {
    outEdges[srcOp];
  }
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  inEdges[dstOp].insert(e);
  outEdges[srcOp].insert(e);
}

// We do this in topological order because it will be easier to parse on
// the other end
void Graph::export_to_file(std::string file_name)
{
  ofstream export_fs;
  export_fs.open(file_name.c_str());
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > 0) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
    {
      opList.push_back(it->first);
    }
  }
  int i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    export_op(export_fs, op);

    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) opList.push_back(it2->dstOp);
    }
  }
  export_fs.close();
  assert(opList.size() == inEdges.size());
}

/* Exports an operator with the following format:
 * guid
 * type
 * parameters (comma separated and type dependent)
 * dependencies (comma separated list of other ops)
 */
void Graph::export_op(ofstream &file_stream, Op &op)
{
  file_stream << op.guid << std::endl;

  file_stream << op.ptr->type << std::endl;

  std::string deps_string;
  std::set<Edge, EdgeCompare> inList = inEdges[op];
  std::set<Edge, EdgeCompare>::const_iterator it;
  int i = 0;
  for (it = inList.begin(); it != inList.end(); it++) {
    deps_string += std::to_string(it->srcOp.guid);
    deps_string += ':';
    deps_string += std::to_string(it->srcIdx);
    deps_string += ',';
    i++;
  }
  if (deps_string.size() > 0)
  {
    deps_string = deps_string.substr(0, deps_string.size()-1);
  }
  file_stream << deps_string.c_str() << std::endl;

  switch (op.ptr->type) {
    case OpBase::OP_CONV2D:
    { 
      Conv2D* conv = (Conv2D*) op.ptr;
      Tensor t = conv->inputs[0];
      Tensor w = conv->inputs[1];

      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3] << ','; // 3
      file_stream << w.dim[0] << ','; // 4
      file_stream << w.dim[2] << ','; // 5
      file_stream << w.dim[3] << ','; // 6
      file_stream << conv->strideH << ','; // 7
      file_stream << conv->strideW << ','; // 8
      file_stream << conv->padH << ','; // 9
      file_stream << conv->padW << ','; // 10
      file_stream << conv->relu; // 11
      break;
    }
    case OpBase::OP_POOL2D_MAX:
    case OpBase::OP_POOL2D_AVG:
    {
      Pool2D* pool = (Pool2D*) op.ptr;
      Tensor t = pool->inputs[0];

      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3] << ','; // 3
      file_stream << pool->type << ','; // 4
      file_stream << pool->kernelH << ','; // 5
      file_stream << pool->kernelW << ','; // 6
      file_stream << pool->strideH << ','; // 7
      file_stream << pool->strideW << ','; // 8
      file_stream << pool->padH << ','; // 9
      file_stream << pool->padW << ','; // 10
      file_stream << pool->relu; // 11
      break;
    }
    case OpBase::OP_SPLIT:
    {
      Split* split = (Split*) op.ptr;
      for (int i = 0; i < split->numOutputs; i++)
      {
        file_stream << split->channels[i];
        if (i < split->numOutputs - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OpBase::OP_CONCAT:
    case OpBase::OP_EW_ADD:
    case OpBase::OP_EW_MUL:
    case OpBase::OP_RELU:
    case OpBase::OP_SIGMOID:
    case OpBase::OP_BATCHNORM:
    case OpBase::OP_NOOP:
    {
      Tensor t = op.ptr->inputs[0];
      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3]; // 3
      break;
    }
    case OpBase::OP_MATMUL: // This doesn't seem to be implemented in run either
    {
      Matmul* matmul = (Matmul*) op.ptr;
      Tensor t = matmul->inputs[0];

      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << matmul->outputC << ','; // 3
      file_stream << matmul->actiMode; // 4
      break;
    }
    default:
      assert(false);
  }
  file_stream << std::endl;
}

size_t Graph::num_in_edges(Op op)
{
  return inEdges[op].size();
}

size_t Graph::num_out_edges(Op op)
{
  return outEdges[op].size();
}

bool Graph::has_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx)
{
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

size_t Graph::hash(void)
{
  size_t total = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    size_t my = 17 * 31 + (size_t)(it->first.ptr);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      my = my * 31 + std::hash<size_t>()((size_t)(e.srcOp.ptr));
      my = my * 31 + std::hash<int>()(e.srcIdx);
      my = my * 31 + std::hash<int>()(e.dstIdx);
    }
    total += my;
  }
  return total;
}

void Graph::print(void)
{
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first.guid == 0) continue;
    printf("	guid(%d) type(%d) runtime(%.4lf) op_ptr(%x): ", it->first.guid, it->first.ptr->type, it->first.ptr->runtime, it->first.ptr);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      printf(" inEdge(guid(%d) idx(%d))", e.srcOp.guid, e.srcIdx);
    }
    printf("\n");
  }
}

bool Graph::check_correctness(void)
{
  bool okay = true;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)) okay = false;
    }
  }
  return okay;
}

float Graph::total_cost(void)
{
  if (totalCost > 0) return totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  float total = 0.0f;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first.ptr != NULL) total += it->first.ptr->runtime;
  }
  totalCost = total;
  return total;
}

float Graph::run(Model* model)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  std::vector<OpBase*> opBaseList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > 0) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0) opList.push_back(it->first);
  }
  int i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare> inList = inEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    assert(inList.size() > 0);
    OpBase* opPtr = NULL;
    // Step 1: prepare inputs
    Tensor inputs[MAX_NUM_INPUTS];
    if (op.ptr->type == OpBase::OP_NOOP) {
      assert(inList.size() == 1);
      Edge e = *inList.begin();
      assert(e.srcOp.ptr == NULL); // NoOp's input must not be any Op
      Tensor t = op.ptr->inputs[0];
      size_t size = sizeof(DATATYPE);
      for (int j = 0; j < t.numDim; j++)
        size *= t.dim[j];
      t.ptr = (DATATYPE*) model->allocate_memory(size);
      inputs[0] = t;
    } else {
      for (it2 = inList.begin(); it2 != inList.end(); it2++) {
        int idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == it2->srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[it2->dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[it2->dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
    // Step 2: create Ops
    switch (op.ptr->type) {
      case OpBase::OP_CONV2D:
      {
        Conv2D* conv = (Conv2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Conv2D(model, inputs[0], inputs[1],
                           conv->strideH, conv->strideW,
                           conv->padH, conv->padW, conv->relu);
#ifdef USE_CUDNN
        ((Conv2D*)opPtr)->fwdAlgo = conv->fwdAlgo;
#endif
        break;
      }
      case OpBase::OP_MATMUL:
      {
        Matmul* matmul = (Matmul*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Matmul(model, inputs[0], inputs[1], matmul->actiMode);
        break;
      }
      case OpBase::OP_EW_ADD:
      case OpBase::OP_EW_MUL:
      {
        Element* element = (Element*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Element(model, op.ptr->type, inputs[0], inputs[1]);
        break;
      }
      case OpBase::OP_POOL2D_MAX:
      case OpBase::OP_POOL2D_AVG:
      {
        Pool2D* pool = (Pool2D*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Pool2D(model, inputs[0], pool->type,
                           pool->kernelH, pool->kernelW,
                           pool->strideH, pool->strideW,
                           pool->padH, pool->padW, pool->relu);
        break;
      }
      case OpBase::OP_RELU:
      case OpBase::OP_SIGMOID:
      {
        Activation* act = (Activation*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Activation(model, inputs[0], act->type, act->inPlace);
        break;
      }
      case OpBase::OP_BATCHNORM:
      {
        opPtr = new BatchNorm(model, inputs[0]);
        break;
      }
      case OpBase::OP_SPLIT:
      {
        Split* split = (Split*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Split(model, inputs[0], split->numOutputs, split->channels);
        break;
      }
      case OpBase::OP_NOOP:
      {
        assert(inList.size() == 1);
        opPtr = new NoOp(model, inputs[0]);
        break;
      }
      case OpBase::OP_CONCAT:
      {
        Concat* concat = (Concat*) op.ptr;
        opPtr = new Concat(model, inList.size(), inputs, concat->needCopy);
        break;
      }
      default:
        printf("op.type = %d\n", op.ptr->type);
        assert(false);
    }
    // Step 3: map new Op
    opPtr->map();
    opBaseList.push_back(opPtr);
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) opList.push_back(it2->dstOp);
    }
  }
  assert(opList.size() == inEdges.size());
  assert(opList.size() == opBaseList.size());

  return model->measure_oplist_runtime(opBaseList);
}

void Graph::print_costs(void)
{
  float exe_time = 0, flops = 0, mem_acc = 0;
  int num_kernels = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++)
    it->first.ptr->collect_costs(exe_time, flops, mem_acc, num_kernels);
  printf("Cost metrics: exe_time(%.4lf) flops(%.4lf) "
         "memory_access(%.4lf) kernel_launches(%d)\n",
         exe_time, flops / 1024.0 / 1024.0 / 1024.0,
         mem_acc * 4.0 / 1024.0 / 1024.0, num_kernels);
}


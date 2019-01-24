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

void Graph::split(Tensor _input, int _num, int* _channels, Tensor* outputs)
{
  int n = _num, channels[MAX_NUM_OUTPUTS];
  for (int i = 0; i < n; i++)
    channels[i] = _channels[i];
  Op op = model->get_or_create_split(_input, n, channels);
  inEdges[op];
  outEdges[op];
  Edge in(_input.idx, _input.op), out(_input.idx, op);
  inEdges[op].insert(in);
  outEdges[_input.op].insert(out);
  for (int i = 0; i < n; i++) {
    outputs[i] = op.ptr->outputs[i];
    outputs[i].op = op;
  }
}

void Graph::split(Tensor _input, int c1, int c2, Tensor* outputs)
{
  int channels[2];
  channels[0] = c1;
  channels[1] = c2;
  Graph::split(_input, 2, channels, outputs);
}

//Tensor Graph::split(Tensor _input, int n, int* channels)
//{
//}

Op Model::get_or_create_split(Tensor _input, int n, int* channels)
{
  // key is (inputN, input H, inputW, outputC[0...,n-1]
  SplitKey key(_input, n, channels);
  Split* splitOp;
  if (split.find(key) != split.end()) {
    splitOp = split[key];
  } else {
    splitOp = new Split(this, _input, n, channels);
    measure_split_cost(splitOp);
    split[key] = splitOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = splitOp;
  return ret;
}

Split::Split(Model* _model, Tensor _input, int n, int* _channels)
  : OpBase(_input, model, OP_SPLIT)
{
  assert(n <= MAX_NUM_OUTPUTS);
  numOutputs = n;
  for (int i = 0; i < n; i++) {
    channels[i] = _channels[i];
    outputs[i].numDim = inputs[0].numDim;
    outputs[i].dim[0] = BATCH_SIZE;
    outputs[i].dim[1] = channels[i];
    for (int j = 2; j < outputs[i].numDim; j++)
      outputs[i].dim[j] = inputs[0].dim[j];
    outputs[i].idx = i;
  }
}

Split::~Split(void)
{}

bool Split::get_parameter(OpParameter para, int* value)
{
  switch (para) {
    case PM_OP_TYPE:
      *value = (int) type;
      return true;
    case PM_NUM_INPUTS:
      *value = numInputs;
      return true;
    case PM_NUM_OUTPUTS:
      *value = numOutputs;
      return true;
    default:
      return false;
  }
}

void Split::map(void)
{
  size_t offset = 0;
  for (int i = 0; i < numOutputs; i++) {
    outputs[i].ptr = (char*)inputs[0].ptr + offset;
    size_t size = sizeof(DATATYPE);
    for (int j = 0; j < outputs[i].numDim; j++)
      size *= outputs[i].dim[j];
    offset += size;
  }
}

void Split::unmap(void)
{}

void Split::forward(void)
{}

void Split::collect_costs(float& exe_time, float& flops,
                          float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += 0;
  flops += 0;
  mem_acc += 0;
  num_kernels += 0;
}

void Model::measure_split_cost(Split* split)
{
  // We assume split cost is zero
  split->runtime = 0;
  printf("measure[split]: cost(%.4lf)\n", split->runtime);
}

// key is (inputN, input H, inputW, n, outputC[0...,n-1]
SplitKey::SplitKey(Tensor input, int n, int* channels)
{
  keys[0] = input.dim[0];
  keys[1] = input.dim[2];
  keys[2] = input.dim[3];
  keys[3] = n;
  for (int i = 0; i < n; i++)
    keys[4 + i] = channels[i];
  for (int i = 4 + n; i < SPLIT_KEY_LENGTH; i++)
    keys[i] = 0;
}

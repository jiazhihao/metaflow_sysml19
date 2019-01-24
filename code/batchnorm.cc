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

Tensor Graph::batchnorm(Tensor _input)
{
  Op op = model->get_or_create_batchnorm(_input);
  add_edge(_input.op, op, _input.idx, 0);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_batchnorm(Tensor _input)
{
  // key is (inputN, inputC, inputH, inputW)
  BatchNormKey key(_input);
  BatchNorm* bnOp;
  if(batchnorm.find(key) != batchnorm.end()) {
    bnOp = batchnorm[key];
  } else {
    bnOp = new BatchNorm(this, _input);
    measure_batchnorm_cost(bnOp);
    batchnorm[key] = bnOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = bnOp;
  return ret;
}

BatchNorm::BatchNorm(Model* _model, Tensor _input)
: OpBase(_input, _model, OP_BATCHNORM)
{
  assert(_input.numDim == 4);
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

BatchNorm::~BatchNorm(void)
{}

bool BatchNorm::get_parameter(PMParameter para, int* value)
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

void BatchNorm::collect_costs(float& exe_time, float& flops,
                              float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  flops += outputSize * 2;
  mem_acc += inputSize;
  num_kernels += 1;
}

// key is (inputN, inputC, inputH, inputW)
BatchNormKey::BatchNormKey(Tensor _input)
{
  keys[0] = _input.dim[0];
  keys[1] = _input.dim[1];
  keys[2] = _input.dim[2];
  keys[3] = _input.dim[3];
}

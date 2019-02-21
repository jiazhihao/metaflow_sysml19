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

Tensor Graph::matmul(Tensor _input, int _outputC, OpBase::ActiMode acti)
{
  Op op = model->get_or_create_matmul(_input, _outputC, acti);
  inEdges[op];
  outEdges[op];
  Edge in(_input.idx, _input.op), out(_input.idx, op);
  inEdges[op].insert(in);
  outEdges[_input.op].insert(out);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_matmul(Tensor _input, int _outputC,
                               OpBase::ActiMode _acti)
{
  // key is (inputX, inputN, inputC, outputC, acti)
  MatmulKey key(_input, _outputC, _acti);
  Matmul* matmulOp;
  if (matmul.find(key) != matmul.end()) {
    matmulOp = matmul[key];
  } else {
    matmulOp = new Matmul(this, _input, _outputC, _acti);
    measure_matmul_cost(matmulOp);
    matmul[key] = matmulOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = matmulOp;
  return ret;
}

Matmul::Matmul(Model* _model, Tensor _input, int _outputC, ActiMode _actiMode)
: OpBase(_input, _model, OP_MATMUL), outputC(_outputC), actiMode(_actiMode)
{
  assert(_input.numDim == 3);
  int inputX = _input.dim[0];
  int inputN = _input.dim[1];
  int inputC = _input.dim[2];
  numOutputs = 1;
  outputs[0].numDim = 3;
  outputs[0].dim[0] = inputX;
  outputs[0].dim[1] = inputN;
  outputs[0].dim[2] = outputC;
  outputs[0].idx = 0;
}

Matmul::~Matmul(void)
{}

bool Matmul::get_parameter(OpParameter para, int* value)
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
    case PM_OUTPUT_C:
      *value = outputC;
      return true;
    case PM_ACTI:
      *value = actiMode;
      return true;
    default:
      return false;
  }
}

void Matmul::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  assert(inputs[0].numDim == 3);
  flops += outputSize * inputs[0].dim[2];
  mem_acc += inputSize;
  num_kernels += 1;
}

// key is (inputX, inputN, inputC, outputC, acti)
MatmulKey::MatmulKey(Tensor _input, int _outputC, OpBase::ActiMode _mode)
{
  assert(_input.numDim == 3);
  keys[0] = _input.dim[0];
  keys[1] = _input.dim[1];
  keys[2] = _input.dim[2];
  keys[3] = _outputC;
  keys[4] = (int)(_mode);
}


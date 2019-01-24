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

Tensor Graph::fc(Tensor _input, int _outputC, OpBase::ActiMode acti)
{
  Tensor weight;
  weight.numDim = 2;
  weight.dim[0] = _outputC;
  weight.dim[1] = _input.dim[1];
  weight.op.guid = 0;
  weight.op.ptr = NULL;
  weight.idx = 0;
  weight = noop(weight);
  return matmul(_input, weight, acti);
}

Tensor Graph::matmul(Tensor _input, Tensor _weight, OpBase::ActiMode acti)
{
  Op op = model->get_or_create_matmul(_input, _weight, acti);
  add_edge(_input.op, op, _input.idx, 0);
  add_edge(_weight.op, op, _weight.idx, 1);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_matmul(Tensor _input, Tensor _weight,
                               OpBase::ActiMode _acti)
{
  // key is (inputX, inputN, inputC, outputC, acti)
  MatmulKey key(_input, _weight, _acti);
  Matmul* matmulOp;
  if (matmul.find(key) != matmul.end()) {
    matmulOp = matmul[key];
  } else {
    matmulOp = new Matmul(this, _input, _weight, _acti);
    measure_matmul_cost(matmulOp);
    matmul[key] = matmulOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = matmulOp;
  return ret;
}

Matmul::Matmul(Model* _model, Tensor _input, Tensor _weight, ActiMode _actiMode)
: OpBase(_input, _weight, _model, OP_MATMUL), actiMode(_actiMode)
{
  assert(_input.numDim == 2);
  assert(_weight.numDim == 2);
  assert(_input.dim[1] == _weight.dim[1]);
  numOutputs = 1;
  outputs[0].numDim = 2;
  outputs[0].dim[0] = _input.dim[0];
  outputs[0].dim[1] = _weight.dim[0];
  outputs[0].idx = 0;
}

Matmul::~Matmul(void)
{}

bool Matmul::get_parameter(PMParameter para, int* value)
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
  assert(inputs[0].numDim == 2);
  flops += outputSize * inputs[0].dim[1];
  mem_acc += inputSize;
  num_kernels += 1;
}

// key is (inputN, inputC, outputC, acti)
MatmulKey::MatmulKey(Tensor _input, Tensor _weight, OpBase::ActiMode _mode)
{
  assert(_input.numDim == 2);
  keys[0] = _input.dim[0];
  keys[1] = _input.dim[1];
  keys[2] = _weight.dim[0];
  keys[3] = (int)(_mode);
}

// -------------------------------------------------------------
// MatmulTemp functions
// -------------------------------------------------------------
/*MatmulTemp::MatmulTemp(OpBase::ActiMode _mode)
: OpTemp(OpBase::MATMUL), mode(_mode), numInpts(2), numOutputs(1)
{}

MatmulTemp::~MatmulTemp(void)
{}

bool MatmulTemp::compute(int n, Tensor* inputs)
{
  if (n != 2 ) return false;
  if (inputs[0].numDim != 2 || inputs[1].numDim != 2) return false;
  if (inputs[0].dim[0] != BATCH_SIZE) return false;
  if (inputs[0].dim[1] != inputs[1].dim[1]) return false;
  outputs[0].numDim = 2;
  outputs[0].dim[0] = BATCH_SIZE;
  outputs[0].dim[1] = inputs[1].dim[0];
  int outputN = outputs[0].dim[0];
  int outputC = outputs[0].dim[1];
  int inputC = inputs[0].dim[1];
  for (int i = 0; i < outputN; i++)
    for (int j = 0; j < outputC; j++) {
      DATATYPE val = 0;
    }
  return true;
}
*/

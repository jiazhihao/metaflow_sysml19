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

Tensor Graph::relu(Tensor _input, bool _inPlace)
{
  Op op = model->get_or_create_activation(_input, OpBase::OP_RELU, _inPlace);
  inEdges[op];
  outEdges[op];
  Edge in(_input.idx, _input.op), out(_input.idx, op);
  inEdges[op].insert(in);
  outEdges[_input.op].insert(out);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Tensor Graph::sigmoid(Tensor _input, bool _inPlace)
{
  Op op = model->get_or_create_activation(_input, OpBase::OP_SIGMOID, _inPlace);
  inEdges[op];
  outEdges[op];
  Edge in(_input.idx, _input.op), out(_input.idx, op);
  inEdges[op].insert(in);
  outEdges[_input.op].insert(out);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_activation(Tensor _input, OpBase::OpType _type, bool _inPlace)
{
  // keys are (inputN, inputC, inputH, inputW, _type, _inPlace)
  ActivationKey key(_input, _type, _inPlace);
  Activation* actOp;
  if (activation.find(key) != activation.end()) {
    actOp = activation[key];
  } else {
    actOp = new Activation(this, _input, _type, _inPlace);
    measure_activation_cost(actOp);
    activation[key] = actOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = actOp;
  return ret;
}

Activation::Activation(Model* _model, Tensor _input, OpType _type, bool _inPlace)
: OpBase(_input, _model, _type), inPlace(_inPlace)
{
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

Activation::~Activation(void)
{
}

bool Activation::get_parameter(OpParameter para, int* value)
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

void Activation::collect_costs(float& exe_time, float& flops,
                              float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  if (type == OpBase::OP_RELU)
    flops += 0; // relu does not involve flops
  else
    flops += outputSize;
  mem_acc += inputSize;
  num_kernels += 1;
}

// keys are (inputN, inputC, inputH, inputW, _type, inPlace)
ActivationKey::ActivationKey(Tensor _input, OpBase::OpType _type, bool _inPlace)
{
  keys[0] = _input.dim[0];
  keys[1] = _input.dim[1];
  keys[2] = _input.dim[2];
  keys[3] = _input.dim[3];
  keys[4] = _type;
  keys[5] = (int)(_inPlace);
}


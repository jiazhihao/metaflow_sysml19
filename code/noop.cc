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

Tensor Graph::noop(Tensor _input)
{
  Op op = model->get_or_create_noop(_input);
  add_edge(_input.op, op, _input.idx, 0);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_noop(Tensor _input)
{
  // key is (inputN, inputC, inputH, inputW)
  NoopKey key(_input);
  NoOp* noOp;
  if (noop.find(key) != noop.end()) {
    noOp = noop[key];
  } else {
    noOp = new NoOp(this, _input);
    noOp->runtime = 0.0f;
    noop[key] = noOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = noOp;
  return ret;
}

NoOp::NoOp(Model* _model, Tensor _input)
: OpBase(_input, model, OP_NOOP)
{
  numOutputs = 1;
  outputs[0] = _input;
}

NoOp::~NoOp(void)
{}

bool NoOp::get_parameter(PMParameter para, int* value)
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

void NoOp::map(void)
{}

void NoOp::unmap(void)
{}

void NoOp::forward(void)
{}

void NoOp::collect_costs(float& exe_time, float& flops,
                         float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += 0;
  flops += 0;
  mem_acc += 0;
  num_kernels += 0;
}

NoopKey::NoopKey(Tensor input)
{
  keys[0] = input.dim[0];
  keys[1] = input.dim[1];
  keys[2] = input.dim[2];
  keys[3] = input.dim[3];
}


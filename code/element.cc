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

Tensor Graph::add(Tensor t1, Tensor t2)
{
  assert(t1.numDim == t2.numDim);
  for (int i = 0; i < t1.numDim; i++)
    assert(t1.dim[i] == t2.dim[i]);
  Op op = model->get_or_create_element(OpBase::OP_EW_ADD, t1, t2);
  inEdges[op];
  outEdges[op];
  {
    Edge in(t1.idx, t1.op), out(t1.idx, op);
    inEdges[op].insert(in);
    outEdges[t1.op].insert(out);
  }
  {
    Edge in(t2.idx, t2.op), out(t2.idx, op);
    inEdges[op].insert(in);
    outEdges[t2.op].insert(out);
  }
#ifdef VERBOSE
  printf("inEdges[guid = %zu ptr = %p]\n", op.guid, op.ptr);
#endif
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Tensor Graph::mul(Tensor t1, Tensor t2)
{
  assert(t1.numDim == t2.numDim);
  for (int i = 0; i < t1.numDim; i++)
    assert(t1.dim[i] == t2.dim[i]);
  Op op = model->get_or_create_element(OpBase::OP_EW_MUL, t1, t2);
  inEdges[op];
  outEdges[op];
  {
    Edge in(t1.idx, t1.op), out(t1.idx, op);
    inEdges[op].insert(in);
    outEdges[t1.op].insert(out);
  }
  {
    Edge in(t2.idx, t2.op), out(t2.idx, op);
    inEdges[op].insert(in);
    outEdges[t2.op].insert(out);
  }
#ifdef VERBOSE
  printf("inEdges[guid = %zu ptr = %p]\n", op.guid, op.ptr);
#endif
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_element(OpBase::OpType type,
                                Tensor t1, Tensor t2)
{
  // key is (inputN, inputC, inputH, inputW, type)
  ElementKey key(t1, type);
  Element* eleOp;
  if (element.find(key) != element.end()) {
    eleOp = element[key];
  } else {
    eleOp = new Element(this, type, t1, t2);
    measure_element_cost(eleOp);
    element[key] = eleOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = eleOp;
  return ret;
}

Element::Element(Model* _model, OpType _type,
                 Tensor _t1, Tensor _t2)
: OpBase(_t1, _t2, _model, _type)
{
  numOutputs = 1;
  outputs[0] = _t1;
  outputs[0].idx = 0;
}

Element::~Element(void)
{}

bool Element::get_parameter(OpParameter para, int* value)
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

void Element::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  flops += outputSize;
  mem_acc += inputSize * 2;
  num_kernels += 1;
}

// key is (inputN, inputC, inputH, inputW, type)
ElementKey::ElementKey(Tensor t, OpBase::OpType type)
{
  keys[0] = t.dim[0];
  keys[1] = t.dim[1];
  keys[2] = t.dim[2];
  keys[3] = t.dim[3];
  keys[4] = (int)(type);
}

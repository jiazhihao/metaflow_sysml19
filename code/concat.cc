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

Tensor Graph::concat(int n, Tensor* _inputs)
{
  for (int i = 1; i < n; i++) {
    assert(_inputs[i].numDim == _inputs[0].numDim);
    assert(_inputs[i].dim[0] == _inputs[0].dim[0]);
    for (int j = 2; j < _inputs[0].numDim; j++)
      assert(_inputs[i].dim[j] == _inputs[0].dim[j]);
  }
  bool needCopy[MAX_NUM_INPUTS];
  for (int i = 0; i < n; i++)
    needCopy[i] = true;
  Op op = model->get_or_create_concat(n, _inputs, needCopy);
  inEdges[op];
  outEdges[op];
  for (int i = 0; i < n; i++) {
    Edge in(_inputs[i].idx, _inputs[i].op), out(_inputs[i].idx, op);
    inEdges[op].insert(in);
    outEdges[_inputs[i].op].insert(out);
  }
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_concat(int n, Tensor* _inputs, bool* _needCopy)
{
  // keys are (n, inputN, inputH, inputW, bitmask(needCopy), inputC[0...n-1])
  ConcatKey key(n, _inputs, _needCopy);
  Concat* concatOp;
  if (concat.find(key) != concat.end()) {
    concatOp = concat[key];
  } else {
    concatOp = new Concat(this, n, _inputs, _needCopy);
    measure_concat_cost(concatOp);
    concat[key] = concatOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = concatOp;
  return ret;
}

Concat::Concat(Model* _model, int n, Tensor* _inputs, bool* _needCopy)
  : OpBase(n, _inputs, _model, OP_CONCAT)
{
  assert(n <= MAX_NUM_INPUTS);
  int outputC = 0;
  for (int i = 0; i < n; i++) {
    needCopy[i] = _needCopy[i];
    outputC += inputs[i].dim[1];
  }
  numOutputs = 1;
  outputs[0].numDim = inputs[0].numDim;
  outputs[0].dim[0] = BATCH_SIZE;
  outputs[0].dim[1] = outputC;
  for (int i = 2; i < outputs[0].numDim; i++)
    outputs[0].dim[i] = inputs[0].dim[i];
  outputs[0].idx = 0;
}

Concat::~Concat(void)
{}

bool Concat::get_parameter(OpParameter para, int* value)
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

void Concat::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  for (int i = 0; i < numInputs; i++)
    if (needCopy[i]) {
      int inputSize = 1;
      for (int j = 0; j < inputs[i].numDim; j++)
        inputSize *= inputs[i].dim[j];
      mem_acc += inputSize;
    }
  // cost metrics
  exe_time += runtime;
  flops += 0;
  num_kernels += 1;
}

// keys are (n, inputN, inputH, inputW, bitmask(needCopy), inputC[0...n-1])
ConcatKey::ConcatKey(int n, Tensor* _inputs, bool* _needCopy)
{
  keys[0] = n;
  keys[1] = _inputs[0].dim[0];
  keys[2] = _inputs[0].dim[2];
  keys[3] = _inputs[0].dim[3];
  keys[4] = 0;
  for (int i = 0; i < n; i++) {
    if (_needCopy[i])
      keys[4] = keys[4] * 2 + 1;
    else
      keys[4] = keys[4] * 2;
  }
  for (int i = 0; i < n; i++)
    keys[5 + i] = _inputs[i].dim[1];
  for (int i = 5 + n; i < CONCAT_KEY_LENGTH; i++)
    keys[i] = 0;
}


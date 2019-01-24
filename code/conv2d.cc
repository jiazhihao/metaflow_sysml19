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

Tensor Graph::conv2d(Tensor _input, int _outputC,
                     int _kernelH, int _kernelW,
                     int _strideH, int _strideW,
                     int _padH, int _padW, bool _relu)
{
  Tensor weight;
  weight.numDim = 4;
  weight.dim[0] = _outputC;
  weight.dim[1] = _input.dim[1];
  weight.dim[2] = _kernelH;
  weight.dim[3] = _kernelW;
  weight.op.guid = 0;
  weight.op.ptr = NULL;
  weight.idx = 0;
  weight = noop(weight);
  return conv2d(_input, weight, _strideH, _strideW, _padH, _padW, _relu);
}

Tensor Graph::conv2d(Tensor _input, Tensor _weight,
                     int _strideH, int _strideW,
                     int _padH, int _padW, bool _relu)
{
  Op op = model->get_or_create_conv2d(_input, _weight, _strideH, _strideW,
                                      _padH, _padW, _relu);
  add_edge(_input.op, op, _input.idx, 0);
  add_edge(_weight.op, op, _weight.idx, 1);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Op Model::get_or_create_conv2d(Tensor _input, Tensor _weight,
                               int _strideH, int _strideW,
                               int _padH, int _padW, bool _relu)
{
  // key is (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,
  //         strideH, strideW, padH, padW, relu)
  Conv2DKey key(_input, _weight, _strideH, _strideW, _padH, _padW, _relu);
  Conv2D* convOp;
  if (conv2d.find(key) != conv2d.end()) {
    convOp = conv2d[key];
  } else {
    convOp = new Conv2D(this, _input, _weight, _strideH, _strideW,
                        _padH, _padW, _relu);
    measure_conv2d_cost(convOp);
    conv2d[key] = convOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = convOp;
  return ret;
}

Conv2D::Conv2D(Model* _model, Tensor _input, Tensor _weight,
               int _strideH, int _strideW,
               int _padH, int _padW, bool _relu)
: OpBase(_input, _weight, _model, OP_CONV2D),
  strideH(_strideH), strideW(_strideW),
  padH(_padH), padW(_padW), relu(_relu)
{
  assert(_input.numDim == 4);
  assert(_weight.numDim == 4);
  assert(_input.dim[1] == _weight.dim[1]);
  //printf("k(%d %d) pad(%d %d) stride(%d %d)\n",
  //       kernelH, kernelW, padH, padW, strideH, strideW);
  int inputH = _input.dim[2];
  int inputW = _input.dim[3];
  int kernelH = _weight.dim[2];
  int kernelW = _weight.dim[3];
  int outputH = 1 + (inputH + 2 * padH - kernelH) / strideH;
  int outputW = 1 + (inputW + 2 * padW - kernelW) / strideW;
  numOutputs = 1;
  outputs[0].numDim = 4;
  outputs[0].dim[0] = _input.dim[0];
  outputs[0].dim[1] = _weight.dim[0];
  outputs[0].dim[2] = outputH;
  outputs[0].dim[3] = outputW;
  outputs[0].idx = 0;
}

Conv2D::~Conv2D(void)
{}

bool Conv2D::get_parameter(PMParameter para, int* value)
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
    case PM_KERNEL_H:
    case PM_KERNEL_W:
      assert(false);
      return true;
    case PM_STRIDE_H:
      *value = strideH;
      return true;
    case PM_STRIDE_W:
      *value = strideW;
      return true;
    case PM_PAD_H:
      *value = padH;
      return true;
    case PM_PAD_W:
      *value = padW;
      return true;
    case PM_RELU:
      *value = (int) relu;
      return true;
    default:
      return false;
  }
}

void Conv2D::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1, weightSize;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  for (int i = 0; i < inputs[1].numDim; i++)
    weightSize *= inputs[1].dim[i];
  // cost metrics
  exe_time += runtime;
  int kernelH = inputs[1].dim[2];
  int kernelW = inputs[1].dim[3];
  int inputC = inputs[0].dim[1];
  flops += outputSize * (kernelH * kernelW * inputC + 1);
  if (relu)
    flops += outputSize;
  mem_acc += inputSize + outputSize + weightSize;
  num_kernels += 1;
}

// keys are (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,
//           strideH, strideW, padH, padW, relu)
Conv2DKey::Conv2DKey(Tensor _input, Tensor _weight,
                     int _strideH, int _strideW,
                     int _padH, int _padW,
                     bool _relu)
{
  assert(_input.dim[1] == _weight.dim[1]);
  keys[0] = _input.dim[0];
  keys[1] = _input.dim[1];
  keys[2] = _input.dim[2];
  keys[3] = _input.dim[3];
  keys[4] = _weight.dim[0];
  keys[5] = _weight.dim[2];
  keys[6] = _weight.dim[3];
  keys[7] = _strideH;
  keys[8] = _strideW;
  keys[9] = _padH;
  keys[10] = _padW;
  keys[11] = (int)(_relu);
}


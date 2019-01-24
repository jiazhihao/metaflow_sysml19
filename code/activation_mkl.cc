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
#include "mkl_helper.h"

void Activation::map(void)
{
  assert(inputs[0].numDim == outputs[0].numDim);

  // Data size.
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  size_t outputSize = BATCH_SIZE * inputC * inputH * inputW;

  // Allocate tensors.
  CHECK_NE(nullptr, outputs[0].ptr = new DATATYPE[outputSize]);
}

void Activation::unmap(void)
{
  delete[] reinterpret_cast<DATATYPE*>(outputs[0].ptr);
  outputs[0].ptr = nullptr;
}

void Activation::forward(void)
{
  // Data size.
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  size_t outputSize = BATCH_SIZE * inputC * inputH * inputW;

  const auto inputPtr = reinterpret_cast<DATATYPE*>(inputs[0].ptr);
  auto outputPtr = reinterpret_cast<DATATYPE*>(outputs[0].ptr);

  switch (type) {
    case OP_RELU:
      vsFunc(outputSize, inputPtr, outputPtr, relu);
      break;
    case OP_SIGMOID:
      vsFunc(outputSize, inputPtr, outputPtr, sigmoid);
      break;
    default:
      assert(false);
  }
}

void Model::measure_activation_cost(Activation* act)
{
  assert(act->inputs[0].numDim == act->outputs[0].numDim);

  // Data size.
  int inputC = act->inputs[0].dim[1];
  int inputH = act->inputs[0].dim[2];
  int inputW = act->inputs[0].dim[3];

  // Bind function.
  auto execute = [&]() {
    size_t outputSize = BATCH_SIZE * inputC * inputH * inputW;

    switch (act->type) {
      case OpBase::OP_RELU:
        vsFunc(outputSize, inputPtr, outputPtr, relu);
        break;
      case OpBase::OP_SIGMOID:
        vsFunc(outputSize, inputPtr, outputPtr, sigmoid);
        break;
      default:
        assert(false);
    }
  };

  // Measure.
  execute();  // warmup
  auto beg = microsecond_timer();
  for (int i = 0; i < REPEAT_TIMES; i++) {
    execute();
  }
  auto end = microsecond_timer();

  act->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // millisecond
  printf("measure[Activation]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
         BATCH_SIZE, inputC, inputH, inputW, act->type, act->runtime);
}


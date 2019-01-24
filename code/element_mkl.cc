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

#include "mkl_vml.h"

void Element::map(void)
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

void Element::unmap(void)
{
  delete[] reinterpret_cast<DATATYPE*>(outputs[0].ptr);
  outputs[0].ptr = nullptr;
}

void Element::forward(void)
{
  // Data size.
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  size_t outputSize = BATCH_SIZE * inputC * inputH * inputW;

  switch (type) {
    case OpBase::OP_EW_ADD:
      vsAdd(outputSize, reinterpret_cast<DATATYPE*>(inputs[0].ptr),
          reinterpret_cast<DATATYPE*>(inputs[1].ptr),
          reinterpret_cast<DATATYPE*>(outputs[0].ptr));
      break;
    case OpBase::OP_EW_MUL:
      vsMul(outputSize, reinterpret_cast<DATATYPE*>(inputs[0].ptr),
          reinterpret_cast<DATATYPE*>(inputs[1].ptr),
          reinterpret_cast<DATATYPE*>(outputs[0].ptr));
      break;
    default:
      assert(false);
  }
}

void Model::measure_element_cost(Element* ele)
{
  assert(ele->inputs[0].numDim == ele->outputs[0].numDim);

  // Data size.
  int inputC = ele->inputs[0].dim[1];
  int inputH = ele->inputs[0].dim[2];
  int inputW = ele->inputs[0].dim[3];

  // Bind function.
  auto execute = [&]() {
    size_t outputSize = BATCH_SIZE * inputC * inputH * inputW;

    switch (ele->type) {
      case OpBase::OP_EW_ADD:
        vsAdd(outputSize, inputPtr, inputPtr, outputPtr);
        break;
      case OpBase::OP_EW_MUL:
        vsMul(outputSize, inputPtr, inputPtr, outputPtr);
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

  ele->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // millisecond
  printf("measure[Element]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
         BATCH_SIZE, inputC, inputH, inputW, ele->type, ele->runtime);
}


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

#include "mkl_cblas.h"
#include "mkl_vml.h"

void Matmul::map(void)
{
  assert(inputs[0].numDim == 3 && outputs[0].numDim == 3);
  assert(inputs[0].dim[0] == outputs[0].dim[0]);
  assert(inputs[0].dim[1] == outputs[0].dim[1]);

  // Data size.
  int inputX = inputs[0].dim[0];
  int inputN = inputs[0].dim[1];
  int batch = inputX * inputN;
  int inputC = inputs[0].dim[2];
  int outputC = outputs[0].dim[2];

  // Allocate tensors.
  size_t outputSize = batch * outputC;
  size_t filterSize = inputC * outputC;
  CHECK_NE(nullptr, outputs[0].ptr = new DATATYPE[outputSize]);
  CHECK_NE(nullptr, filterPtr = new DATATYPE[filterSize]);
}

void Matmul::unmap(void)
{
  delete[] reinterpret_cast<DATATYPE*>(outputs[0].ptr);
  delete[] reinterpret_cast<DATATYPE*>(filterPtr);
  outputs[0].ptr = nullptr;
  filterPtr = nullptr;
}

void Matmul::forward(void)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Data size.
  int inputX = inputs[0].dim[0];
  int inputN = inputs[0].dim[1];
  int batch = inputX * inputN;
  int inputC = inputs[0].dim[2];
  int outputC = outputs[0].dim[2];

  auto outputPtr = reinterpret_cast<DATATYPE*>(outputs[0].ptr);

  // O = W * I.
  // O should be outputC-by-batch, W should be outputC-by-inputC, and I should
  // be inputC-by-batch. So the matrices are column-major, and leading
  // dimensions are the channel dimensions.
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, outputC, batch, inputC,
      alpha, reinterpret_cast<DATATYPE*>(filterPtr), inputC,
      reinterpret_cast<DATATYPE*>(inputs[0].ptr), inputC,
      beta, outputPtr, outputC);
  size_t outputSize = outputC * batch;
  switch (actiMode) {
    case OpBase::AC_MODE_NONE:
      break;
    case OpBase::AC_MODE_SIGMOID:
      vsFunc(outputSize, outputPtr, outputPtr, sigmoid);
      break;
    case OpBase::AC_MODE_RELU:
      vsFunc(outputSize, outputPtr, outputPtr, relu);
      break;
    case OpBase::AC_MODE_TANH:
      vsTanh(outputSize, outputPtr, outputPtr);
      break;
    default:
      assert(false);
  }
}

void Model::measure_matmul_cost(Matmul* mm)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;

  assert(mm->inputs[0].numDim == 3 && mm->outputs[0].numDim == 3);

  // Data size.
  int inputX = mm->inputs[0].dim[0];
  int inputN = mm->inputs[0].dim[1];
  int batch = inputX * inputN;
  int inputC = mm->inputs[0].dim[2];
  int outputC = mm->outputs[0].dim[2];

  // Bind function.
  auto execute = [&]() {
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, outputC, batch, inputC,
        alpha, filterPtr, inputC, inputPtr, inputC, beta, outputPtr, outputC);
    size_t outputSize = outputC * batch;
    switch (mm->actiMode) {
      case OpBase::AC_MODE_NONE:
        break;
      case OpBase::AC_MODE_SIGMOID:
        vsFunc(outputSize, outputPtr, outputPtr, sigmoid);
        break;
      case OpBase::AC_MODE_RELU:
        vsFunc(outputSize, outputPtr, outputPtr, relu);
        break;
      case OpBase::AC_MODE_TANH:
        vsTanh(outputSize, outputPtr, outputPtr);
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

  mm->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // millisecond
  printf("measure[Matmul]: i(%d %d %d) o(%d) acti(%d) cost(%.4lf)\n",
         mm->inputs[0].dim[0], mm->inputs[0].dim[1], inputC, outputC,
         mm->actiMode, mm->runtime);
}


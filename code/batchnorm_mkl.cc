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

#include <cfloat>
#include "ops.h"
#include "mkl_helper.h"

#include "mkl_dnn.h"

void BatchNorm::map(void)
{
  assert(inputs[0].numDim == outputs[0].numDim);
  int numDim = outputs[0].numDim;

  float eps = FLT_EPSILON;
  unsigned int flags = dnnUseScaleShift;
  if (!model->isTraining) flags |= dnnUseInputMeanVariance;

  // Data size.
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  const size_t srcSize[] = {
    static_cast<size_t>(inputW), static_cast<size_t>(inputH),
    static_cast<size_t>(inputC), BATCH_SIZE};  // NCHW

  // Input layout.
  dnnLayout_t srcLayout;
  size_t srcStrides[MAX_DIM];
  getStridesFromSizes(srcStrides, srcSize, numDim);
  CHECK_MKL(dnnLayoutCreate_F32(&srcLayout, numDim, srcSize, srcStrides));

  // Create computation primitives and assign resources.
  CHECK_MKL(dnnBatchNormalizationCreateForward_v2_F32(
        &comp, nullptr, srcLayout, eps, flags));
  rsrc[dnnResourceSrc] = inputs[0].ptr;
  assert(rsrc[dnnResourceSrc] != nullptr);
  for (auto rType : {dnnResourceDst, dnnResourceScaleShift,
      dnnResourceMean, dnnResourceVariance}) {
    dnnLayout_t layout;
    CHECK_MKL(dnnLayoutCreateFromPrimitive_F32(&layout, comp, rType));
    CHECK_MKL(dnnAllocateBuffer_F32(&rsrc[rType], layout));
    CHECK_MKL(dnnLayoutDelete_F32(layout));
  }
  outputs[0].ptr = rsrc[dnnResourceDst];

  CHECK_MKL(dnnLayoutDelete_F32(srcLayout));
}

void BatchNorm::unmap(void)
{
  CHECK_MKL(dnnDelete_F32(comp));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrc[dnnResourceDst]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrc[dnnResourceScaleShift]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrc[dnnResourceMean]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrc[dnnResourceVariance]));
  comp = nullptr;
  rsrc.fill(nullptr);
  outputs[0].ptr = nullptr;
}

void BatchNorm::forward(void)
{
  CHECK_MKL(dnnExecute_F32(comp, rsrc.data()));
}

void Model::measure_batchnorm_cost(BatchNorm* bn)
{
  dnnPrimitive_t comp = nullptr;
  void* rsrc[dnnResourceNumber] = {nullptr};

  assert(bn->inputs[0].numDim == bn->outputs[0].numDim);
  int numDim = bn->outputs[0].numDim;

  float eps = FLT_EPSILON;
  unsigned int flags = dnnUseScaleShift;
  if (!isTraining) flags |= dnnUseInputMeanVariance;

  // Data size.
  int inputC = bn->inputs[0].dim[1];
  int inputH = bn->inputs[0].dim[2];
  int inputW = bn->inputs[0].dim[3];
  const size_t srcSize[] = {
    static_cast<size_t>(inputW), static_cast<size_t>(inputH),
    static_cast<size_t>(inputC), BATCH_SIZE};  // NCHW

  // Input layout.
  dnnLayout_t srcLayout;
  size_t srcStrides[MAX_DIM];
  getStridesFromSizes(srcStrides, srcSize, numDim);
  CHECK_MKL(dnnLayoutCreate_F32(&srcLayout, numDim, srcSize, srcStrides));

  // Create computation primitives and assign resources.
  CHECK_MKL(dnnBatchNormalizationCreateForward_v2_F32(
        &comp, nullptr, srcLayout, eps, flags));
  rsrc[dnnResourceSrc] = inputPtr;
  rsrc[dnnResourceDst] = outputPtr;
  rsrc[dnnResourceScaleShift] = scalePtr;
  rsrc[dnnResourceMean] = runningMean;
  rsrc[dnnResourceVariance] = runningVar;

  // Measure.
  CHECK_MKL(dnnExecute_F32(comp, rsrc));  // warmup
  auto beg = microsecond_timer();
  for (int i = 0; i < REPEAT_TIMES; i++) {
    CHECK_MKL(dnnExecute_F32(comp, rsrc));
  }
  auto end = microsecond_timer();

  CHECK_MKL(dnnDelete_F32(comp));
  CHECK_MKL(dnnLayoutDelete_F32(srcLayout));

  bn->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // millisecond
  printf("measure[BatchNorm]: i(%d %d %d %d) cost(%.4lf)\n",
         BATCH_SIZE, bn->inputs[0].dim[1], bn->inputs[0].dim[2],
         bn->inputs[0].dim[3], bn->runtime);
}


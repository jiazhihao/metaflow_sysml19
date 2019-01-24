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

#include "mkl_dnn.h"

void Pool2D::map(void)
{
  float relu_neg_slope = 0.;
  assert(inputs[0].numDim == outputs[0].numDim);
  int numDim = outputs[0].numDim;

  // Pooling type.
  dnnAlgorithm_t poolingAlgo;
  if (type == OpBase::OP_POOL2D_MAX)
    poolingAlgo = dnnAlgorithmPoolingMax;
  else if (type == OpBase::OP_POOL2D_AVG)
    poolingAlgo = dnnAlgorithmPoolingAvgExcludePadding;

  // Data size.
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  int outputH = outputs[0].dim[2];
  int outputW = outputs[0].dim[3];
  const size_t srcSize[] = {
    static_cast<size_t>(inputW), static_cast<size_t>(inputH),
    static_cast<size_t>(inputC), BATCH_SIZE};  // NCHW
  const size_t kernelSize[] = {
    static_cast<size_t>(kernelW), static_cast<size_t>(kernelH)};
  const size_t kernelStride[] = {
    static_cast<size_t>(strideW), static_cast<size_t>(strideH)};
  const int inputOffset[] = {-padW, -padH};

  // Input layout.
  dnnLayout_t srcLayout;
  size_t srcStrides[MAX_DIM];
  getStridesFromSizes(srcStrides, srcSize, numDim);
  CHECK_MKL(dnnLayoutCreate_F32(&srcLayout, numDim, srcSize, srcStrides));

  // Create computation primitives and assign resources.
  dnnPrimitive_t comp = nullptr;
  std::array<void*, dnnResourceNumber> rsrc;

  CHECK_MKL(dnnPoolingCreateForward_F32(
        &comp, nullptr, poolingAlgo, srcLayout, kernelSize, kernelStride, inputOffset,
        dnnBorderZeros));
  rsrc[dnnResourceSrc] = inputs[0].ptr;
  assert(rsrc[dnnResourceSrc] != nullptr);
  for (auto rType : {dnnResourceDst, dnnResourceWorkspace}) {
    dnnLayout_t layout;
    CHECK_MKL(dnnLayoutCreateFromPrimitive_F32(&layout, comp, rType));
    CHECK_MKL(dnnAllocateBuffer_F32(&rsrc[rType], layout));
    CHECK_MKL(dnnLayoutDelete_F32(layout));
  }

  compList.push_back(comp);
  rsrcList.push_back(rsrc);

  if (relu) {
    dnnPrimitive_t comp2 = nullptr;
    std::array<void*, dnnResourceNumber> rsrc2;

    dnnLayout_t dstLayout;
    CHECK_MKL(dnnLayoutCreateFromPrimitive_F32(&dstLayout, comp, dnnResourceDst));
    CHECK_MKL(dnnReLUCreateForward_F32(&comp2, nullptr, dstLayout, relu_neg_slope));

    rsrc2[dnnResourceSrc] = rsrc2[dnnResourceDst] = rsrc[dnnResourceDst];  // shared

    CHECK_MKL(dnnLayoutDelete_F32(dstLayout));

    compList.push_back(comp2);
    rsrcList.push_back(rsrc2);
  }

  outputs[0].ptr = rsrc[dnnResourceDst];

  CHECK_MKL(dnnLayoutDelete_F32(srcLayout));
}

void Pool2D::unmap(void)
{
  assert(!compList.empty() && !rsrcList.empty());
  CHECK_MKL(dnnDelete_F32(compList[0]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrcList[0][dnnResourceDst]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrcList[0][dnnResourceWorkspace]));
  if (compList.size() > 1) {
    assert(compList.size() == 2 && rsrcList.size() == 2);
    CHECK_MKL(dnnDelete_F32(compList[1]));
    // No need to release resources as they are shared with CONV.
  }
  compList.clear();
  rsrcList.clear();
  outputs[0].ptr = nullptr;
}

void Pool2D::forward(void)
{
  assert(compList.size() == rsrcList.size());
  for (size_t i = 0; i < compList.size(); i++)
    CHECK_MKL(dnnExecute_F32(compList[i], rsrcList[i].data()));
}

void Model::measure_pool2d_cost(Pool2D* pool)
{
  dnnPrimitive_t comp = nullptr;
  void* rsrc[dnnResourceNumber] = {nullptr};
  // For ReLU.
  dnnPrimitive_t comp2 = nullptr;
  void* rsrc2[dnnResourceNumber] = {nullptr};

  float relu_neg_slope = 0.;

  assert(pool->inputs[0].numDim == pool->outputs[0].numDim);
  int numDim = pool->outputs[0].numDim;

  // Pooling type.
  dnnAlgorithm_t poolingAlgo;
  if (pool->type == OpBase::OP_POOL2D_MAX)
    poolingAlgo = dnnAlgorithmPoolingMax;
  else if (pool->type == OpBase::OP_POOL2D_AVG)
    poolingAlgo = dnnAlgorithmPoolingAvgExcludePadding;

  // Data size.
  int inputC = pool->inputs[0].dim[1];
  int inputH = pool->inputs[0].dim[2];
  int inputW = pool->inputs[0].dim[3];
  int outputH = pool->outputs[0].dim[2];
  int outputW = pool->outputs[0].dim[3];
  const size_t srcSize[] = {
    static_cast<size_t>(inputW), static_cast<size_t>(inputH),
    static_cast<size_t>(inputC), BATCH_SIZE};  // NCHW
  const size_t kernelSize[] = {
    static_cast<size_t>(pool->kernelW), static_cast<size_t>(pool->kernelH)};
  const size_t kernelStride[] = {
    static_cast<size_t>(pool->strideW), static_cast<size_t>(pool->strideH)};
  const int inputOffset[] = {-pool->padW, -pool->padH};

  // Input layout.
  dnnLayout_t srcLayout;
  size_t srcStrides[MAX_DIM];
  getStridesFromSizes(srcStrides, srcSize, numDim);
  CHECK_MKL(dnnLayoutCreate_F32(&srcLayout, numDim, srcSize, srcStrides));

  // Create computation primitives and assign resources.
  CHECK_MKL(dnnPoolingCreateForward_F32(
        &comp, nullptr, poolingAlgo, srcLayout, kernelSize, kernelStride, inputOffset,
        dnnBorderZeros));
  rsrc[dnnResourceSrc] = inputPtr;
  rsrc[dnnResourceDst] = outputPtr;
  rsrc[dnnResourceWorkspace] = workSpace;

  if (pool->relu) {
    dnnLayout_t dstLayout;
    CHECK_MKL(dnnLayoutCreateFromPrimitive_F32(&dstLayout, comp, dnnResourceDst));
    CHECK_MKL(dnnReLUCreateForward_F32(&comp2, nullptr, dstLayout, relu_neg_slope));
    CHECK_MKL(dnnLayoutDelete_F32(dstLayout));

    rsrc2[dnnResourceSrc] = outputPtr;
    rsrc2[dnnResourceDst] = outputPtr;  // shared
  }

  // Measure.
  CHECK_MKL(dnnExecute_F32(comp, rsrc));  // warmup
  auto beg = microsecond_timer();
  for (int i = 0; i < REPEAT_TIMES; i++) {
    CHECK_MKL(dnnExecute_F32(comp, rsrc));
    if (comp2 != nullptr) CHECK_MKL(dnnExecute_F32(comp2, rsrc2));
  }
  auto end = microsecond_timer();

  CHECK_MKL(dnnDelete_F32(comp));
  CHECK_MKL(dnnLayoutDelete_F32(srcLayout));
  if (comp2) CHECK_MKL(dnnDelete_F32(comp2));

  pool->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // millisecond
  printf("measure[Pool2D]: i(%d %d %d %d) k(%d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
         BATCH_SIZE, inputC, inputH, inputW, pool->kernelH, pool->kernelW,
         pool->strideH, pool->strideW, pool->padH, pool->padW, pool->runtime);
}


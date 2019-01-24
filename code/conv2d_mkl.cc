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

void Conv2D::map(void)
{
  float relu_neg_slope = 0.;
  assert(inputs[0].numDim == outputs[0].numDim);
  int numDim = outputs[0].numDim;

  // Data size.
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  int outputC = outputs[0].dim[1];
  int outputH = outputs[0].dim[2];
  int outputW = outputs[0].dim[3];
  const size_t srcSize[] = {
    static_cast<size_t>(inputW), static_cast<size_t>(inputH),
    static_cast<size_t>(inputC), BATCH_SIZE};  // NCHW
  const size_t dstSize[] = {
    static_cast<size_t>(outputW), static_cast<size_t>(outputH),
    static_cast<size_t>(outputC), BATCH_SIZE};  // NCHW
  const size_t filterSize[] = {
    static_cast<size_t>(kernelW), static_cast<size_t>(kernelH),
    static_cast<size_t>(inputC), static_cast<size_t>(outputC)};
  const size_t biasSize[] = {static_cast<size_t>(outputC)};
  const size_t convStrides[] = {
    static_cast<size_t>(strideW), static_cast<size_t>(strideH)};
  const int inputOffset[] = {-padW, -padH};

  // Create computation primitives and assign resources.
  dnnPrimitive_t comp = nullptr;
  std::array<void*, dnnResourceNumber> rsrc;

  CHECK_MKL(dnnConvolutionCreateForwardBias_F32(
        &comp, nullptr, dnnAlgorithmConvolutionDirect, numDim,
        srcSize, dstSize, filterSize, convStrides, inputOffset,
        dnnBorderZeros));
  rsrc[dnnResourceSrc] = inputs[0].ptr;
  assert(rsrc[dnnResourceSrc] != nullptr);
  for (auto rType : {dnnResourceDst, dnnResourceFilter, dnnResourceBias}) {
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
}

void Conv2D::unmap(void)
{
  assert(!compList.empty() && !rsrcList.empty());
  CHECK_MKL(dnnDelete_F32(compList[0]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrcList[0][dnnResourceDst]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrcList[0][dnnResourceFilter]));
  CHECK_MKL(dnnReleaseBuffer_F32(rsrcList[0][dnnResourceBias]));
  if (compList.size() > 1) {
    assert(compList.size() == 2 && rsrcList.size() == 2);
    CHECK_MKL(dnnDelete_F32(compList[1]));
    // No need to release resources as they are shared with CONV.
  }
  compList.clear();
  rsrcList.clear();
  outputs[0].ptr = nullptr;
}

void Conv2D::forward(void)
{
  assert(compList.size() == rsrcList.size());
  for (size_t i = 0; i < compList.size(); i++)
    CHECK_MKL(dnnExecute_F32(compList[i], rsrcList[i].data()));
}

void Model::measure_conv2d_cost(Conv2D* conv)
{
  dnnPrimitive_t comp = nullptr;
  void* rsrc[dnnResourceNumber] = {nullptr};
  // For ReLU.
  dnnPrimitive_t comp2 = nullptr;
  void* rsrc2[dnnResourceNumber] = {nullptr};

  float relu_neg_slope = 0.;

  assert(conv->inputs[0].numDim == conv->outputs[0].numDim);
  int numDim = conv->outputs[0].numDim;

  // Data size.
  int inputC = conv->inputs[0].dim[1];
  int inputH = conv->inputs[0].dim[2];
  int inputW = conv->inputs[0].dim[3];
  int outputC = conv->outputs[0].dim[1];
  int outputH = conv->outputs[0].dim[2];
  int outputW = conv->outputs[0].dim[3];
  const size_t srcSize[] = {
    static_cast<size_t>(inputW), static_cast<size_t>(inputH),
    static_cast<size_t>(inputC), BATCH_SIZE};  // NCHW
  const size_t dstSize[] = {
    static_cast<size_t>(outputW), static_cast<size_t>(outputH),
    static_cast<size_t>(outputC), BATCH_SIZE};  // NCHW
  const size_t filterSize[] = {
    static_cast<size_t>(conv->kernelW), static_cast<size_t>(conv->kernelH),
    static_cast<size_t>(inputC), static_cast<size_t>(outputC)};
  const size_t biasSize[] = {static_cast<size_t>(outputC)};
  const size_t convStrides[] = {
    static_cast<size_t>(conv->strideW), static_cast<size_t>(conv->strideH)};
  const int inputOffset[] = {-conv->padW, -conv->padH};

  // Create computation primitives and assign resources.
  CHECK_MKL(dnnConvolutionCreateForwardBias_F32(
        &comp, nullptr, dnnAlgorithmConvolutionDirect, numDim,
        srcSize, dstSize, filterSize, convStrides, inputOffset,
        dnnBorderZeros));
  rsrc[dnnResourceSrc] = inputPtr;
  rsrc[dnnResourceDst] = outputPtr;
  rsrc[dnnResourceFilter] = filterPtr;
  rsrc[dnnResourceBias] = biasPtr;

  if (conv->relu) {
    dnnLayout_t layout;
    CHECK_MKL(dnnLayoutCreateFromPrimitive_F32(&layout, comp, dnnResourceDst));
    CHECK_MKL(dnnReLUCreateForward_F32(&comp2, nullptr, layout, relu_neg_slope));
    CHECK_MKL(dnnLayoutDelete_F32(layout));

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
  if (comp2) CHECK_MKL(dnnDelete_F32(comp2));

  conv->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // millisecond
  printf("measure[Conv2D]: i(%d %d %d %d) o(%d) k(%d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
         BATCH_SIZE, inputC, inputH, inputW, outputC, conv->kernelH, conv->kernelW,
         conv->strideH, conv->strideW, conv->padH, conv->padW, conv->runtime);
}


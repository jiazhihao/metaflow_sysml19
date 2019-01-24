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
#include "cuda_helper.h"

void Pool2D::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  // set descriptors
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  cudnnPoolingMode_t mode;
  if (type == OP_POOL2D_MAX)
    mode = CUDNN_POOLING_MAX;
  else if (type == OP_POOL2D_AVG)
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN,
      kernelH, kernelW, padH, padW, strideH, strideW));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, 
      inputTensor, &n, &c, &h, &w));
  assert(n == BATCH_SIZE);
  assert(c == inputC);
  assert(outputs[0].dim[2] == h);
  assert(outputs[0].dim[3] == w);
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  if (relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN, 0.0));
  }
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * n * c * h * w;
  checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
}

void Pool2D::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
  if (relu) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  // free tensors
  checkCUDA(cudaFree(outputs[0].ptr));
}

void Pool2D::forward(void)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnPoolingForward(model->dnn, poolDesc,
      &alpha, inputTensor, inputs[0].ptr,
      &beta, outputTensor, outputs[0].ptr));
  if (relu) {
    checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
        &alpha, outputTensor, outputs[0].ptr,
        &beta, outputTensor, outputs[0].ptr));
  }
}

void Model::measure_pool2d_cost(Pool2D* pool)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputC = pool->inputs[0].dim[1];
  int inputH = pool->inputs[0].dim[2];
  int inputW = pool->inputs[0].dim[3];
  int outputH = pool->outputs[0].dim[2];
  int outputW = pool->outputs[0].dim[3];
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  cudnnPoolingMode_t mode;
  if (pool->type == OpBase::OP_POOL2D_MAX)
    mode = CUDNN_POOLING_MAX;
  else if (pool->type == OpBase::OP_POOL2D_AVG)
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, mode,
      CUDNN_PROPAGATE_NAN, pool->kernelH, pool->kernelW, pool->padH, pool->padW,
      pool->strideH, pool->strideW));
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  int n, c, h, w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc,
      inputTensor, &n, &c, &h, &w));
  assert(n == BATCH_SIZE);
  assert(c == inputC);
  assert(outputH == h);
  assert(outputW == w);
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  size_t inputSize = sizeof(DATATYPE) * BATCH_SIZE * inputC * inputH * inputW;
  size_t outputSize = sizeof(DATATYPE) * BATCH_SIZE * inputC * outputH * outputW;
  assert(inputSize < MAX_TENSOR_SIZE);
  assert(outputSize < MAX_TENSOR_SIZE);
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    checkCUDNN(cudnnPoolingForward(dnn, poolDesc,
        &alpha, inputTensor, inputPtr,
        &beta, outputTensor, outputPtr));
    if (pool->relu) {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, outputTensor, outputPtr,
          &beta, outputTensor, outputPtr));
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  pool->runtime = milliseconds / REPEAT_TIMES;
  printf("measure[Pool2D]: i(%d %d %d %d) k(%d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
         BATCH_SIZE, inputC, inputH, inputW, pool->kernelH, pool->kernelW,
         pool->strideH, pool->strideW, pool->padH, pool->padW, pool->runtime);
}


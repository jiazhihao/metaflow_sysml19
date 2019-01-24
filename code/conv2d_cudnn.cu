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

void Conv2D::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  int inputC = inputs[0].dim[1];
  int inputH = inputs[0].dim[2];
  int inputW = inputs[0].dim[3];
  int outputC = inputs[1].dim[0];
  int kernelH = inputs[1].dim[2];
  int kernelW = inputs[1].dim[3];
  // set descriptors
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, outputC, 1, 1));
  checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW, outputC, inputC, kernelH, kernelW));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padH, padW,
      strideH, strideW, 1/*dilationH*/, 1/*dilationW*/,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
      inputTensor, filterDesc, &n, &c, &h, &w));
  assert(n == BATCH_SIZE);
  assert(c == outputC);
  assert(outputs[0].dim[2] == h);
  assert(outputs[0].dim[3] == w);
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  if (relu) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  }
  int outputH = h;
  int outputW = w;
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * BATCH_SIZE * outputC * outputH * outputW;
  size_t biasSize = sizeof(DATATYPE) * outputC;
  checkCUDA(cudaMalloc(&biasPtr, biasSize));
  checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
}

void Conv2D::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputTensor));
  checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
  if (relu) {
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  // free tensors
  checkCUDA(cudaFree(outputs[0].ptr));
  checkCUDA(cudaFree(biasPtr));
}

void Conv2D::forward(void)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (relu) {
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        model->dnn, &alpha, inputTensor, inputs[0].ptr, filterDesc, inputs[1].ptr,
        convDesc, fwdAlgo, model->workSpace, model->workSpaceSize,
        &beta, outputTensor, outputs[0].ptr, biasTensor, biasPtr, actiDesc,
        outputTensor, outputs[0].ptr));
  } else {
    checkCUDNN(cudnnConvolutionForward(
        model->dnn, &alpha, inputTensor, inputs[0].ptr, filterDesc, inputs[1].ptr,
        convDesc, fwdAlgo, model->workSpace, model->workSpaceSize,
        &beta, outputTensor, outputs[0].ptr));
    checkCUDNN(cudnnAddTensor(model->dnn, &alpha, biasTensor, biasPtr,
        &alpha, outputTensor, outputs[0].ptr));
  }
}

void Model::measure_conv2d_cost(Conv2D* conv)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int inputC = conv->inputs[0].dim[1];
  int inputH = conv->inputs[0].dim[2];
  int inputW = conv->inputs[0].dim[3];
  int kernelH = conv->inputs[1].dim[2];
  int kernelW = conv->inputs[1].dim[3];
  int outputC = conv->outputs[0].dim[1];
  int outputH = conv->outputs[0].dim[2];
  int outputW = conv->outputs[0].dim[3];
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, 1, outputC, 1, 1));
  checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW, outputC, inputC, kernelH, kernelW));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, conv->padH, conv->padW,
      conv->strideH, conv->strideW, 1/*dilationH*/, 1/*dilationW*/,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));
  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
      inputTensor, filterDesc, &n, &c, &h, &w));
  assert(n == BATCH_SIZE);
  assert(c == outputC);
  assert(outputH == h);
  assert(outputW == w);
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, n, c, h, w));
  size_t inputSize = sizeof(DATATYPE) * BATCH_SIZE * inputC * inputH * inputW;
  size_t filterSize = sizeof(DATATYPE) * inputC * outputC
                      * kernelH * kernelW;
  size_t outputSize = sizeof(DATATYPE) * BATCH_SIZE * outputC * outputH * outputW;
  assert(inputSize < MAX_TENSOR_SIZE);
  assert(filterSize < MAX_TENSOR_SIZE);
  assert(outputSize < MAX_TENSOR_SIZE);

  const int reqAlgCnt = 8;
  int cnt = 0;
  cudnnConvolutionFwdAlgoPerf_t perfResults[reqAlgCnt];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithmEx(
      dnn, inputTensor, inputPtr, filterDesc, filterPtr, convDesc,
      outputTensor, outputPtr, reqAlgCnt, &cnt, perfResults,
      workSpace, workSpaceSize));
  assert(cnt > 0);
  checkCUDNN(perfResults[0].status);
  for (int i = 0; i < cnt; i++) {
    printf("fwdAlgo(%d) time(%.2lfms) space(%dMB)\n", perfResults[i].algo,
           perfResults[i].time, perfResults[i].memory / 1024 / 1024);
  }
  conv->fwdAlgo = perfResults[0].algo;
 
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    if (conv->relu) {
      checkCUDNN(cudnnConvolutionBiasActivationForward(
          dnn, &alpha, inputTensor, inputPtr, filterDesc, filterPtr,
          convDesc, conv->fwdAlgo, workSpace, workSpaceSize,
          &beta, outputTensor, outputPtr, biasTensor, biasPtr, actiDesc,
          outputTensor, outputPtr));
    } else {
      checkCUDNN(cudnnConvolutionForward(
          dnn, &alpha, inputTensor, inputPtr, filterDesc, filterPtr,
          convDesc, conv->fwdAlgo, workSpace, workSpaceSize,
          &beta, outputTensor, outputPtr));
      checkCUDNN(cudnnAddTensor(dnn, &alpha, biasTensor, biasPtr,
          &alpha, outputTensor, outputPtr));
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  conv->runtime = milliseconds / REPEAT_TIMES;
  printf("measure[Conv2D]: i(%d %d %d %d) o(%d) k(%d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
         BATCH_SIZE, inputC, inputH, inputW, outputC, kernelH, kernelW,
         conv->strideH, conv->strideW, conv->padH, conv->padW, conv->runtime);
}


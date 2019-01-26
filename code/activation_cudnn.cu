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

void Activation::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  if (inputs[0].numDim == 4) {
    int inputC = inputs[0].dim[1];
    int inputH = inputs[0].dim[2];
    int inputW = inputs[0].dim[3];
    // set descriptors
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, BATCH_SIZE, inputC, inputH, inputW));
  } else if (inputs[0].numDim == 3) {
    int dims[] = {inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2]};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(inputTensor, CUDNN_DATA_FLOAT, 
                                          3, dims, strides));
  } else {
    assert(false);
  }
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  cudnnActivationMode_t mode;
  switch (type) {
    case OP_RELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case OP_SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  if (!inPlace) {
    size_t outputSize = sizeof(DATATYPE);
    for (int i = 0; i < inputs[0].numDim; i++)
      outputSize *= inputs[0].dim[i];
    checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
  } else {
    outputs[0].ptr = inputs[0].ptr;
  }
}

void Activation::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  if (!inPlace) {
    checkCUDA(cudaFree(outputs[0].ptr));
  }
}

void Activation::forward(void)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
      &alpha, inputTensor, inputs[0].ptr,
      &beta, inputTensor, outputs[0].ptr));
}

void Model::measure_activation_cost(Activation* act)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (act->inputs[0].numDim == 4) {
    int inputB = act->inputs[0].dim[0];
    int inputC = act->inputs[0].dim[1];
    int inputH = act->inputs[0].dim[2];
    int inputW = act->inputs[0].dim[3];
    checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, inputB, inputC, inputH, inputW));
  } else if (act->inputs[0].numDim == 3) {
    int dims[] = {act->inputs[0].dim[0], act->inputs[0].dim[1], act->inputs[0].dim[2]};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(inputTensor, CUDNN_DATA_FLOAT, 
                                          3, dims, strides));
  } else {
    assert(false);
  }
  cudnnActivationMode_t mode;
  switch (act->type) {
    case OpBase::OP_RELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case OpBase::OP_SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    if (act->inPlace) {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, inputTensor, inputPtr,
          &beta, inputTensor, inputPtr));
    } else {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, inputTensor, inputPtr,
          &beta, inputTensor, outputPtr));
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  act->runtime = milliseconds / REPEAT_TIMES;
#ifdef VERBOSE
  printf("measure[Activation]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
         act->inputs[0].dim[0], act->inputs[0].dim[1], act->inputs[0].dim[2],
         act->inputs[0].dim[3], act->type, act->runtime);
#endif
}


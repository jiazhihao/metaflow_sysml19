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

void Concat::map(void)
{
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].ptr, outputSize));
}

void Concat::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].ptr));
}

void Concat::forward(void)
{
  off_t offset = 0;
  for (int i = 0; i < numInputs; i++) {
    size_t size = sizeof(DATATYPE);
    for (int j = 0; j < inputs[i].numDim; j++)
      size *= inputs[i].dim[j];
    if (needCopy[i])
      checkCUDA(cudaMemcpyAsync(((char*)outputs[0].ptr) + offset,
                                inputs[i].ptr, size,
                                cudaMemcpyDeviceToDevice));
    offset += size;
  }
}

void Model::measure_concat_cost(Concat* concat)
{
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    for (int j = 0; j < concat->numInputs; j++) {
      if (concat->needCopy[j]) {
        size_t size = sizeof(DATATYPE);
        for (int k = 0; k < concat->inputs[j].numDim; k++)
          size *= concat->inputs[j].dim[k];
        checkCUDA(cudaMemcpyAsync(outputPtr, inputPtr, size,
                                  cudaMemcpyDeviceToDevice));
      }
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  concat->runtime = milliseconds / REPEAT_TIMES;
#ifdef VERBOSE
  printf("measure[Concat]: cost(%.4lf)\n", concat->runtime);
#endif
}


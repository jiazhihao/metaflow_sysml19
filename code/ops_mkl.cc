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
#include <cstring>
#include <functional>
#include <numeric>
#include "mkl_helper.h"

Model::Model(bool training)
  : isTraining(training)
{
  global_unique_id = 100;
  workSpaceSize = WORK_SPACE_SIZE;
  CHECK_NE(nullptr, workSpace = new DATATYPE[workSpaceSize]);
  printf("handle.workSpace = 0x%lx\n", reinterpret_cast<uintptr_t>(workSpace));

  // Allocate tensors for perf profiling.
  CHECK_NE(nullptr, inputPtr = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, outputPtr = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, filterPtr = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, biasPtr = new DATATYPE[MAX_TENSOR_SIZE]);
  // Allocate tensors for batch norm.
  CHECK_NE(nullptr, scalePtr = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, runningMean = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, runningVar = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, saveMean = new DATATYPE[MAX_TENSOR_SIZE]);
  CHECK_NE(nullptr, saveVar = new DATATYPE[MAX_TENSOR_SIZE]);
}

void Concat::map(void)
{
  auto outputSize = std::accumulate(outputs[0].dim, outputs[0].dim + outputs[0].numDim,
      1, std::multiplies<int>());
  CHECK_NE(nullptr, outputs[0].ptr = new DATATYPE[outputSize]);
}

void Concat::unmap(void)
{
  delete[] reinterpret_cast<DATATYPE*>(outputs[0].ptr);
  outputs[0].ptr = nullptr;
}

void Concat::forward(void)
{
  char* outputPtr = reinterpret_cast<char*>(outputs[0].ptr);
  size_t offset = 0;
  for (int i = 0; i < numInputs; i++) {
    auto size = std::accumulate(inputs[i].dim, inputs[i].dim + inputs[i].numDim,
        sizeof(DATATYPE), std::multiplies<int>());
    if (needCopy[i])
      std::memcpy(outputPtr + offset, inputs[i].ptr, size);
    offset += size;
  }
  assert(offset == std::accumulate(outputs[0].dim, outputs[0].dim + outputs[0].numDim,
      sizeof(DATATYPE), std::multiplies<int>()));
}

void Model::measure_concat_cost(Concat* concat)
{
  auto beg = microsecond_timer();
  for (int i = 0; i < REPEAT_TIMES; i++) {
    for (int j = 0; j < concat->numInputs; j++) {
      if (concat->needCopy[j]) {
        size_t size = sizeof(DATATYPE);
        for (int k = 0; k < concat->inputs[j].numDim; k++)
          size *= concat->inputs[j].dim[k];
        std::memcpy(outputPtr, inputPtr, size);
      }
    }
  }
  auto end = microsecond_timer();

  concat->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;
  printf("measure[Concat] cost(%.4lf)\n", concat->runtime);
}

float Model::measure_oplist_runtime(const std::vector<OpBase*>& opBaseList)
{
  const int num_runs = 10;
  // warmup
  for (int i = 0; i < opBaseList.size(); i++)
    opBaseList[i]->forward();
  // measure runtime
  auto beg = microsecond_timer();
  for (int times = 0; times < num_runs; times++) {
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
  }
  auto end = microsecond_timer();
  return (end - beg) / 1.e3 / num_runs;
}

void* Model::allocate_memory(size_t size)
{
  return malloc(size);
}

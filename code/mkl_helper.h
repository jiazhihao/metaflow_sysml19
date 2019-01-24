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

#ifndef _MKL_HELPER_H_
#define _MKL_HELPER_H_

#include <sstream>
#include <iostream>
#include <time.h>
#include <cmath>

#define _STR(x) #x
#define STR(x) _STR(x)

#define _ERROR_HEAD \
  std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] "

#define CHECK_EQ(expect, actual) if ((expect) != (actual)) {            \
  _ERROR_HEAD << "value != " << STR(expect) << std::endl;               \
  exit(1);                                                              \
}

#define CHECK_NE(notExpect, actual) if ((notExpect) == (actual)) {      \
  _ERROR_HEAD << "value == " << STR(notExpect) << std::endl;            \
  exit(1);                                                              \
}

inline long long microsecond_timer() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return (tv.tv_sec * 1000 * 1000) + (tv.tv_nsec / 1000);
}

#include "mkl_dnn.h"

static inline const char* mkl_error_str(dnnError_t e) {
  switch(e) {
    case E_SUCCESS:
      return "E_SUCCESS";
    case E_INCORRECT_INPUT_PARAMETER:
      return "E_INCORRECT_INPUT_PARAMETER";
    case E_MEMORY_ERROR:
      return "E_MEMORY_ERROR";
    case E_UNSUPPORTED_DIMENSION:
      return "E_UNSUPPORTED_DIMENSION";
    case E_UNIMPLEMENTED:
      return "E_UNIMPLEMENTED";
  }
  return "UNKNOWN ERROR";
}

#define CHECK_MKL(call) do {                                            \
  auto e = call;                                                        \
  if (e != E_SUCCESS) {                                                 \
    _ERROR_HEAD << STR(call) << " = " << mkl_error_str(e) << std::endl; \
    exit(1);                                                            \
  }                                                                     \
} while (0)

static inline void getStridesFromSizes(size_t strides[], const size_t sizes[], size_t dimension) {
  strides[0] = 1;
  for (size_t i = 1; i < dimension; i++) {
    strides[i] = strides[i - 1] * sizes[i - 1];
  }
}

static inline float relu(float val) { return val > 0.f ? val : 0.f; }

static inline float sigmoid(float val) { return 1.f / (1.f + expf(-val)); }

template<typename Func>
static inline void vsFunc(size_t size, const float* input, float* output, Func func) {
  for (size_t i = 0; i < size; i++)
#pragma omp parallel
    output[i] = func(input[i]);
}

#endif  // _MKL_HELPER_H_


#include "cuda_helper.h"

__global__
void assign_kernel(float* ptr, int size, float value)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = value;
  }
}


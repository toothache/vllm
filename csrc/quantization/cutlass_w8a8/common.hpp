#pragma once

#include "cutlass/cutlass.h"
#include <climits>

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                        \
  {                                                  \
    TORCH_CHECK(status == cutlass::Status::kSuccess, \
                cutlassGetStatusString(status))      \
  }

inline uint32_t next_pow_2(uint32_t num) {
  if (num <= 1) return num;
#ifdef _MSC_VER
  num--;
  num |= num >> 1;
  num |= num >> 2;
  num |= num >> 4;
  num |= num >> 8;
  num |= num >> 16;
  num++;
  return num;
#else
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
#endif
}

inline int get_cuda_max_shared_memory_per_block_opt_in(int const device) {
  int max_shared_mem_per_block_opt_in = 0;
  cudaDeviceGetAttribute(&max_shared_mem_per_block_opt_in,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin,
                        device);
  return max_shared_mem_per_block_opt_in;
}


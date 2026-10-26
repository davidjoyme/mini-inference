// kernels/fused_moe/csrc/moe_align.cu - group (token, expert) pairs by expert in ONE CUDA kernel.
//
// Replaces the ~35 small PyTorch ops of the Python moe_align (sort, scatter, cumsum, search, ...)
// with a single launch. Same outputs (see fused_moe_triton.moe_align):
//   sorted_ids [max_padded]  pair index per row, grouped by expert, each expert's group padded
//                            to a multiple of block_m with num_pairs (= "no pair")
//   expert_ids [max_blocks]  the expert that owns each block_m-row tile
//   num_padded [1]           rows actually used
// Within one expert the order of pairs may differ from run to run (atomic placement); that's
// fine, since every row of the matmul is computed independently.
//
// One thread block does everything (an MoE layer has few experts, e.g. 64):
//   1. count pairs per expert          shared-memory atomics
//   2. padded prefix sum over experts  one thread (num_experts is small)
//   3. owner expert of every tile      binary search over the offsets
//   4. place each pair in its group    per-expert atomic cursors
// It runs on PyTorch's current CUDA stream, with no host synchronization, so it can be
// captured in a CUDA graph.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename index_t>
__global__ void moe_align_kernel(const index_t* __restrict__ topk_ids, int num_pairs, int num_experts,
                                 int block_m, int max_padded, int max_blocks,
                                 int* __restrict__ sorted_ids, int* __restrict__ expert_ids,
                                 int* __restrict__ num_padded) {
  extern __shared__ int smem[];
  int* counts = smem;                        // [num_experts]
  int* offsets = counts + num_experts;       // [num_experts + 1], padded start of each expert
  int* cursor = offsets + num_experts + 1;   // [num_experts], next free row of each expert

  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) counts[e] = 0;
  for (int i = threadIdx.x; i < max_padded; i += blockDim.x) sorted_ids[i] = num_pairs;
  __syncthreads();

  // 1. count
  for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
    atomicAdd(&counts[static_cast<int>(topk_ids[i])], 1);
  }
  __syncthreads();

  // 2. padded prefix sum
  if (threadIdx.x == 0) {
    int total = 0;
    offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
      total += (counts[e] + block_m - 1) / block_m * block_m;
      offsets[e + 1] = total;
    }
    *num_padded = total;
  }
  __syncthreads();

  // 3. owner of each tile: first expert whose padded range ends after the tile's first row
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) cursor[e] = offsets[e];
  for (int b = threadIdx.x; b < max_blocks; b += blockDim.x) {
    const int row = b * block_m;
    int owner = num_experts - 1;               // tiles past the end are never computed
    if (row < offsets[num_experts]) {
      int lo = 0, hi = num_experts - 1;
      while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= row) lo = mid + 1; else hi = mid;
      }
      owner = lo;
    }
    expert_ids[b] = owner;
  }
  __syncthreads();

  // 4. place each pair in its expert's group
  for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
    const int e = static_cast<int>(topk_ids[i]);
    const int pos = atomicAdd(&cursor[e], 1);
    sorted_ids[pos] = i;
  }
}

void moe_align(torch::Tensor topk_ids, int64_t num_experts, int64_t block_m,
               torch::Tensor sorted_ids, torch::Tensor expert_ids, torch::Tensor num_padded) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(sorted_ids.scalar_type() == at::kInt && expert_ids.scalar_type() == at::kInt &&
              num_padded.scalar_type() == at::kInt, "outputs must be int32");
  const at::cuda::CUDAGuard device_guard(topk_ids.device());
  const int num_pairs = static_cast<int>(topk_ids.numel());
  const int threads = 1024;
  const size_t shared = (3 * num_experts + 1) * sizeof(int);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto* ids) {
    moe_align_kernel<<<1, threads, shared, stream>>>(
        ids, num_pairs, static_cast<int>(num_experts), static_cast<int>(block_m),
        static_cast<int>(sorted_ids.numel()), static_cast<int>(expert_ids.numel()),
        sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(), num_padded.data_ptr<int>());
  };
  if (topk_ids.scalar_type() == at::kLong) {
    launch(topk_ids.data_ptr<int64_t>());
  } else if (topk_ids.scalar_type() == at::kInt) {
    launch(topk_ids.data_ptr<int>());
  } else {
    TORCH_CHECK(false, "topk_ids must be int32 or int64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align", &moe_align, "Group (token, expert) pairs by expert, padded to block_m (CUDA)");
}
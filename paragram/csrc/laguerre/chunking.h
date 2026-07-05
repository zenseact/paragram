#pragma once

#include <cstdlib>
#include <cuda_runtime.h>

namespace laguerre
{

struct ChunkPlan
{
    int chunk_blocks;
    int chunk_threads;
    int num_chunks;
};

inline ChunkPlan plan_convex_cell_chunks(int total_blocks, int block_stride, size_t full_bytes)
{
    int chunk_blocks = total_blocks;
    const char *force = std::getenv("PARAGRAM_NUM_CHUNKS");

    if (force != nullptr && std::atoi(force) > 0)
    {
        int num_chunks = std::atoi(force);
        chunk_blocks = (total_blocks + num_chunks - 1) / num_chunks;
    }
    else
    {
        double frac = 0.5;
        const char *frac_env = std::getenv("PARAGRAM_MEM_FRAC");
        if (frac_env != nullptr)
        {
            double parsed = std::atof(frac_env);
            if (parsed > 0.05 && parsed < 0.95)
                frac = parsed;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        size_t budget = static_cast<size_t>(frac * static_cast<double>(free_bytes));
        if (budget > 0 && full_bytes > budget)
        {
            int num_chunks = static_cast<int>((full_bytes + budget - 1) / budget);
            chunk_blocks = (total_blocks + num_chunks - 1) / num_chunks;
        }
    }

    if (chunk_blocks < 1)
        chunk_blocks = 1;
    if (chunk_blocks > total_blocks)
        chunk_blocks = total_blocks;

    int chunk_threads = chunk_blocks * block_stride;
    int num_chunks = (total_blocks + chunk_blocks - 1) / chunk_blocks;
    return {chunk_blocks, chunk_threads, num_chunks};
}

} // namespace laguerre

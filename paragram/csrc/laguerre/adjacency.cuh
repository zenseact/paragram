#include <cuda_runtime.h>
#include <tuple>

namespace laguerre
{

static __global__ void gather_adjacency_kernel(int P,
                                        int num_points,
                                        const int *__restrict__ neighbor_data,
                                        int *__restrict__ adjacency,
                                        int *__restrict__ offsets,
                                        int *__restrict__ adjacency_sizes,
                                        int *__restrict__ offsets_count)
{
    __shared__ int s_count;
    __shared__ int s_offset;
    if (threadIdx.x == 0)
        s_count = 0;
    __syncthreads();

    // Neighbors are in num_active x P array
    int neighbor = neighbor_data[blockIdx.x * P + threadIdx.x];

    // Get number of neighbors for this cell
    int n = -1;
    if (neighbor >= 0) // -1 indicates no neighbor
        n = atomicAdd(&s_count, 1);

    __syncthreads();

    // Get offset in global adjacency array to write to
    if (threadIdx.x == 0)
    {
        s_offset = atomicAdd(offsets_count, s_count);
        offsets[blockIdx.x] = s_offset;
        adjacency_sizes[blockIdx.x] = s_count;
    }
    __syncthreads();

    // Write neighbor to global adjacency array
    if (n >= 0)
        adjacency[s_offset + n] = neighbor;
}

static __global__ void sort_adjacency_kernel(const int *__restrict__ adjacency,
                                      const int *__restrict__ adjacency_offsets,
                                      const int *__restrict__ adjacency_sizes,
                                      const int *__restrict__ desired_adjacency_offsets,
                                      int *__restrict__ adjacency_out,
                                      int num_points)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_points)
        return;

    int in_offset = adjacency_offsets[gid];
    int size = adjacency_sizes[gid];
    int out_offset = desired_adjacency_offsets[gid];

    for (int i = 0; i < size; ++i)
    {
        adjacency_out[out_offset + i] = adjacency[in_offset + i];
    }
}

static __device__ bool connection_exists(int target, const int *adj, int start_offset, int count)
{
    for (int i = 0; i < count; i++)
    {
        if (adj[start_offset + i] == target)
        {
            return true;
        }
    }
    return false;
}

static __global__ void count_symmetrized_adjacency_sizes(int num_points,
                                                  const int *__restrict__ old_offsets,
                                                  const int *__restrict__ old_counts,
                                                  const int *__restrict__ old_adj,
                                                  int *__restrict__ new_counts)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_points)
        return;

    // 1. Every point keeps its existing outgoing connections
    // We add the original count to the new count.
    // atomicAdd(&new_counts[u], old_counts[u]);

    int u_start = old_offsets[u];
    int u_deg = old_counts[u];

    // 2. Check all neighbors of u
    for (int i = 0; i < u_deg; i++)
    {
        int v = old_adj[u_start + i];

        // We know u -> v exists.
        // We must check if v -> u exists.
        int v_start = old_offsets[v];
        int v_deg = old_counts[v];

        // Remove id the other cell is empty
        if (v_deg <= 0)
            continue;

        atomicAdd(&new_counts[u], 1);

        if (!connection_exists(u, old_adj, v_start, v_deg))
        {
            // v -> u is MISSING.
            // We must reserve space in v's list to add u.
            atomicAdd(&new_counts[v], 1);
        }
    }
}

static __global__ void populate_symmetric_adjacency(int num_points,
                                             const int *__restrict__ old_offsets,
                                             const int *__restrict__ old_counts,
                                             const int *__restrict__ old_adj,
                                             int *__restrict__ write_heads, // Initialized as a COPY of new_offsets
                                             int *__restrict__ new_adj      // The massive new array
)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_points)
        return;

    int u_start = old_offsets[u];
    int u_deg = old_counts[u];

    for (int i = 0; i < u_deg; i++)
    {
        int v = old_adj[u_start + i];

        // TASK B: Handle the reverse connection (v -> u)
        int v_start = old_offsets[v];
        int v_deg = old_counts[v];

        if (v_deg <= 0)
            continue;

        // TASK A: Copy the existing connection (u -> v)
        // We claim a spot in u's new list and write v
        int pos_u = atomicAdd(&write_heads[u], 1);
        new_adj[pos_u] = v;

        if (!connection_exists(u, old_adj, v_start, v_deg))
        {
            // v -> u was missing in the old graph.
            // We claim a spot in v's new list and write u.
            int pos_v = atomicAdd(&write_heads[v], 1);
            new_adj[pos_v] = u;
        }
    }
}

} // namespace laguerre
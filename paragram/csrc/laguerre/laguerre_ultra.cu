#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/torch.h>
#include <tuple>

#include "../utils/cuda_helpers.h"
#include "adjacency.cuh"
#include "chunking.h"
#include "faster_convex_cell.cuh"
#include "laguerre.h"
#include "pwr_bvh.cuh"

// Thrust Includes
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>

#define ASSERT(condition, message)                                                                                     \
    if (!(condition))                                                                                                  \
    {                                                                                                                  \
        throw std::runtime_error(message);                                                                             \
    }

namespace laguerre
{

// Kernel to combine points (Nx3) and weights (Nx1) into float4 array
__global__ void combine_points_weights_kernel(const float *__restrict__ points,
                                              const float *__restrict__ weights,
                                              float4 *__restrict__ output,
                                              int num_points)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_points)
        return;

    output[gid] = make_float4(points[gid * 3 + 0], points[gid * 3 + 1], points[gid * 3 + 2], weights[gid]);
}

__global__ void compute_bvh_power_diagram_kernel(
    int num_points,
    cuBQL::box_t<float, 3> world_bounds,
    const float4 *points,
    pwr_bvh::BinaryPwrBVH<float, 3> bvh,
    ConvexCellMemory cell_memory,       // External device memory (strided SoA)
    int *neighbors,                     // Output: [num_points * MAX_PLANES]
    Status *status,                     // Output: [num_points]
    int *initial_guesses_ids = nullptr, // Optional if provided will start by clipping by these planes
    int *initial_guesses_offsets = nullptr,
    int point_offset = 0)
{
    int idx = point_offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points)
        return;

    float4 p_data = points[idx];
    float3 p = make_float3(p_data.x, p_data.y, p_data.z);
    float w_i = p_data.w;

    pwr_bvh::BoxRadii initial_box_radii{1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};

    // Step 1: Set up Convex Cell memory from external strided device memory
    Status local_status = success;
    ConvexCell cell;
    cell.SetMemory(cell_memory, blockIdx.x, threadIdx.x);

    // Bounds: we can use world bounds or large bounds
    float3 min_bound = make_float3(world_bounds.lower.x, world_bounds.lower.y, world_bounds.lower.z);
    float3 max_bound = make_float3(world_bounds.upper.x, world_bounds.upper.y, world_bounds.upper.z);

    cell.CCInit(idx, points, &local_status, min_bound, max_bound);

    // Step 2: Clip by initial guesses if provided
    if (initial_guesses_ids != nullptr)
    {
        int offset = initial_guesses_offsets[idx];
        int num_guesses = initial_guesses_offsets[idx + 1] - offset;
        for (int i = 0; i < num_guesses; ++i)
        {
            int guess_idx = initial_guesses_ids[offset + i];
            if (guess_idx < 0)
                continue;
            float4 guess_p = points[guess_idx];

            cell.CCClipByPlane(guess_idx, guess_p);
        }
        // cell.CCUpdateRadius();
        cell.CCUpdateBounds();
        initial_box_radii = pwr_bvh::BoxRadii(cell.lower_bound, cell.upper_bound);
    }

    // Step 3: Clip by traversing BVH in shrinking radius pattern

    // Lambda for Node evaluation (pruning)
    auto nodeLambda = [&](const pwr_bvh::BinaryPwrBVH<float, 3>::node_t &node) -> float
    {
        float dist_sq = cuBQL::fSqrDistance_rd({p.x, p.y, p.z}, node.bounds);

        // 1. Handle overlapping or very close case
        // (Avoids division by zero later)
        if (dist_sq < 1e-12f)
            return 0.0f;

        float w_diff = w_i - node.max_weight;
        w_diff = w_diff > 0 ? 0 : w_diff;

        // 2. Determine the sign without sqrt
        // The original term was: (dist_sq + w_diff) / (2 * dist)
        // Since (2 * dist) is always positive, the sign depends only on (dist_sq + w_diff).
        float numerator = dist_sq + w_diff;

        // Equivalent to fmaxf(..., 0.0f) check
        if (numerator <= 0.0f)
            return 0.0f;

        // 3. Compute the square directly
        // Formula: ((d^2 + w_diff) / 2d)^2  ==  (numerator^2) / (4 * d^2)
        return (numerator * numerator) / (4.0f * dist_sq);
    };

    auto leafBoxLambda = [&](const uint32_t *leafPrims, int numPrims) -> pwr_bvh::BoxRadii
    {
        bool changed = false;
        for (int i = 0; i < numPrims; ++i)
        {
            int neigh_idx = leafPrims[i];
            if (neigh_idx == idx)
                continue;

            float4 neigh_p = points[neigh_idx];

            if (cell.CCClipByPlane(neigh_idx, neigh_p))
            {
                changed = true;
                cell.CCUpdateBounds();
            }
        }
        // cell.CCUpdateBounds();
        if (changed && cell.nb_plane >= MAX_PLANES * 0.85)
        {
            cell.CCGarbageCollect();
        }
        pwr_bvh::BoxRadii bounds(cell.lower_bound, cell.upper_bound);
        return bounds;
    };

    // Initial search radius: infinity
    bool traversal_success = pwr_bvh::shrinking_power_box_query(leafBoxLambda, nodeLambda, bvh, p, initial_box_radii);

    Status final_status = traversal_success ? local_status : security_radius_not_reached;

    // Write output
    // if (final_status == success)
    //{
    for (int v = 0; v < cell.nb_vertex; ++v)
    {
        uchar3 vertex = *cell.CCVertex(v);
        for (int i = 0; i < 3; ++i)
        {
            int plane = cell.CCIthPlane(v, i);
            int neigh = *cell.CCPlaneNeighbour(plane);
            if (neigh >= 0)
            {
                neighbors[idx * MAX_PLANES + plane] = neigh;
            }
        }
    }
    //}

    status[idx] = final_status;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_laguerre_voronoi_ultra(torch::Tensor points_in,
                             torch::Tensor weights_in,
                             int knn,
                             bool debug,
                             std::optional<torch::Tensor> initial_guesses_in,
                             std::optional<torch::Tensor> initial_guesses_offsets_in,
                             int leaf_size)
{
    // 0. Validate inputs
    int num_points = points_in.size(0);
    torch::Tensor points = points_in.contiguous();
    torch::Tensor weights = weights_in.contiguous();
    torch::Tensor initial_guesses;
    torch::Tensor initial_guesses_offsets;

    bool use_initial_guesses = false;
    if (initial_guesses_in.has_value() && initial_guesses_offsets_in.has_value())
    {
        use_initial_guesses = true;
        knn = 0;
        initial_guesses = initial_guesses_in->contiguous();
        initial_guesses_offsets = initial_guesses_offsets_in->contiguous();

        ASSERT(initial_guesses.device().type() == at::kCUDA, "initial_guesses must be on CUDA device");
        ASSERT(initial_guesses_offsets.device().type() == at::kCUDA, "initial_guesses_offsets must be on CUDA device");
        ASSERT(initial_guesses_offsets.size(0) == num_points + 1,
               "initial_guesses_offsets must have size num_points + 1");
    }

    ASSERT(points.device().type() == at::kCUDA, "points must be on CUDA device");
    ASSERT(weights.device().type() == at::kCUDA, "weights must be on CUDA device");
    ASSERT(points.scalar_type() == at::kFloat, "points must be of type float");
    ASSERT(weights.scalar_type() == at::kFloat, "weights must be of type float");

    ASSERT(points.size(1) == 3, "points must have 3 dimensions");
    ASSERT(weights.size(1) == 1, "weights must have 1 dimension");
    ASSERT(num_points == weights.size(0), "points and weights must have the same number of elements");

#ifdef MODERN_CUDA_CARDS
    if (leaf_size < 0)
        leaf_size = 10;
#else
    if (leaf_size < 0)
        leaf_size = 17;
    // For older cards, knn does not help much, so we disable it
    knn = 0;
#endif

    // configure_shared_memory(points.device().index());

    // 1. Prepare data
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    torch::Tensor points4 = torch::empty({num_points, 4}, options);
    {
        int grid_size = (num_points + 256 - 1) / 256;
        combine_points_weights_kernel<<<grid_size, 256>>>(points.data_ptr<float>(),
                                                          weights.data_ptr<float>(),
                                                          reinterpret_cast<float4 *>(points4.data_ptr<float>()),
                                                          num_points);
    }
    float4 *d_points4 = reinterpret_cast<float4 *>(points4.data_ptr<float>());

    // 2. Build BVH
    if (debug)
    {
        cudaDeviceSynchronize();
    }
    auto start_bvh = std::chrono::high_resolution_clock::now();

    // Use torch for KNN results allocation (better memory pool management)
    torch::Tensor knn_ids_tensor;
    int *d_knn_ids = nullptr;

    if (!use_initial_guesses && knn > 0)
    {
        knn_ids_tensor = torch::empty({num_points * knn}, torch::dtype(torch::kInt32).device(points.device()));
        d_knn_ids = knn_ids_tensor.data_ptr<int>();
    }

    pwr_bvh::BinaryPwrBVH<float, 3> bvh =
        pwr_bvh::build_pwr_bvh_and_knn(d_points4, points.data_ptr<float>(), num_points, knn, d_knn_ids, leaf_size);

    if (debug)
    {
        cudaDeviceSynchronize();
        auto end_bvh = std::chrono::high_resolution_clock::now();
        printf("BVH Creation time: %f ms\n", std::chrono::duration<float, std::milli>(end_bvh - start_bvh).count());
        pwr_bvh::print_leaf_statistics(bvh);
    }

    // Create offsets for knn
    torch::Tensor knn_offsets;
    if (!use_initial_guesses && knn > 0)
    {
        knn_offsets =
            torch::arange(0, num_points * knn + knn, knn, torch::dtype(torch::kInt32).device(points.device()));
    }

    // 3. Launch Kernel
    torch::Tensor neighbors =
        torch::full({num_points, MAX_PLANES}, -1, torch::dtype(torch::kInt32).device(points.device()));
    torch::Tensor status = torch::zeros({num_points}, torch::dtype(torch::kInt32).device(points.device()));

    // Compute world bounds for simple init
    cuBQL::box_t<float, 3> root_bounds;
    // We can copy root bounds from bvh.nodes[0].bounds
    cudaMemcpy(&root_bounds, &bvh.nodes[0].bounds, sizeof(cuBQL::box_t<float, 3>), cudaMemcpyDeviceToHost);

    // Enlarge slightly for safety
    root_bounds.lower = root_bounds.lower - 1.0f;
    root_bounds.upper = root_bounds.upper + 1.0f;

    if (debug)
    {
        cudaDeviceSynchronize();
    }
    auto start_laguerre = std::chrono::high_resolution_clock::now();

    int *d_initial_guesses_ptr = nullptr;
    int *d_initial_guesses_offsets_ptr = nullptr;

    if (use_initial_guesses)
    {
        d_initial_guesses_ptr = initial_guesses.data_ptr<int>();
        d_initial_guesses_offsets_ptr = initial_guesses_offsets.data_ptr<int>();
    }
    else if (knn > 0)
    {
        d_initial_guesses_ptr = d_knn_ids;
        d_initial_guesses_offsets_ptr = knn_offsets.data_ptr<int>();
    }

    // Allocate one reusable chunk of convex-cell scratch. Per-cell work is independent,
    // so chunking reduces peak transient memory without changing output.
    int grid_size = (num_points + CELL_BLOCK_STRIDE - 1) / CELL_BLOCK_STRIDE;
    int total_blocks = grid_size;
    size_t full_cell_memory_bytes =
        ConvexCellMemory::required_bytes(static_cast<size_t>(total_blocks) * CELL_BLOCK_STRIDE);
    ChunkPlan chunk_plan = plan_convex_cell_chunks(total_blocks, CELL_BLOCK_STRIDE, full_cell_memory_bytes);
    size_t cell_memory_bytes = ConvexCellMemory::required_bytes(chunk_plan.chunk_threads);

    // Use torch tensor for memory management (auto-freed when tensor goes out of scope)
    torch::Tensor cell_memory_buffer =
        torch::empty({static_cast<long>(cell_memory_bytes)}, torch::dtype(torch::kByte).device(points.device()));
    void *d_cell_memory_buffer = cell_memory_buffer.data_ptr();

    ConvexCellMemory cell_memory;
    cell_memory.setup_from_buffer(d_cell_memory_buffer, chunk_plan.chunk_threads);

    if (debug)
    {
        printf("ConvexCell Memory: %.2f MB/chunk for %d threads, %d chunk(s) over %d points\n",
               cell_memory_bytes / (1024.0f * 1024.0f),
               chunk_plan.chunk_threads,
               chunk_plan.num_chunks,
               num_points);
    }

    for (int c = 0; c < chunk_plan.num_chunks; ++c)
    {
        int point_offset = c * chunk_plan.chunk_blocks * CELL_BLOCK_STRIDE;
        if (point_offset >= num_points)
            break;

        int remaining_points = num_points - point_offset;
        int launch_blocks = (remaining_points + CELL_BLOCK_STRIDE - 1) / CELL_BLOCK_STRIDE;
        if (launch_blocks > chunk_plan.chunk_blocks)
            launch_blocks = chunk_plan.chunk_blocks;

        compute_bvh_power_diagram_kernel<<<launch_blocks, CELL_BLOCK_STRIDE>>>(
            num_points,
            root_bounds,
            d_points4,
            bvh,
            cell_memory,
            neighbors.data_ptr<int>(),
            reinterpret_cast<Status *>(status.data_ptr<int>()),
            d_initial_guesses_ptr,
            d_initial_guesses_offsets_ptr,
            point_offset);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // Clean up before throw
        pwr_bvh::free_bvh(bvh);
        throw std::runtime_error("Kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // This frees significant memory (nodes + primIDs arrays)
    pwr_bvh::free_bvh(bvh);
    knn_ids_tensor = torch::Tensor(); // Release memory back to PyTorch pool
    cell_memory_buffer = torch::Tensor();

    // 4. Compact adjacency
    torch::Tensor unsorted_adj_offsets =
        torch::zeros({num_points + 1}, torch::dtype(torch::kInt32).device(points.device()));
    torch::Tensor tmp_adjacency_sizes = torch::zeros({num_points}, torch::dtype(torch::kInt32).device(points.device()));
    torch::Tensor tmp_adj_offsets_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(points.device()));
    torch::Tensor tmp_adjacency =
        torch::zeros({num_points * MAX_PLANES}, torch::dtype(torch::kInt32).device(points.device()));

    gather_adjacency_kernel<<<num_points, MAX_PLANES>>>(MAX_PLANES,
                                                        num_points,
                                                        neighbors.data_ptr<int>(),
                                                        tmp_adjacency.data_ptr<int>(),
                                                        unsorted_adj_offsets.data_ptr<int>(),
                                                        tmp_adjacency_sizes.data_ptr<int>(),
                                                        tmp_adj_offsets_count.data_ptr<int>());

    torch::Tensor adjacency_offsets_out;
    torch::Tensor adjacency_out;
    if (false)
    {
        // Create Offsets
        torch::Tensor adj_sizes_extended =
            torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(points.device())), tmp_adjacency_sizes},
                       0);
        // cumsum returns Long by default, convert back to Int32
        adjacency_offsets_out = torch::cumsum(adj_sizes_extended, 0).to(torch::kInt32);

        // Create Adjacency
        int adjacency_size;
        cuda_check(
            cudaMemcpy(&adjacency_size, tmp_adj_offsets_count.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
        adjacency_out = torch::zeros({adjacency_size}, torch::dtype(torch::kInt32).device(points.device()));

        // Cumsum of adjacency sizes to get sorted offsets
        sort_adjacency_kernel<<<(num_points + 31) / 32, 32>>>(tmp_adjacency.data_ptr<int>(),
                                                              unsorted_adj_offsets.data_ptr<int>(),
                                                              tmp_adjacency_sizes.data_ptr<int>(),
                                                              adjacency_offsets_out.data_ptr<int>(),
                                                              adjacency_out.data_ptr<int>(),
                                                              num_points);
    }
    else
    {
        int blockSize = 256;
        int numBlocks = (num_points + blockSize - 1) / blockSize;
        torch::Tensor new_adjacency_sizes =
            torch::zeros({num_points}, torch::dtype(torch::kInt32).device(points.device()));
        count_symmetrized_adjacency_sizes<<<numBlocks, blockSize>>>(num_points,
                                                                    unsorted_adj_offsets.data_ptr<int>(),
                                                                    tmp_adjacency_sizes.data_ptr<int>(),
                                                                    tmp_adjacency.data_ptr<int>(),
                                                                    new_adjacency_sizes.data_ptr<int>());

        torch::Tensor adj_sizes_extended =
            torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(points.device())), new_adjacency_sizes},
                       0);
        adjacency_offsets_out = torch::cumsum(adj_sizes_extended, 0).to(torch::kInt32);

        int new_adjacency_size;
        int *d_new_adjacency_count = adjacency_offsets_out.data_ptr<int>() + num_points;
        cuda_check(cudaMemcpy(&new_adjacency_size, d_new_adjacency_count, sizeof(int), cudaMemcpyDeviceToHost));

        adjacency_out = torch::empty({new_adjacency_size}, torch::dtype(torch::kInt32).device(points.device()));
        torch::Tensor write_heads = adjacency_offsets_out.clone();
        populate_symmetric_adjacency<<<numBlocks, blockSize>>>(num_points,
                                                               unsorted_adj_offsets.data_ptr<int>(),
                                                               tmp_adjacency_sizes.data_ptr<int>(),
                                                               tmp_adjacency.data_ptr<int>(),
                                                               write_heads.data_ptr<int>(),
                                                               adjacency_out.data_ptr<int>());
    }

    cudaDeviceSynchronize();

    if (debug)
    {
        auto end_laguerre = std::chrono::high_resolution_clock::now();
        printf("Laguerre Construction time: %f ms\n",
               std::chrono::duration<float, std::milli>(end_laguerre - start_laguerre).count());
    }

    return std::make_tuple(adjacency_out, adjacency_offsets_out, status);
}

} // namespace laguerre

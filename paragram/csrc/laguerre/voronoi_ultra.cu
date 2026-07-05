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
#include "laguerre.h"
#include "voronoi_bvh.cuh"
#include "voronoi_convex_cell.cuh"

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

// Pack float3 points into float4 with w=0 (no weight)
__global__ void pack_points_kernel(const float *__restrict__ points, float4 *__restrict__ output, int num_points)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_points)
        return;

    output[gid] = make_float4(points[gid * 3 + 0], points[gid * 3 + 1], points[gid * 3 + 2], 0.0f);
}

__global__ void compute_bvh_voronoi_kernel(
    int num_points,
    cuBQL::box_t<float, 3> world_bounds,
    const float4 *points,
    cuBQL::BinaryBVH<float, 3> bvh,
    ConvexCellMemory cell_memory,
    int *neighbors,
    Status *status,
    int *initial_guesses_ids = nullptr,
    int *initial_guesses_offsets = nullptr,
    int point_offset = 0)
{
    int idx = point_offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points)
        return;

    float4 p_data = points[idx];
    float3 p = make_float3(p_data.x, p_data.y, p_data.z);

    voronoi_bvh::BoxRadii initial_box_radii{1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10};

    Status local_status = success;
    ConvexCell cell;
    cell.SetMemory(cell_memory, blockIdx.x, threadIdx.x);

    float3 min_bound = make_float3(world_bounds.lower.x, world_bounds.lower.y, world_bounds.lower.z);
    float3 max_bound = make_float3(world_bounds.upper.x, world_bounds.upper.y, world_bounds.upper.z);

    cell.CCInit(idx, points, &local_status, min_bound, max_bound);

    // Clip by initial guesses if provided
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
        cell.CCUpdateBounds();
        initial_box_radii = voronoi_bvh::BoxRadii(cell.lower_bound, cell.upper_bound);
    }

    // Node lambda: pure squared Euclidean distance / 4
    // For unweighted Voronoi, the bisector plane between seed and a point at
    // distance d is at distance d/2 from seed. Squared: d^2/4.
    auto nodeLambda = [&](const auto &node) -> float
    {
        float dist_sq = cuBQL::fSqrDistance_rd({p.x, p.y, p.z}, node.bounds);
        return dist_sq * 0.25f;
    };

    auto leafBoxLambda = [&](const uint32_t *leafPrims, int numPrims) -> voronoi_bvh::BoxRadii
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
        if (changed && cell.nb_plane >= MAX_PLANES * 0.85)
        {
            cell.CCGarbageCollect();
        }
        voronoi_bvh::BoxRadii bounds(cell.lower_bound, cell.upper_bound);
        return bounds;
    };

    bool traversal_success = voronoi_bvh::shrinking_voronoi_box_query(leafBoxLambda, nodeLambda, bvh, p, initial_box_radii);

    Status final_status = traversal_success ? local_status : security_radius_not_reached;

    // Write output
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

    status[idx] = final_status;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_voronoi_ultra(torch::Tensor points_in,
                    int knn,
                    bool debug,
                    std::optional<torch::Tensor> initial_guesses_in,
                    std::optional<torch::Tensor> initial_guesses_offsets_in,
                    int leaf_size)
{
    // 0. Validate inputs
    int num_points = points_in.size(0);
    torch::Tensor points = points_in.contiguous();
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
    ASSERT(points.scalar_type() == at::kFloat, "points must be of type float");
    ASSERT(points.size(1) == 3, "points must have 3 dimensions");

#ifdef MODERN_CUDA_CARDS
    if (leaf_size < 0)
        leaf_size = 10;
#else
    if (leaf_size < 0)
        leaf_size = 17;
    knn = 0;
#endif

    // 1. Pack points as float4 with w=0
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    torch::Tensor points4 = torch::empty({num_points, 4}, options);
    {
        int grid_size = (num_points + 256 - 1) / 256;
        pack_points_kernel<<<grid_size, 256>>>(points.data_ptr<float>(),
                                               reinterpret_cast<float4 *>(points4.data_ptr<float>()),
                                               num_points);
    }
    float4 *d_points4 = reinterpret_cast<float4 *>(points4.data_ptr<float>());

    // 2. Build BVH (cuBQL directly, no weight propagation)
    if (debug)
    {
        cudaDeviceSynchronize();
    }
    auto start_bvh = std::chrono::high_resolution_clock::now();

    torch::Tensor knn_ids_tensor;
    int *d_knn_ids = nullptr;

    if (!use_initial_guesses && knn > 0)
    {
        knn_ids_tensor = torch::empty({num_points * knn}, torch::dtype(torch::kInt32).device(points.device()));
        d_knn_ids = knn_ids_tensor.data_ptr<int>();
    }

    cuBQL::BinaryBVH<float, 3> bvh =
        pwr_bvh::build_cubql_bvh_and_knn(d_points4, points.data_ptr<float>(), num_points, knn, d_knn_ids, leaf_size);

    if (debug)
    {
        cudaDeviceSynchronize();
        auto end_bvh = std::chrono::high_resolution_clock::now();
        printf("BVH Creation time: %f ms\n", std::chrono::duration<float, std::milli>(end_bvh - start_bvh).count());
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

    cuBQL::box_t<float, 3> root_bounds;
    cudaMemcpy(&root_bounds, &bvh.nodes[0].bounds, sizeof(cuBQL::box_t<float, 3>), cudaMemcpyDeviceToHost);

    root_bounds.lower = root_bounds.lower - 1.0f;
    root_bounds.upper = root_bounds.upper + 1.0f;

    if (debug)
    {
        cudaDeviceSynchronize();
    }
    auto start_voronoi = std::chrono::high_resolution_clock::now();

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

    int grid_size = (num_points + CELL_BLOCK_STRIDE - 1) / CELL_BLOCK_STRIDE;
    int total_blocks = grid_size;
    size_t full_cell_memory_bytes =
        ConvexCellMemory::required_bytes(static_cast<size_t>(total_blocks) * CELL_BLOCK_STRIDE);
    ChunkPlan chunk_plan = plan_convex_cell_chunks(total_blocks, CELL_BLOCK_STRIDE, full_cell_memory_bytes);
    size_t cell_memory_bytes = ConvexCellMemory::required_bytes(chunk_plan.chunk_threads);

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

        compute_bvh_voronoi_kernel<<<launch_blocks, CELL_BLOCK_STRIDE>>>(
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
        voronoi_bvh::free_bvh(bvh);
        throw std::runtime_error("Kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    voronoi_bvh::free_bvh(bvh);
    knn_ids_tensor = torch::Tensor();
    cell_memory_buffer = torch::Tensor();

    // 4. Compact adjacency (symmetrized)
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
        auto end_voronoi = std::chrono::high_resolution_clock::now();
        printf("Voronoi Construction time: %f ms\n",
               std::chrono::duration<float, std::milli>(end_voronoi - start_voronoi).count());
    }

    return std::make_tuple(adjacency_out, adjacency_offsets_out, status);
}

} // namespace laguerre

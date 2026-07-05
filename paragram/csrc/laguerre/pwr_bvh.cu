#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// Define implementation for cuBQL builder - MUST come before any cuBQL include
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "pwr_bvh.cuh"

#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/pointData/knn.h"
// #include "cuBQL/builder/cuda/radixBuilder.h" // Does not exist, included by cuda.h

namespace pwr_bvh
{

#define MAX_STACK_K 128

using vec3f = cuBQL::vec_t<float, 3>;

// Define Node type alias for convenience within this file
using Node = BinaryPwrBVH<float, 3>::Node;

// Kernel to create boxes from float4 points
static __global__ void create_boxes_kernel(cuBQL::box_t<float, 3> *boxes, const float4 *points, int numPoints)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numPoints)
        return;

    // Extract position from float4 (x,y,z,w)
    float4 p = points[gid];
    cuBQL::vec_t<float, 3> pos;
    pos.x = p.x;
    pos.y = p.y;
    pos.z = p.z;

    boxes[gid].lower = pos;
    boxes[gid].upper = pos;
}

// Helper to copy data and init
__global__ void init_pwr_bvh_kernel(const cuBQL::BinaryBVH<float, 3>::node_t *__restrict__ src_nodes,
                                    Node *__restrict__ dst_nodes,
                                    int num_nodes,
                                    const float4 *__restrict__ points,
                                    const uint32_t *__restrict__ d_primIDs,
                                    int *__restrict__ counters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_nodes)
        return;

    // Zero out counters for propagation
    if (counters)
        counters[gid] = 0;

    // Copy basic data
    const auto &src = src_nodes[gid];
    dst_nodes[gid].bounds = src.bounds;
    dst_nodes[gid].admin.offsetAndCountBits = src.admin.offsetAndCountBits;

    // Initialize weight
    if (src.admin.count > 0) // Leaf
    {
        // Leaf node: max_weight is the max weight of primitives in this leaf
        // Usually leaf has 1 primitive for point clouds, but cuBQL supports more.
        // We will iterate them.
        float max_w = -INFINITY;
        int count = src.admin.count;
        int offset = src.admin.offset;

        for (int i = 0; i < count; ++i)
        {
            uint32_t primIdx = d_primIDs[offset + i];
            float w = points[primIdx].w; // point is float4, w is weight
            max_w = fmaxf(max_w, w);
        }
        dst_nodes[gid].max_weight = max_w;
    }
    else
    {
        // Internal node: init to -inf, will be computed by propagation
        dst_nodes[gid].max_weight = -INFINITY;
    }
}

// Kernel to compute parent pointers
__global__ void compute_parents_kernel(const Node *__restrict__ nodes, int num_nodes, int *__restrict__ parents)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_nodes)
        return;

    if (nodes[gid].admin.count == 0) // Internal node
    {
        int left = nodes[gid].admin.offset;
        int right = left + 1;

        // Safety check
        if (left < num_nodes)
            parents[left] = gid;
        if (right < num_nodes)
            parents[right] = gid;
    }

    // Root's parent is initialized to -1 by memset or we can handle it.
    // We rely on parents being -1 initially.
    if (gid == 0)
        parents[gid] = -1;
}

// Bottom-up propagation kernel
// WARNING: This kernel will only work for binary BVHs
__global__ void propagate_weights_kernel(Node *__restrict__ nodes,
                                         int num_nodes,
                                         const int *__restrict__ parents,
                                         int *__restrict__ counters)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_nodes)
        return;

    // Start only from leaves
    if (nodes[gid].admin.count > 0)
    {
        int curr = gid;
        while (true)
        {
            int parent = parents[curr];
            if (parent == -1)
                break; // Reached root

            // Atomic increment
            int old = atomicAdd(&counters[parent], 1);
            if (old == 0)
            {
                // First child arrived, we are done.
                break;
            }
            else
            {
                // Second child arrived (old == 1). We process the parent.
                // Parent must be an internal node with children at offset, offset+1
                int left = nodes[parent].admin.offset;
                int right = left + 1;

                float w_left = nodes[left].max_weight;
                float w_right = nodes[right].max_weight;

                nodes[parent].max_weight = fmaxf(w_left, w_right);

                // Continue up
                curr = parent;
            }
        }
    }
}

__global__ void cuBQL_knn_kernel(int num_points,
                                 int idOffset,
                                 const vec3f *__restrict__ points,
                                 cuBQL::BinaryBVH<float, 3> bvh,
                                 int *__restrict__ results,
                                 cuBQL::knn::Candidate *globalCandidates, // Pre-allocated for large K (can be null)
                                 int K)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_points)
        return;

    const int requestK = K + 1; // +1 for self-exclusion

    // Use stack for small K, global memory for large K
    cuBQL::knn::Candidate stackCandidates[MAX_STACK_K];
    cuBQL::knn::Candidate *candidates;

    if (requestK <= MAX_STACK_K)
    {
        candidates = stackCandidates;
    }
    else
    {
        candidates = globalCandidates + (size_t)gid * requestK;
    }

    vec3f queryPoint = points[gid];

    // Initialize candidates
    for (int i = 0; i < requestK; i++)
    {
        candidates[i].primID = -1;
        candidates[i].sqrDist = INFINITY;
    }

    // Use cuBQL's built-in findKNN function
    cuBQL::knn::Result result =
        cuBQL::points::findKNN<float, 3>(candidates, requestK, bvh, points, queryPoint, INFINITY);

    // Copy results to output, excluding self (the query point itself)
    int outputIdx = 0;
    for (int i = 0; i < result.numFound && outputIdx < K; i++)
    {
        if (candidates[i].primID != gid)
        {
            results[gid * K + outputIdx] = candidates[i].primID;
            outputIdx++;
        }
    }

    // Fill remaining slots with a valid but distant index
    unsigned int fillValue = (outputIdx > 0) ? results[gid * K + outputIdx - 1] : 0;
    for (; outputIdx < K; outputIdx++)
    {
        results[gid * K + outputIdx] = fillValue;
    }
}

BinaryPwrBVH<float, 3>
build_pwr_bvh_and_knn(const float4 *points, float *points3, int num_points, int K, int *d_knn_ids, int leaf_size)
{
    BinaryPwrBVH<float, 3> bvh;

    if (num_points == 0)
        return bvh;

    // 1. Build standard cuBQL BVH
    cuBQL::BinaryBVH<float, 3> cubql_bvh;
    cuBQL::box_t<float, 3> *d_boxes;
    cudaMalloc(&d_boxes, num_points * sizeof(cuBQL::box_t<float, 3>));

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    create_boxes_kernel<<<numBlocks, blockSize>>>(d_boxes, points, num_points);

    cuBQL::BuildConfig buildConfig;
    // buildConfig.enableELH();
    buildConfig.makeLeafThreshold = leaf_size;
    // buildConfig.enableELH();
    // buildConfig.maxAllowedLeafSize = 1 << 5;
    // cuBQL::cuda::radixBuilder(cubql_bvh, d_boxes, num_points, buildConfig);
    cuBQL::gpuBuilder(cubql_bvh, d_boxes, num_points, buildConfig);

    cudaFree(d_boxes);

    // 2. Build KNN
    if (K > 0)
    {
        cuBQL_knn_kernel<<<numBlocks, blockSize>>>(num_points,
                                                   0,
                                                   reinterpret_cast<vec3f *>(points3),
                                                   cubql_bvh,
                                                   d_knn_ids,
                                                   nullptr,
                                                   K);
    }

    // 3. Allocate PwrBVH
    bvh.numNodes = cubql_bvh.numNodes;
    bvh.numPrims = cubql_bvh.numPrims;

    cudaMalloc(&bvh.nodes, bvh.numNodes * sizeof(Node));
    cudaMalloc(&bvh.primIDs, bvh.numPrims * sizeof(uint32_t));

    // Copy primIDs directly
    cudaMemcpy(bvh.primIDs, cubql_bvh.primIDs, bvh.numPrims * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // 4. Prepare for weight propagation
    int *d_parents;
    int *d_counters;
    cudaMalloc(&d_parents, bvh.numNodes * sizeof(int));
    cudaMalloc(&d_counters, bvh.numNodes * sizeof(int));
    cudaMemset(d_parents, 0xFF, bvh.numNodes * sizeof(int)); // Init to -1

    // 5. Init PwrBVH nodes (copy structure + init leaf weights)
    numBlocks = (bvh.numNodes + blockSize - 1) / blockSize;
    init_pwr_bvh_kernel<<<numBlocks, blockSize>>>(cubql_bvh.nodes,
                                                  bvh.nodes,
                                                  bvh.numNodes,
                                                  points,
                                                  bvh.primIDs,
                                                  d_counters // Init counters to 0 here
    );

    // 6. Compute parents
    compute_parents_kernel<<<numBlocks, blockSize>>>(bvh.nodes, bvh.numNodes, d_parents);

    // 7. Propagate weights
    propagate_weights_kernel<<<numBlocks, blockSize>>>(bvh.nodes, bvh.numNodes, d_parents, d_counters);

    // Cleanup temporary buffers
    cudaFree(d_parents);
    cudaFree(d_counters);

    // Free original cuBQL BVH
    // Note: cuBQL::free just frees the pointers inside the struct, but cuBQL struct itself is on stack here.
    // But we need to call cuBQL::cuda::free equivalent?
    // Usually cuBQL provides a free function or we just free the pointers if we know what they are.
    // cuBQL::cuda::free(cubql_bvh); // Assuming this exists or we can just free nodes/primIDs
    if (cubql_bvh.nodes)
        cudaFree(cubql_bvh.nodes);
    if (cubql_bvh.primIDs)
        cudaFree(cubql_bvh.primIDs);

    return bvh;
}

cuBQL::BinaryBVH<float, 3>
build_cubql_bvh_and_knn(const float4 *points, float *points3, int num_points, int K, int *d_knn_ids, int leaf_size)
{
    cuBQL::BinaryBVH<float, 3> bvh;

    if (num_points == 0)
        return bvh;

    // 1. Build standard cuBQL BVH
    cuBQL::box_t<float, 3> *d_boxes;
    cudaMalloc(&d_boxes, num_points * sizeof(cuBQL::box_t<float, 3>));

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    create_boxes_kernel<<<numBlocks, blockSize>>>(d_boxes, points, num_points);

    cuBQL::BuildConfig buildConfig;
    buildConfig.makeLeafThreshold = leaf_size;
    cuBQL::gpuBuilder(bvh, d_boxes, num_points, buildConfig);

    cudaFree(d_boxes);

    // 2. Build KNN if requested
    if (K > 0)
    {
        cuBQL_knn_kernel<<<numBlocks, blockSize>>>(num_points,
                                                   0,
                                                   reinterpret_cast<vec3f *>(points3),
                                                   bvh,
                                                   d_knn_ids,
                                                   nullptr,
                                                   K);
    }

    return bvh;
}

// Kernel to collect leaf counts
__global__ void collect_leaf_counts_kernel(const Node *nodes, int num_nodes, int *leaf_counts, int *num_leaves)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_nodes)
        return;

    if (nodes[gid].admin.count > 0) // Leaf node
    {
        int leaf_idx = atomicAdd(num_leaves, 1);
        if (leaf_idx < num_nodes) // Safety check
        {
            leaf_counts[leaf_idx] = nodes[gid].admin.count;
        }
    }
}

void print_leaf_statistics(const BinaryPwrBVH<float, 3> &bvh)
{
    if (bvh.numNodes == 0 || bvh.nodes == nullptr)
    {
        printf("BVH Leaf Statistics: BVH is empty\n");
        return;
    }

    // Allocate device memory for leaf counts
    int *d_leaf_counts;
    int *d_num_leaves;
    cudaError_t err;
    err = cudaMalloc(&d_leaf_counts, bvh.numNodes * sizeof(int));
    if (err != cudaSuccess)
    {
        printf("BVH Leaf Statistics: Failed to allocate device memory\n");
        return;
    }
    err = cudaMalloc(&d_num_leaves, sizeof(int));
    if (err != cudaSuccess)
    {
        printf("BVH Leaf Statistics: Failed to allocate device memory\n");
        cudaFree(d_leaf_counts);
        return;
    }
    cudaMemset(d_num_leaves, 0, sizeof(int));

    // Collect leaf counts
    int blockSize = 256;
    int numBlocks = (bvh.numNodes + blockSize - 1) / blockSize;
    collect_leaf_counts_kernel<<<numBlocks, blockSize>>>(bvh.nodes, bvh.numNodes, d_leaf_counts, d_num_leaves);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("BVH Leaf Statistics: Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_leaf_counts);
        cudaFree(d_num_leaves);
        return;
    }

    // Get number of leaves
    int num_leaves = 0;
    err = cudaMemcpy(&num_leaves, d_num_leaves, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("BVH Leaf Statistics: Failed to copy data from device\n");
        cudaFree(d_leaf_counts);
        cudaFree(d_num_leaves);
        return;
    }

    if (num_leaves == 0)
    {
        printf("BVH Leaf Statistics: No leaves found\n");
        cudaFree(d_leaf_counts);
        cudaFree(d_num_leaves);
        return;
    }

    // Copy leaf counts to host
    std::vector<int> leaf_counts(num_leaves);
    err = cudaMemcpy(leaf_counts.data(), d_leaf_counts, num_leaves * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("BVH Leaf Statistics: Failed to copy leaf counts from device\n");
        cudaFree(d_leaf_counts);
        cudaFree(d_num_leaves);
        return;
    }

    // Compute statistics
    int min_count = leaf_counts[0];
    int max_count = leaf_counts[0];
    long long sum = 0;

    for (int i = 0; i < num_leaves; ++i)
    {
        min_count = (leaf_counts[i] < min_count) ? leaf_counts[i] : min_count;
        max_count = (leaf_counts[i] > max_count) ? leaf_counts[i] : max_count;
        sum += leaf_counts[i];
    }

    double mean_count = (double)sum / num_leaves;

    // Print statistics
    printf("BVH Leaf Statistics:\n");
    printf("  Total leaves: %d\n", num_leaves);
    printf("  Min points per leaf: %d\n", min_count);
    printf("  Max points per leaf: %d\n", max_count);
    printf("  Mean points per leaf: %.2f\n", mean_count);
    printf("  Total points in leaves: %lld\n", sum);

    cudaFree(d_leaf_counts);
    cudaFree(d_num_leaves);
}

} // namespace pwr_bvh

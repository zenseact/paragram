#pragma once
// Weightless BVH traversal for standard Voronoi diagrams.
// Uses cuBQL::BinaryBVH directly — no weight propagation needed.

#include <cuda_runtime.h>

#include <cuBQL/bvh.h>
#include <cuBQL/math/conservativeDistances.h>

// Forward declaration: BVH build function (implemented in pwr_bvh.cu)
namespace pwr_bvh
{
cuBQL::BinaryBVH<float, 3>
build_cubql_bvh_and_knn(const float4 *points, float *points3, int num_points, int K, int *d_knn_ids, int leaf_size);
} // namespace pwr_bvh

namespace voronoi_bvh
{

inline void free_bvh(cuBQL::BinaryBVH<float, 3> &bvh)
{
    if (bvh.nodes)
    {
        cudaFree(bvh.nodes);
        bvh.nodes = nullptr;
    }
    if (bvh.primIDs)
    {
        cudaFree(bvh.primIDs);
        bvh.primIDs = nullptr;
    }
    bvh.numNodes = 0;
    bvh.numPrims = 0;
}

// ******************************************************************
// BoxRadii — squared distances from seed to Voronoi cell bounding box corners
// ******************************************************************

struct BoxRadii
{
    float A, B, C, D, E, F, G, H;

    __device__ __forceinline__ BoxRadii(float a, float b, float c, float d, float e, float f, float g, float h)
    {
        A = a;
        B = b;
        C = c;
        D = d;
        E = e;
        F = f;
        G = g;
        H = h;
    }

    __device__ __forceinline__ BoxRadii(float3 lower, float3 upper)
    {
        A = lower.x * lower.x + lower.y * lower.y + lower.z * lower.z;
        B = upper.x * upper.x + lower.y * lower.y + lower.z * lower.z;
        C = lower.x * lower.x + upper.y * upper.y + lower.z * lower.z;
        D = upper.x * upper.x + upper.y * upper.y + lower.z * lower.z;
        E = lower.x * lower.x + lower.y * lower.y + upper.z * upper.z;
        F = upper.x * upper.x + lower.y * lower.y + upper.z * upper.z;
        G = lower.x * lower.x + upper.y * upper.y + upper.z * upper.z;
        H = upper.x * upper.x + upper.y * upper.y + upper.z * upper.z;
    }

    __device__ __forceinline__ static BoxRadii Invalid()
    {
        return BoxRadii(-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f);
    }

    __device__ __forceinline__ bool is_valid() { return A >= 0.0f; }
};

template <typename VecType>
__device__ __forceinline__ float
get_directional_safety_radius(BoxRadii seed_radii, float3 seed_point, VecType bounds_min, VecType bounds_max)
{
    bool i0 = bounds_min.x <= seed_point.x;
    bool i1 = bounds_min.y <= seed_point.y;
    bool i2 = bounds_min.z <= seed_point.z;
    bool i3 = bounds_max.x >= seed_point.x;
    bool i4 = bounds_max.y >= seed_point.y;
    bool i5 = bounds_max.z >= seed_point.z;

    bool c0 = i0 & i1 & i2;
    bool c1 = i3 & i1 & i2;
    bool c2 = i0 & i4 & i2;
    bool c3 = i3 & i4 & i2;
    bool c4 = i0 & i1 & i5;
    bool c5 = i3 & i1 & i5;
    bool c6 = i0 & i4 & i5;
    bool c7 = i3 & i4 & i5;

    float v0 = c0 ? seed_radii.A : 0.0f;
    float v1 = c1 ? seed_radii.B : 0.0f;
    float v2 = c2 ? seed_radii.C : 0.0f;
    float v3 = c3 ? seed_radii.D : 0.0f;
    float v4 = c4 ? seed_radii.E : 0.0f;
    float v5 = c5 ? seed_radii.F : 0.0f;
    float v6 = c6 ? seed_radii.G : 0.0f;
    float v7 = c7 ? seed_radii.H : 0.0f;

    float m01 = fmaxf(v0, v1);
    float m23 = fmaxf(v2, v3);
    float m45 = fmaxf(v4, v5);
    float m67 = fmaxf(v6, v7);
    float m03 = fmaxf(m01, m23);
    float m47 = fmaxf(m45, m67);
    return fmaxf(m03, m47);
}

__device__ inline void voronoi_stack_push(uint32_t *stack_idx,
                                          float *stack_dist,
                                          int &stack_top,
                                          int cap,
                                          uint32_t idx,
                                          float dist)
{
    if (stack_top < cap)
    {
        stack_idx[stack_top] = idx;
        stack_dist[stack_top] = dist;
        stack_top++;
        return;
    }

    int far_pos = 0;
    float far_dist = stack_dist[0];
    for (int i = 1; i < cap; i++)
    {
        if (stack_dist[i] > far_dist)
        {
            far_dist = stack_dist[i];
            far_pos = i;
        }
    }
    if (dist < far_dist)
    {
        stack_idx[far_pos] = idx;
        stack_dist[far_pos] = dist;
    }
}

// ******************************************************************
// Traversal — shrinking box query for standard Voronoi
// Uses an unsorted stack with min-extraction on pop.
// Push is O(1) (append). Pop scans for the entry with minimum dist
// and swap-removes it — this preserves nearest-first traversal order
// (critical for pruning) without the O(N) insertion shift of SortedStack.
// ******************************************************************

template <typename LeafLambda, typename NodeLambda>
inline __device__ bool shrinking_voronoi_box_query(const LeafLambda &lambdaToExecuteForEachCandidateLeaf,
                                                   const NodeLambda &lambdaToExecuteForEachCandidateNode,
                                                   cuBQL::BinaryBVH<float, 3> bvh,
                                                   float3 seed_point,
                                                   voronoi_bvh::BoxRadii seed_radii)
{
    const int stackSize = 64;
    uint32_t stack_idx[stackSize];
    float stack_dist[stackSize];
    int stack_top = 0;

    if (bvh.numNodes == 0)
        return false;

    auto node = bvh.nodes[0].admin;

#ifndef MODERN_CUDA_CARDS
    // Traverse to first leaf (for older cards)
    while (node.count == 0)
    {
        uint32_t n0Idx = (uint32_t)node.offset + 0;
        uint32_t n1Idx = (uint32_t)node.offset + 1;
        const auto &n0 = bvh.nodes[n0Idx];
        const auto &n1 = bvh.nodes[n1Idx];

        float d0 = lambdaToExecuteForEachCandidateNode(n0);
        float d1 = lambdaToExecuteForEachCandidateNode(n1);

        if (d0 < d1)
        {
            node = n0.admin;
            voronoi_stack_push(stack_idx, stack_dist, stack_top, stackSize, n1Idx, d1);
        }
        else
        {
            node = n1.admin;
            voronoi_stack_push(stack_idx, stack_dist, stack_top, stackSize, n0Idx, d0);
        }
    }
#endif

    while (true)
    {
        // Traverse inner nodes downward
        while (node.count == 0)
        {
            uint32_t n0Idx = (uint32_t)node.offset + 0;
            uint32_t n1Idx = (uint32_t)node.offset + 1;
            const auto &n0 = bvh.nodes[n0Idx];
            const auto &n1 = bvh.nodes[n1Idx];

            float r0 = get_directional_safety_radius(seed_radii, seed_point, n0.bounds.lower, n0.bounds.upper);
            float r1 = get_directional_safety_radius(seed_radii, seed_point, n1.bounds.lower, n1.bounds.upper);
            float d0 = lambdaToExecuteForEachCandidateNode(n0);
            float d1 = lambdaToExecuteForEachCandidateNode(n1);
            float diff0 = d0 - r0;
            float diff1 = d1 - r1;

            if (fminf(diff0, diff1) > 0.f)
            {
                // Both children pruned
                node.count = 0;
                break;
            }

            // Descend into nearer child, push far child
            uint32_t farIdx;
            float farDist;
            if (diff0 < diff1)
            {
                node = n0.admin;
                farIdx = n1Idx;
                farDist = d1;
            }
            else
            {
                node = n1.admin;
                farIdx = n0Idx;
                farDist = d0;
            }

            if (fmaxf(diff0, diff1) < 0.f)
            {
                voronoi_stack_push(stack_idx, stack_dist, stack_top, stackSize, farIdx, farDist);
            }
        }

        if (node.count != 0)
        {
            seed_radii = lambdaToExecuteForEachCandidateLeaf(bvh.primIDs + node.offset, node.count);
            if (!seed_radii.is_valid())
                return true;
        }

        // Pop nearest valid node from stack (min-extraction)
        bool found = false;
        while (stack_top > 0)
        {
            // Find entry with minimum dist
            int min_pos = 0;
            float min_dist = stack_dist[0];
            for (int i = 1; i < stack_top; i++)
            {
                if (stack_dist[i] < min_dist)
                {
                    min_dist = stack_dist[i];
                    min_pos = i;
                }
            }

            // Extract it (swap with last)
            uint32_t idx = stack_idx[min_pos];
            float dist = stack_dist[min_pos];
            stack_top--;
            if (min_pos < stack_top)
            {
                stack_idx[min_pos] = stack_idx[stack_top];
                stack_dist[min_pos] = stack_dist[stack_top];
            }

            // Check validity with current safety radius
            float r = get_directional_safety_radius(seed_radii,
                                                    seed_point,
                                                    bvh.nodes[idx].bounds.lower,
                                                    bvh.nodes[idx].bounds.upper);

            if (dist - r <= 0.f)
            {
                node = bvh.nodes[idx].admin;
                found = true;
                break;
            }
            // Pruned — continue to next nearest
        }
        if (!found)
            return true; // Stack empty, done
    }
    return false;
}

} // namespace voronoi_bvh

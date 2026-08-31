#pragma once
// Weightless fork of faster_convex_cell.cuh for standard Voronoi diagrams.
// Changes from original:
//   - CCComputeBisectorPlane: removed weight terms (standard perpendicular bisector)
//   - CCIsPointTooFar: simplified to pure squared distance check

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

using uchar = unsigned char;
using uchar3 = uchar3;
using uchar4 = uchar4;

#define END_OF_LIST 255
#define NO_CELL_LIMITS -1e10f
#define CUBE_EPSILON 1e-5f
#define VOLUME_EPSILON2 1e-10f

#define MAX_PLANES 64
#define MAX_VERTS 64

#define CELL_BLOCK_STRIDE 32

namespace laguerre
{

struct ConvexCellMemory
{
    uchar4 *vertices;
    float4 *vertex_pos;
    uchar *next_boundaries;
    float4 *planes;
    int *planes_neighbour;

    __host__ static size_t required_bytes(int num_threads)
    {
        return num_threads * (MAX_VERTS * sizeof(uchar4) +
                              MAX_VERTS * sizeof(float4) +
                              MAX_PLANES * sizeof(uchar) +
                              MAX_PLANES * sizeof(float4) +
                              MAX_PLANES * sizeof(int));
    }

    __host__ __device__ void setup_from_buffer(void *buffer, int num_threads)
    {
        char *ptr = (char *)buffer;

        vertices = (uchar4 *)ptr;
        ptr += MAX_VERTS * num_threads * sizeof(uchar4);

        vertex_pos = (float4 *)ptr;
        ptr += MAX_VERTS * num_threads * sizeof(float4);

        next_boundaries = (uchar *)ptr;
        ptr += MAX_PLANES * num_threads * sizeof(uchar);

        planes = (float4 *)ptr;
        ptr += MAX_PLANES * num_threads * sizeof(float4);

        planes_neighbour = (int *)ptr;
    }
};

__device__ inline float dot3(float4 A, float4 B) { return A.x * B.x + A.y * B.y + A.z * B.z; }

__device__ inline float4 mul3(float s, float4 A) { return make_float4(s * A.x, s * A.y, s * A.z, 1.0f); }

__device__ inline float4 operator*(float s, const float4 &a) { return make_float4(s * a.x, s * a.y, s * a.z, s * a.w); }

__device__ inline float4 operator*(const float4 &a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

__device__ inline float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline float4 operator-(const float4 &a, const float4 &b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ inline void swapUChar3(uchar3 &a, uchar3 &b)
{
    uchar3 t = a;
    a = b;
    b = t;
}

enum Status
{
    success = 0,
    plane_overflow,
    vertex_overflow,
    empty_cell,
    security_radius_not_reached,
    inconsistent_boundary,
    max_status_num,
};

struct ConvexCell
{
    float4 voro_seed;
    Status *status;
    const float4 *pts;

    uchar4 *m_vertices;
    float4 *m_vertex_pos;
    uchar *m_next_boundaries;
    float4 *m_planes;
    int *m_planesNeighbour;

    int local_tid;

    uchar nb_vertex;
    uchar nb_plane;
    float max_radius;
    float3 upper_bound;
    float3 lower_bound;
    uint64_t plane_bitmask; // Bloom filter for fast CCHasPlane negative lookups

    __device__ void SetMemory(const ConvexCellMemory &mem, int block_idx, int local_thread_idx)
    {
        local_tid = local_thread_idx;
        int block_base_vert = block_idx * CELL_BLOCK_STRIDE * MAX_VERTS;
        int block_base_plane = block_idx * CELL_BLOCK_STRIDE * MAX_PLANES;

        m_vertices = mem.vertices + block_base_vert;
        m_vertex_pos = mem.vertex_pos + block_base_vert;
        m_next_boundaries = mem.next_boundaries + block_base_plane;
        m_planes = mem.planes + block_base_plane;
        m_planesNeighbour = mem.planes_neighbour + block_base_plane;
    }

    __device__ uchar &CCBoundaryNextRef(int p) { return m_next_boundaries[p * CELL_BLOCK_STRIDE + local_tid]; }
    __device__ uchar *CCBoundaryNext(int p) { return &m_next_boundaries[p * CELL_BLOCK_STRIDE + local_tid]; }

    __device__ int &CCPlaneNeighbourRef(int p) { return m_planesNeighbour[p * CELL_BLOCK_STRIDE + local_tid]; }
    __device__ int *CCPlaneNeighbour(int p) { return &m_planesNeighbour[p * CELL_BLOCK_STRIDE + local_tid]; }

    __device__ float4 CCPlane(int p) { return m_planes[p * CELL_BLOCK_STRIDE + local_tid]; }

    __device__ void CCSetPlane(int p, float x, float y, float z, float w)
    {
        m_planes[p * CELL_BLOCK_STRIDE + local_tid] = make_float4(x, y, z, w);
    }

    __device__ void CCSetPlaneR4(int p, float4 plane) { m_planes[p * CELL_BLOCK_STRIDE + local_tid] = plane; }

    __device__ uchar4 &CCVertexRef(int v) { return m_vertices[v * CELL_BLOCK_STRIDE + local_tid]; }
    __device__ uchar3 *CCVertex(int v) { return (uchar3 *)&m_vertices[v * CELL_BLOCK_STRIDE + local_tid]; }

    __device__ float4 &CCVertexPosRef(int v) { return m_vertex_pos[v * CELL_BLOCK_STRIDE + local_tid]; }
    __device__ float4 CCVertexPos(int v) { return m_vertex_pos[v * CELL_BLOCK_STRIDE + local_tid]; }
    __device__ void CCSetVertexPos(int v, float4 pos) { m_vertex_pos[v * CELL_BLOCK_STRIDE + local_tid] = pos; }

    __device__ uchar CCIthPlane(uchar t, int i) { return ((uchar *)CCVertex(t))[i]; }

    __device__ static inline int hash_vid(int vid)
    {
        return (int)(((unsigned int)vid * 2654435761u) >> 26) & 63;
    }

    __device__ bool CCHasPlane(int vid)
    {
        uint64_t bit = 1ULL << hash_vid(vid);
        if (!(plane_bitmask & bit))
            return false; // Definitely not present
        // Hash collision possible — verify with linear scan
        for (int i = 0; i < nb_plane; i++)
        {
            if (*CCPlaneNeighbour(i) == vid)
                return true;
        }
        return false;
    }

    __device__ void CCInit(int id, const float4 *p_pts, Status *p_status, float3 bbox_min, float3 bbox_max)
    {
        status = p_status;
        pts = p_pts;
        voro_seed = p_pts[id];
        max_radius = 1e20f;
        plane_bitmask = 0;

        float const eps = CUBE_EPSILON;
        float const vmin_x = bbox_min.x - eps;
        float const vmax_x = bbox_max.x + eps;
        float const vmin_y = bbox_min.y - eps;
        float const vmax_y = bbox_max.y + eps;
        float const vmin_z = bbox_min.z - eps;
        float const vmax_z = bbox_max.z + eps;

        CCSetPlane(0, 1.0, 0.0, 0.0, -vmin_x);
        CCSetPlane(1, -1.0, 0.0, 0.0, vmax_x);
        CCSetPlane(2, 0.0, 1.0, 0.0, -vmin_y);
        CCSetPlane(3, 0.0, -1.0, 0.0, vmax_y);
        CCSetPlane(4, 0.0, 0.0, 1.0, -vmin_z);
        CCSetPlane(5, 0.0, 0.0, -1.0, vmax_z);
        nb_plane = 6;

        *CCVertex(0) = make_uchar3(2, 5, 0);
        *CCVertex(1) = make_uchar3(5, 3, 0);
        *CCVertex(2) = make_uchar3(1, 5, 2);
        *CCVertex(3) = make_uchar3(5, 1, 3);
        *CCVertex(4) = make_uchar3(4, 2, 0);
        *CCVertex(5) = make_uchar3(4, 0, 3);
        *CCVertex(6) = make_uchar3(2, 4, 1);
        *CCVertex(7) = make_uchar3(4, 3, 1);

        for (int i = 0; i < 8; ++i)
        {
            CCSetVertexPos(i, CCComputeVertexCoordinates(*CCVertex(i), false));
        }

        nb_vertex = 8;

#pragma unroll
        for (int i = 0; i < MAX_PLANES; ++i)
            *CCPlaneNeighbour(i) = -1;

        lower_bound = make_float3(vmin_x - voro_seed.x, vmin_y - voro_seed.y, vmin_z - voro_seed.z);
        upper_bound = make_float3(vmax_x - voro_seed.x, vmax_y - voro_seed.y, vmax_z - voro_seed.z);

        float3 max_dist = make_float3(fmaxf(fabsf(lower_bound.x), fabsf(upper_bound.x)),
                                      fmaxf(fabsf(lower_bound.y), fabsf(upper_bound.y)),
                                      fmaxf(fabsf(lower_bound.z), fabsf(upper_bound.z)));
        max_radius = sqrtf(max_dist.x * max_dist.x + max_dist.y * max_dist.y + max_dist.z * max_dist.z);
    }

    __device__ float4 CCComputeVertexCoordinates(uchar3 t, bool persp_divide)
    {
        float4 const pi1 = CCPlane(t.x);
        float4 const pi2 = CCPlane(t.y);
        float4 const pi3 = CCPlane(t.z);
        float4 result;
        float const m12 = pi2.x * pi3.y - pi2.y * pi3.x;
        float const m13 = pi2.x * pi3.z - pi2.z * pi3.x;
        float const m14 = pi2.x * pi3.w - pi2.w * pi3.x;
        float const m23 = pi2.y * pi3.z - pi2.z * pi3.y;
        float const m24 = pi2.y * pi3.w - pi2.w * pi3.y;
        float const m34 = pi2.z * pi3.w - pi2.w * pi3.z;
        result.x = -pi1.w * m23 - pi1.y * m34 + pi1.z * m24;
        result.y = pi1.x * m34 + pi1.w * m13 - pi1.z * m14;
        result.z = -pi1.x * m24 + pi1.y * m14 - pi1.w * m12;
        result.w = pi1.x * m23 - pi1.y * m13 + pi1.z * m12;
        if (persp_divide)
        {
            float inv_w = 1.0f / result.w;
            return make_float4(result.x * inv_w, result.y * inv_w, result.z * inv_w, 1.0f);
        }
        return result;
    }

    __device__ bool CCVertexIsInHalfSpace(int v_idx, float4 eqn)
    {
        float4 v = CCVertexPos(v_idx);
        return dot3(v, eqn) + v.w * eqn.w > 0;
    }

    // Standard perpendicular bisector (no weight terms)
    __device__ float4 CCComputeBisectorPlane(float4 p)
    {
        float4 dir = voro_seed - p;
        float4 ave2 = voro_seed + p;
        float dot = dot3(ave2, dir);
        float dirNorm = sqrt(dot3(dir, dir));
        return make_float4(dir.x / dirNorm, dir.y / dirNorm, dir.z / dirNorm, -dot / 2 / dirNorm);
    }

    __device__ int CCAddPlane(int vid, float4 plane)
    {
        if (nb_plane >= MAX_PLANES)
        {
            *status = plane_overflow;
            return -1;
        }
        CCSetPlaneR4(nb_plane, plane);
        *CCPlaneNeighbour(nb_plane) = vid;
        if (vid >= 0)
            plane_bitmask |= 1ULL << hash_vid(vid);
        return nb_plane++;
    }

    __device__ void CCNewVertex(uchar i, uchar j, uchar k)
    {
        if (nb_vertex >= MAX_VERTS)
        {
            *status = vertex_overflow;
            return;
        }
        *CCVertex(nb_vertex) = make_uchar3(i, j, k);
        CCSetVertexPos(nb_vertex, CCComputeVertexCoordinates(*CCVertex(nb_vertex), false));
        nb_vertex++;
    }

    __device__ uchar CCComputeBoundary(uchar nb_removed)
    {
#pragma unroll
        for (int i = 0; i < MAX_PLANES; ++i)
            *CCBoundaryNext(i) = END_OF_LIST;
        uchar firstBoundary = END_OF_LIST;

        int nb_iter = 0;
        uchar t = nb_vertex;

        while (nb_removed > 0)
        {
            if (nb_iter++ > 1000)
            {
                *status = inconsistent_boundary;
                return firstBoundary;
            }
            bool is_in_border[3];
            bool next_is_opp[3];
            for (int e = 0; e < 3; ++e)
                is_in_border[e] = (*CCBoundaryNext(CCIthPlane(t, e)) != END_OF_LIST);
            for (int e = 0; e < 3; ++e)
                next_is_opp[e] = (*CCBoundaryNext(CCIthPlane(t, (e + 1) % 3)) == CCIthPlane(t, e));

            bool new_border_is_simple = true;
            for (int e = 0; e < 3; ++e)
                if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3])
                    new_border_is_simple = false;

            if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2])
            {
                if (firstBoundary == END_OF_LIST)
                {
                    for (int e = 0; e < 3; ++e)
                        *CCBoundaryNext(CCIthPlane(t, e)) = CCIthPlane(t, (e + 1) % 3);
                    firstBoundary = CCVertex(t)->x;
                }
                else
                    new_border_is_simple = false;
            }

            if (!new_border_is_simple)
            {
                t++;
                if (t == nb_vertex + nb_removed)
                    t = nb_vertex;
                continue;
            }

            for (int e = 0; e < 3; ++e)
                if (!next_is_opp[e])
                    *CCBoundaryNext(CCIthPlane(t, e)) = CCIthPlane(t, (e + 1) % 3);

            for (int e = 0; e < 3; ++e)
                if (next_is_opp[e] && next_is_opp[(e + 1) % 3])
                {
                    if (firstBoundary == CCIthPlane(t, (e + 1) % 3))
                        firstBoundary = *CCBoundaryNext(CCIthPlane(t, (e + 1) % 3));
                    *CCBoundaryNext(CCIthPlane(t, (e + 1) % 3)) = END_OF_LIST;
                }

            swapUChar3(*CCVertex(t), *CCVertex(nb_vertex + nb_removed - 1));
            t = nb_vertex;
            nb_removed--;
        }
        return firstBoundary;
    }

    // Simplified: pure squared distance check (no weight terms)
    __device__ inline bool CCIsPointTooFar(float4 p)
    {
        float dx = voro_seed.x - p.x;
        float dy = voro_seed.y - p.y;
        float dz = voro_seed.z - p.z;
        float sq_dist = dx * dx + dy * dy + dz * dz;
        return sq_dist > 4.0f * max_radius * max_radius;
    }

    __device__ bool CCClipByPlane(int vid, float4 p)
    {
        if (*status == plane_overflow)
            return false;

#ifdef MODERN_CUDA_CARDS
        if (CCIsPointTooFar(p) || CCHasPlane(vid))
            return false;
#else
        if (CCIsPointTooFar(p))
            return false;
#endif

        float4 eqn = CCComputeBisectorPlane(p);

        uchar nb_removed = 0;

#ifdef MODERN_CUDA_CARDS
        int current_nb_vertex = nb_vertex;
        bool intersects = false;
        for (int k = 0; k < current_nb_vertex; ++k)
        {
            if (CCVertexIsInHalfSpace(k, eqn))
            {
                intersects = true;
                break;
            }
        }

        if (!intersects)
            return false;
#endif

        int plane_idx = CCAddPlane(vid, eqn);
        if (plane_idx < 0)
            return false;

        int i = 0;
        while (i < nb_vertex)
        {
            if (CCVertexIsInHalfSpace(i, eqn))
            {
                nb_vertex--;
                swapUChar3(*CCVertex(i), *CCVertex(nb_vertex));
                float4 t = CCVertexPos(i);
                CCSetVertexPos(i, CCVertexPos(nb_vertex));
                CCSetVertexPos(nb_vertex, t);

                nb_removed++;
            }
            else
                i++;
        }

        if (nb_vertex < 1)
        {
            *status = empty_cell;
            nb_plane--;
            return false;
        }

#ifndef MODERN_CUDA_CARDS
        if (nb_removed == 0)
        {
            nb_plane--;
            return false;
        }
#endif

        uchar firstBoundary = CCComputeBoundary(nb_removed);
        if (*status != success)
            return false;
        if (firstBoundary == END_OF_LIST)
            return false;

        uchar cir = firstBoundary;
        bool changed = false;
        int iteration_limit = 1000;
        int iterations = 0;
        do
        {
            if (++iterations > iteration_limit)
            {
                return changed;
            }
            int newCir = *CCBoundaryNext(cir);
            CCNewVertex(plane_idx, cir, newCir);
            if (*status != success)
                return changed;
            cir = newCir;
            changed = true;
        } while (cir != firstBoundary);
        return changed;
    }

    __device__ void CCGarbageCollect()
    {
        for (int i = 0; i < nb_plane; ++i)
            *CCBoundaryNext(i) = END_OF_LIST;
        for (int i = 0; i < nb_vertex; ++i)
        {
            uchar3 pl = *CCVertex(i);
            *CCBoundaryNext(pl.x) = 0;
            *CCBoundaryNext(pl.y) = 0;
            *CCBoundaryNext(pl.z) = 0;
        }
        int n = 0;
        for (int i = 0; i < nb_plane; ++i)
        {
            uchar *c = CCBoundaryNext(i);
            if (*c != 0)
                continue;
            if (i != n)
            {
                CCSetPlaneR4(n, CCPlane(i));
                *CCPlaneNeighbour(n) = *CCPlaneNeighbour(i);
            }
            *c = n++;
        }
        for (int i = 0; i < nb_vertex; ++i)
        {
            uchar3 *pl = CCVertex(i);
            pl->x = *CCBoundaryNext(pl->x);
            pl->y = *CCBoundaryNext(pl->y);
            pl->z = *CCBoundaryNext(pl->z);
        }
        nb_plane = n;
        // Rebuild bitmask after plane compaction
        plane_bitmask = 0;
        for (int i = 0; i < nb_plane; ++i)
        {
            int vid = *CCPlaneNeighbour(i);
            if (vid >= 0)
                plane_bitmask |= 1ULL << hash_vid(vid);
        }
    }

    __device__ void CCUpdateRadius()
    {
        float max_r2 = 0.0f;
        for (int v = 0; v < nb_vertex; ++v)
        {
            float4 pos_h = CCVertexPos(v);
            float inv_w = 1.0f / pos_h.w;
            float4 pc = make_float4(pos_h.x * inv_w, pos_h.y * inv_w, pos_h.z * inv_w, 1.0f);
            float4 diff = pc - voro_seed;
            float d2 = dot3(diff, diff);
            if (d2 > max_r2)
                max_r2 = d2;
        }
        max_radius = sqrt(max_r2);
    }

    __device__ void CCUpdateBounds()
    {
        float3 upper = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        float3 lower = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float max_r2 = 0.0f;
        for (int v = 0; v < nb_vertex; ++v)
        {
            float4 pos_h = CCVertexPos(v);
            float inv_w = 1.0f / pos_h.w;
            float4 pc = make_float4(pos_h.x * inv_w, pos_h.y * inv_w, pos_h.z * inv_w, 1.0f);
            float4 diff = pc - voro_seed;
            float d2 = dot3(diff, diff);
            if (d2 > max_r2)
                max_r2 = d2;
            upper.x = fmaxf(upper.x, diff.x);
            upper.y = fmaxf(upper.y, diff.y);
            upper.z = fmaxf(upper.z, diff.z);
            lower.x = fminf(lower.x, diff.x);
            lower.y = fminf(lower.y, diff.y);
            lower.z = fminf(lower.z, diff.z);
        }
        upper_bound = upper;
        lower_bound = lower;
        max_radius = sqrtf(max_r2);
    }
};

} // namespace laguerre

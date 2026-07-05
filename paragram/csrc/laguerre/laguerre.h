#pragma once

#include <optional>
#include <torch/torch.h>
#include <tuple>

namespace laguerre
{

// Returns (adjacency, adjacency_offsets, status)
// status is per-point: 0=success, other values indicate errors
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_laguerre_voronoi_ultra(torch::Tensor points_in,
                             torch::Tensor weights_in,
                             int knn = 8,
                             bool debug = false,
                             std::optional<torch::Tensor> initial_guesses = std::nullopt,
                             std::optional<torch::Tensor> initial_guesses_offsets = std::nullopt,
                             int leaf_size = -1);

// Weightless Voronoi: no weight tensor needed, uses cuBQL BVH directly
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_voronoi_ultra(torch::Tensor points_in,
                    int knn = 8,
                    bool debug = false,
                    std::optional<torch::Tensor> initial_guesses = std::nullopt,
                    std::optional<torch::Tensor> initial_guesses_offsets = std::nullopt,
                    int leaf_size = -1);

} // namespace laguerre
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Diagram:
    adjacency: torch.Tensor
    offsets: torch.Tensor
    status: torch.Tensor


def _validate_points(points: torch.Tensor) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if not points.is_cuda:
        raise ValueError("points must be on a CUDA device")
    if points.dtype != torch.float32:
        raise ValueError("points must have dtype torch.float32")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points.contiguous()


def _validate_weights(weights: torch.Tensor, num_points: int, device: torch.device) -> torch.Tensor:
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if not weights.is_cuda:
        raise ValueError("weights must be on a CUDA device")
    if weights.device != device:
        raise ValueError("weights must be on the same device as points")
    if weights.dtype != torch.float32:
        raise ValueError("weights must have dtype torch.float32")
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or weights.shape != (num_points, 1):
        raise ValueError("weights must have shape (N,) or (N, 1)")
    return weights.contiguous()


def _validate_initial_guesses(
    initial_guesses: Optional[torch.Tensor],
    initial_guesses_offsets: Optional[torch.Tensor],
    num_points: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if initial_guesses is None and initial_guesses_offsets is None:
        return None, None
    if initial_guesses is None or initial_guesses_offsets is None:
        raise ValueError("initial_guesses and initial_guesses_offsets must be provided together")
    if not initial_guesses.is_cuda or not initial_guesses_offsets.is_cuda:
        raise ValueError("initial guess tensors must be on a CUDA device")
    if initial_guesses.device != device or initial_guesses_offsets.device != device:
        raise ValueError("initial guess tensors must be on the same device as points")
    if initial_guesses.dtype != torch.int32 or initial_guesses_offsets.dtype != torch.int32:
        raise ValueError("initial guess tensors must have dtype torch.int32")
    if initial_guesses_offsets.ndim != 1 or initial_guesses_offsets.shape[0] != num_points + 1:
        raise ValueError("initial_guesses_offsets must have shape (N + 1,)")
    return initial_guesses.contiguous(), initial_guesses_offsets.contiguous()


def power_diagram(
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    knn: int = 8,
    debug: bool = False,
    initial_guesses: Optional[torch.Tensor] = None,
    initial_guesses_offsets: Optional[torch.Tensor] = None,
    leaf_size: int = -1,
) -> Diagram:
    points = _validate_points(points)
    weights = _validate_weights(weights, points.shape[0], points.device)
    initial_guesses, initial_guesses_offsets = _validate_initial_guesses(
        initial_guesses,
        initial_guesses_offsets,
        points.shape[0],
        points.device,
    )
    from .cuda import build_laguerre_voronoi_ultra

    adjacency, offsets, status = build_laguerre_voronoi_ultra(
        points,
        weights,
        knn=knn,
        debug=debug,
        initial_guesses=initial_guesses,
        initial_guesses_offsets=initial_guesses_offsets,
        leaf_size=leaf_size,
    )
    return Diagram(adjacency=adjacency, offsets=offsets, status=status)


def voronoi_diagram(
    points: torch.Tensor,
    *,
    knn: int = 8,
    debug: bool = False,
    initial_guesses: Optional[torch.Tensor] = None,
    initial_guesses_offsets: Optional[torch.Tensor] = None,
    leaf_size: int = -1,
) -> Diagram:
    points = _validate_points(points)
    initial_guesses, initial_guesses_offsets = _validate_initial_guesses(
        initial_guesses,
        initial_guesses_offsets,
        points.shape[0],
        points.device,
    )
    from .cuda import build_voronoi_ultra

    adjacency, offsets, status = build_voronoi_ultra(
        points,
        knn=knn,
        debug=debug,
        initial_guesses=initial_guesses,
        initial_guesses_offsets=initial_guesses_offsets,
        leaf_size=leaf_size,
    )
    return Diagram(adjacency=adjacency, offsets=offsets, status=status)

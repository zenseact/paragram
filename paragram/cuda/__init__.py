from .._backend import _C

if _C is None:
    raise ImportError("Paragram CUDA extension is unavailable. A CUDA-enabled PyTorch install and nvcc are required.")

build_laguerre_voronoi_ultra = _C.build_laguerre_voronoi_ultra
build_voronoi_ultra = _C.build_voronoi_ultra

__all__ = [
    "build_laguerre_voronoi_ultra",
    "build_voronoi_ultra",
]

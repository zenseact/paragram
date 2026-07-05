<div align="center">

<h1>Paragram</h1>
<p><em>Scalable GPU Construction of 3D Voronoi and Power Diagrams - SIGGRAPH 2026</em></p>

</div>

<p align="center">
    <!-- project badges -->
    <a href="https://research.zenseact.com/publications/paragram/"><img src="https://img.shields.io/badge/Project-Page-ffa"/></a>
    <!-- paper badges -->
    <a href="https://arxiv.org/abs/2605.06408">
        <img src='https://img.shields.io/badge/arXiv-Paper-aff'>
    </a>
</p>

Paragram is a standalone PyTorch CUDA package for GPU construction of 3D Voronoi and power diagrams. It packages the algorithm used in the SIGGRAPH26 paper "Scalable GPU Construction of 3D Voronoi and Power Diagrams".

It is designed to be a drop-in replacement for the CPU-based CGAL and Geogram libraries, with a focus on speed and scalability for large point clouds. The library is implemented in C++ and CUDA, and provides a simple Python interface for easy integration into existing workflows. The CUDA compilations is performed lazily on first import, and the library automatically chunks convex-cell scratch memory when free VRAM is limited.

> [!NOTE]
> Lower-resource GPUs can run larger point clouds with a small runtime penalty when chunking is active. This was introduced after the paper allowing for much larger point clouds to be processed despite limited VRAM.

## Install

The CUDA extension is built lazily on first import (can take a minute the first time). It requires CUDA 12.x and NVCC. The instalations can be done via pip/uv or any other Python package manager.

```bash
pip install git+https://github.com/zenseact/paragram.git
```

You can also add it to your existing python project by adding it as a dependency in your `pyproject.toml`:

```
dependencies = [
    ...
    "paragram @ git+https://github.com/zenseact/paragram.git"
]
```

### Compile Options
Useful build environment variables:

- `TORCH_CUDA_ARCH_LIST`: CUDA architectures passed to PyTorch extension builds.
- `MAX_JOBS`: parallel compile jobs.
- `FAST_COMPILE=1`: compile with `-O0`.
- `DEBUG_CUDA=1`: compile with debug CUDA flags.
- `NO_FAST_MATH=1`: disable `-use_fast_math`.

## Usage

```python
import torch
import paragram

points = torch.rand(100_000, 3, device="cuda", dtype=torch.float32)
weights = torch.zeros(100_000, 1, device="cuda", dtype=torch.float32)

voronoi = paragram.voronoi_diagram(points)
power = paragram.power_diagram(points, weights)

print(voronoi.adjacency.shape, voronoi.offsets.shape, voronoi.status.shape)
```

Both APIs return a `Diagram` with:

- `adjacency`: flattened `torch.int32` CUDA adjacency list.
- `offsets`: `torch.int32` CUDA CSR offsets with shape `(N + 1,)`.
- `status`: `torch.int32` CUDA status per input point.

## Memory Controls

Paragram chunks convex-cell scratch memory automatically when free VRAM is limited. This low-memory path was added after the paper to let lower-resource GPUs run larger point clouds. It reduces peak scratch allocation with a small runtime penalty when chunking is active, so the paper timing tables should be compared with the default unchunked path.

Override automatic chunking with:

- `PARAGRAM_NUM_CHUNKS=<n>`: force exactly `n` chunks.
- `PARAGRAM_MEM_FRAC=<0..1>`: cap scratch allocation to this fraction of free VRAM.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{taveira2026paragram,
  author    = {Taveira, Bernardo and Lindstr{\"o}m, Carl and Fatemi, Maryam and Hammarstrand, Lars and Kahl, Fredrik},
  title     = {Scalable GPU Construction of 3D Voronoi and Power Diagrams},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3799902.3811229},
  doi       = {10.1145/3799902.3811229},
  booktitle = {ACM SIGGRAPH 2026 Conference Papers},
  series    = {SIGGRAPH '26}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

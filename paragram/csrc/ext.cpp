#include <torch/extension.h>

#include "laguerre/laguerre.h"
#include "utils/cuda_helpers.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    paragram::global_cuda_init();

    m.def("build_laguerre_voronoi_ultra",
          &laguerre::build_laguerre_voronoi_ultra,
          "Build weighted 3D power-diagram adjacency",
          py::arg("points"),
          py::arg("weights"),
          py::arg("knn") = 8,
          py::arg("debug") = false,
          py::arg("initial_guesses") = py::none(),
          py::arg("initial_guesses_offsets") = py::none(),
          py::arg("leaf_size") = -1);

    m.def("build_voronoi_ultra",
          &laguerre::build_voronoi_ultra,
          "Build unweighted 3D Voronoi adjacency",
          py::arg("points"),
          py::arg("knn") = 8,
          py::arg("debug") = false,
          py::arg("initial_guesses") = py::none(),
          py::arg("initial_guesses_offsets") = py::none(),
          py::arg("leaf_size") = -1);
}

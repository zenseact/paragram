import glob
import importlib
import json
import os
import time
from subprocess import DEVNULL, call

import torch
from packaging import version
from rich.console import Console
from torch.utils.cpp_extension import _find_cuda_home
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    _get_build_directory,
    _import_module_from_library,
    _jit_compile,
)

PATH = os.path.dirname(os.path.abspath(__file__))
VERBOSE = os.getenv("VERBOSE", "0") == "1"
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
FAST_COMPILE = os.getenv("FAST_COMPILE", "0") == "1"
DEBUG_CUDA = os.getenv("DEBUG_CUDA", "0") == "1"
PROFILE_CUDA = os.getenv("PROFILE_CUDA", "0") == "1"
USE_PRECOMPILED_HEADERS = os.getenv("USE_PRECOMPILED_HEADERS", "0") == "1"

MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"

if version.parse(torch.__version__) < version.parse("2.2") and USE_PRECOMPILED_HEADERS:
    Console().print(
        "[yellow]paragram: precompiled headers require torch >= 2.2; disabling them.[/yellow]"
    )
    USE_PRECOMPILED_HEADERS = False


def _modern_cuda_card() -> bool:
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 10


def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
):
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)

    try:
        if USE_PRECOMPILED_HEADERS:
            from torch.utils.cpp_extension import (
                _check_and_build_extension_h_precompiler_headers,
            )

            _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths)
            head_file = os.path.join(_TORCH_PATH, "include", "torch", "extension.h")
            extra_cflags += ["-include", head_file, "-Winvalid-pch"]

        try:
            return _jit_compile(
                name,
                sources,
                extra_cflags,
                extra_cuda_cflags,
                extra_ldflags,
                extra_include_paths,
                build_directory,
                verbose,
                with_cuda=None,
                is_python_module=True,
                is_standalone=False,
                keep_intermediates=True,
            )
        except TypeError as e:
            if "_jit_compile() missing" not in str(e):
                raise
            return _jit_compile(
                name,
                sources,
                extra_cflags,
                extra_cuda_cflags,
                None,
                extra_ldflags,
                extra_include_paths,
                build_directory,
                verbose,
                with_cuda=None,
                with_sycl=None,
                is_python_module=True,
                is_standalone=False,
                keep_intermediates=True,
            )
    except OSError:
        return _import_module_from_library(name, build_directory, True)


def cuda_toolkit_available():
    cuda_home = _find_cuda_home()
    if not cuda_home:
        return False

    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.isfile(nvcc_path):
        return True

    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    cuda_home = _find_cuda_home()
    if not cuda_home:
        return None

    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            return f.read().strip().split()[-1]
    if os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            return json.load(f)["cuda"]["version"]
    raise RuntimeError("Cannot find the CUDA version file in CUDA_HOME.")


_C = None

try:
    _C = importlib.import_module("paragram.csrc")
    if not hasattr(_C, "build_laguerre_voronoi_ultra"):
        _C = None
except ImportError:
    _C = None

if _C is None:
    if cuda_toolkit_available():
        name = "paragram_cuda"
        build_dir = _get_build_directory(name, verbose=False)
        cubql_path = os.path.join(PATH, "external", "cubql")

        opt_level = "-O0" if (FAST_COMPILE or DEBUG_CUDA) else "-O3"
        extra_cflags = [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level, "--extended-lambda"]
        if _modern_cuda_card():
            extra_cuda_cflags += ["-Xptxas", "-dlcm=ca", "--maxrregcount=128", "-DMODERN_CUDA_CARDS"]
        if not NO_FAST_MATH:
            extra_cuda_cflags += ["-use_fast_math"]
        if DEBUG_CUDA:
            extra_cuda_cflags += ["-g", "-G"]
            extra_cflags += ["-g"]
            Console().print("[yellow]paragram: CUDA debug build enabled.[/yellow]")
        if PROFILE_CUDA or DEBUG_CUDA:
            extra_cuda_cflags += ["-lineinfo"]

        sources = [
            source
            for source in sorted(glob.glob(os.path.join(PATH, "csrc", "laguerre", "*.cu")))
            if os.path.basename(source) != "utils.cu"
        ]
        sources.append(os.path.join(PATH, "csrc", "ext.cpp"))

        tic = time.time()
        with Console().status(
            f"[bold yellow]Paragram: building CUDA extension with MAX_JOBS={os.environ['MAX_JOBS']}",
            spinner="bouncingBall",
        ):
            _C = load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=["-lcuda"],
                extra_include_paths=[os.path.join(PATH, "csrc"), cubql_path],
                build_directory=build_dir,
                verbose=VERBOSE,
            )
        toc = time.time()
        Console().print(f"[green]paragram: CUDA extension ready in {toc - tic:.2f} seconds.[/green]")
    else:
        Console().print("[yellow]paragram: CUDA toolkit not found; CUDA extension is unavailable.[/yellow]")

if need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS")

__all__ = ["_C"]

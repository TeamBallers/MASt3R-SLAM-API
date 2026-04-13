from pathlib import Path
from setuptools import setup
import re
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()


def _nvcc_version_tuple():
    if not CUDA_HOME:
        return (0, 0)

    nvcc_path = os.path.join(CUDA_HOME, "bin", "nvcc")
    if not os.path.exists(nvcc_path):
        return (0, 0)

    try:
        output = subprocess.check_output([nvcc_path, "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return (0, 0)

    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if not match:
        return (0, 0)

    return (int(match.group(1)), int(match.group(2)))

include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")
    nvcc_version = _nvcc_version_tuple()
    enable_sm120 = os.environ.get("MAST3R_ENABLE_SM120", "0") == "1"
    nvcc_arch_flags = [
        "-O3",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
    ]
    # Native sm_120 compilation is opt-in because many environments mix a newer
    # PyTorch build with an older system nvcc, which hard-fails on compute_120.
    if enable_sm120 and nvcc_version >= (12, 8):
        nvcc_arch_flags.append("-gencode=arch=compute_120,code=sm_120")
    elif enable_sm120:
        print(
            "MAST3R_ENABLE_SM120=1 was requested, but the NVCC compiler from "
            f"CUDA_HOME={CUDA_HOME!r} is older than 12.8. Building without sm_120 and "
            "keeping compute_90 PTX fallback. To compile native sm_120 kernels, point "
            "CUDA_HOME at a CUDA 12.8+ toolkit before installing."
        )
    extra_compile_args["nvcc"] = nvcc_arch_flags
    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

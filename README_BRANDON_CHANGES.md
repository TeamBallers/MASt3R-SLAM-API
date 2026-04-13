# MASt3R-SLAM-API: Agent Change Log

This document summarizes all code changes made during troubleshooting and stabilization of the install/runtime pipeline.

## 1) Build and CUDA Toolchain Compatibility

### setup.py
- Added CUDA arch handling logic to avoid unconditional `compute_120` compilation failures on NVCC 12.4.
- Added NVCC version detection using `CUDA_HOME` so architecture flags depend on the actual compiler being used.
- Made native `sm_120` compilation opt-in via environment variable (`MAST3R_ENABLE_SM120=1`) instead of automatic.
- Kept `compute_90` PTX fallback for compatibility when native `sm_120` is unavailable.

### pyproject.toml
- Removed direct git dependency on `lietorch @ git+https://github.com/princeton-vl/lietorch.git`.
- Reason: this was repeatedly forcing remote lietorch rebuilds in environments with mixed CUDA toolchains.

### thirdparty/lietorch-src/setup.py
- Added Eigen include fallback:
	- Primary: `thirdparty/lietorch-src/eigen`
	- Fallback: `thirdparty/eigen` (repo-level Eigen)
- Reason: local lietorch submodule Eigen directory was empty in this workspace, causing `Eigen/Dense` include failures.

### mast3r_slam/backend/src/matching_kernels.cu
- Replaced deprecated dispatch usage:
	- `AT_DISPATCH_FLOATING_TYPES_AND_HALF(D11.type(), ...)`
	- -> `AT_DISPATCH_FLOATING_TYPES_AND_HALF(D11.scalar_type(), ...)`
- Reason: PyTorch 2.11 C++ API compatibility.

### mast3r_slam/backend/src/gn_kernels.cu
- Replaced `torch::linalg::linalg_norm(...)` with `dx.norm()` in 3 optimization loops.
- Reason: newer libtorch C++ API compatibility and compile stability.


## 2) Runtime / PyTorch Checkpoint Loading Compatibility

### thirdparty/mast3r/mast3r/model.py
- Changed checkpoint loading to:
	- `torch.load(model_path, map_location='cpu', weights_only=False)`
- Reason: PyTorch 2.6+ default changed `weights_only=True`, which broke loading checkpoints containing `argparse.Namespace`.

### thirdparty/mast3r/mast3r/retrieval/processor.py
- Changed retrieval checkpoint loading to:
	- `torch.load(modelname, 'cpu', weights_only=False)`
- Reason: same PyTorch default behavior change as above.

### thirdparty/mast3r/mast3r/retrieval/model.py
- Changed optional pretrained retrieval loading to:
	- `torch.load(pretrained_retrieval, 'cpu', weights_only=False)`
- Reason: same PyTorch default behavior change as above.


## 3) WSL / Multiprocessing CUDA Stability

### main.py
- Added runtime CUDA probe helper (`resolve_runtime_device`) that attempts actual CUDA tensor allocation and provides actionable error text.
- Added WSL detection and automatic single-process fallback:
	- disables visualization process on WSL
	- enables single-thread config mode
	- runs backend optimization inline (`run_backend_step`) instead of a spawned CUDA-sharing worker
- Added safe teardown guards:
	- only `join()` backend/viz processes when those handles are not `None`.
- Reason: avoid CUDA IPC `invalid resource handle` issues in WSL spawn + shared CUDA tensor flows.

### mast3r_slam/frame.py
- Added optional shared-memory behavior for state/keyframe tensors:
	- new helper `_maybe_share(...)`
	- new `shared` parameter for `SharedStates` and `SharedKeyframes`
- In single-process mode, tensors are no longer forced into shared CUDA IPC storage.
- Reason: remove unnecessary/fragile CUDA IPC usage on WSL.


## 4) Pipeline Robustness and Input Hygiene

### process_images.py
- Added `check_runtime_dependencies()` preflight:
	- validates `torch` and `lietorch` imports before conversion/SLAM launch
	- prints targeted recovery steps when `lietorch` is broken
- Added output cleanup and consistency checks:
	- removes stale image files from output directory before conversion
	- validates all output PNGs share one resolution before launching SLAM
- Reason: prevent mixed-resolution datasets and early-fail with actionable diagnostics.


## 5) Optional Runtime Dependency Behavior

### mast3r_slam/dataloader.py
- Made `pyrealsense2` import optional at module load time.
- Added explicit error message only when `RealsenseDataset` is actually used.
- Reason: allow non-Realsense workflows to run even if RealSense dependencies are not installed.

### thirdparty/mast3r/dust3r/dust3r/utils/image.py
- Removed hard dependency on `torchvision.transforms` for image normalization.
- Replaced with lightweight local normalization class (`_ImgNorm`) using NumPy + PyTorch.
- Reason: reduce import/runtime friction in constrained environments.

### thirdparty/mast3r/dust3r/croco/models/curope/setup.py
- Added Windows-only NVCC flag handling (`-allow-unsupported-compiler`) and centralized `nvcc_flags` variable usage.
- Reason: improve cross-platform build behavior for this third-party extension.


## 6) Documentation Updates

### README.md
- Added notes on:
	- newer GPU architecture support requirements
	- mixed-toolchain behavior (`torch` vs system `nvcc`)
	- local lietorch install flow (`thirdparty/lietorch-src`)
	- CUDA arch environment guidance (`TORCH_CUDA_ARCH_LIST`, optional native `sm_120` path)


## Notes
- The warning `Warning, cannot find cuda-compiled version of RoPE2D` is informational (fallback path), not a hard failure.
- Keyframe output folder stores selected keyframes only, not all input frames.


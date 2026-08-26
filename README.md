# AOTInductor External Weights Example

Demonstrates exporting a PyTorch model with `torch.export` + AOTInductor,
loading and updating its weights from an externally-dumped state dict, and
running inference both in Python and in C++. The example targets **CUDA**.

## Files

- [export_aoti_artifacts.py](export_aoti_artifacts.py) — Compiles the AOTI package and dumps it plus the initial/updated weights and sample I/O tensors to disk under `aoti_package/`, for consumption by the C++ example. Also runs the same Python-side verification in-process before handing the artifacts off to C++.
- [aoti_cpp_inference.cpp](aoti_cpp_inference.cpp) — Loads the AOTI package and the dumped weights/tensors in C++, runs inference, and verifies the outputs against the ones computed by PyTorch.
- [CMakeLists.txt](CMakeLists.txt) — Builds `aoti_cpp_inference` against libtorch.

## Requirements

- A CUDA-capable GPU and a CUDA-enabled PyTorch build.
- CMake and a C++17 compiler.

## Running via Docker

All of the above requirements are already available in NVIDIA's PyTorch
container, which is the easiest way to run this example:

```bash
docker run --rm -it --gpus all -v $(pwd):/mnt -w /mnt nvcr.io/nvidia/pytorch:26.07-py3
```

## Usage

1. Generate the AOTI package and dump the weights:

   ```bash
   python export_aoti_artifacts.py
   ```

   This writes to `aoti_package/`:
   - `model.pt2` — the AOTInductor-compiled package.
   - `weights_initial/data.pt` — the initial `state_dict`.
   - `weights_updated/data.pt` — the `state_dict` after an in-place weight update.
   - `io_data/data.pt` — the sample input `x` and the expected outputs for both weight sets.

2. Build the C++ example:

   ```bash
   cmake -S . -B build -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
   cmake --build build -j
   ```

3. Run it:

   ```bash
   ./build/aoti_cpp_inference
   ```

   Expected output:

   ```
   Initial AOTI inference succeeded and matches PyTorch output.
   Weight update verified successfully! Updated AOTI output matches PyTorch.
   ```

## Notes

- `torch::pickle_load` (used on the C++ side) hardcodes the archive's internal
  root folder name to `"data"`. This only happens when the file saved by
  `torch.save` is itself named `data.<ext>`, which is why each weight/tensor
  dict is saved as `<subdir>/data.pt` instead of a single flat file.
- `always_keep_tensor_constants: True` is required in `inductor_configs`;
  without it, small constants like the linear layer's bias get folded into
  the graph at compile time instead of being exposed as updatable constants,
  causing `RuntimeError: Constant not found: fc.bias` when calling
  `load_constants`.
- `load_constants(..., allow_h2d_copy=True)` lets a CPU `state_dict` be loaded
  into a CUDA-resident compiled model; here it's used mainly for convenience
  since the weights are already on CUDA.

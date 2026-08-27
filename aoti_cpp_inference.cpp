// AOTI C++ inference example.
//
// This program loads an AOTInductor-compiled model package together with
// externally-dumped weights (produced by export_aoti_artifacts.py) and runs
// inference, demonstrating how to update the model's constants at runtime.
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

// torch::pickle_load() expects the zip archive's internal root folder to be
// named "data", which is only the case when the file was saved by Python as
// "data.<ext>" (see save_state_dict() in export_aoti_artifacts.py).
std::unordered_map<std::string, at::Tensor>
load_tensor_dict(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    TORCH_CHECK(file.is_open(), "Failed to open file: ", path);
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    torch::IValue ivalue = torch::pickle_load(data);
    std::unordered_map<std::string, at::Tensor> result;
    for (const auto& item : ivalue.toGenericDict())
    {
        result.emplace(item.key().toStringRef(), item.value().toTensor());
    }
    return result;
}

} // namespace

int main(int argc, char* argv[])
{
    const std::string artifacts_dir = argc > 1 ? argv[1] : "aoti_package";
    const std::string package_path = artifacts_dir + "/model.pt2";

    torch::inductor::AOTIModelPackageLoader loader(package_path,
                                                   /*model_name=*/"model",
                                                   /*run_single_threaded=*/false,
                                                   /*num_runners=*/1);
    // swap_constant_buffer()/free_inactive_constant_buffer() are only
    // exposed on the runner, not directly on AOTIModelPackageLoader.
    torch::inductor::AOTIModelContainerRunner* runner = loader.get_runner();

    // Run inference on an explicit stream (instead of the default one) and
    // hand its handle to run().
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard stream_guard(stream);
    void* stream_handle = stream.stream();

    auto io_data = load_tensor_dict(artifacts_dir + "/io_data/data.pt");
    const at::Tensor& x = io_data.at("x");
    const at::Tensor& orig_output = io_data.at("orig_output");
    const at::Tensor& updated_orig_output = io_data.at("updated_orig_output");

    // 1. First-time load: nothing is being served yet, so there's no active
    // buffer worth preserving -- load the weights straight into the active
    // buffer instead of going through the inactive-buffer/swap dance.
    //
    // load_constants() (not the lower-level update_constant_buffer()) is
    // used because it translates the FQN keys (e.g. "fc.weight") to the
    // engine's internal mangled constant names; update_constant_buffer()
    // expects those internal names directly.
    //
    // The dumped weights are CPU tensors, so allow_h2d_copy=true lets the
    // engine copy them straight from host memory into its GPU-resident
    // buffer. If we instead loaded them as GPU tensors here, the engine
    // would still make its own copy into its buffer, leaving 2 GPU-resident
    // copies alive at once (ours + the engine's) instead of 1.
    auto initial_weights =
        load_tensor_dict(artifacts_dir + "/weights_initial/data.pt");
    loader.load_constants(initial_weights,
                          /*use_inactive=*/false,
                          /*check_full_update=*/true,
                          /*user_managed=*/false,
                          /*allow_h2d_copy=*/true);
    // The H2D copy above is asynchronous, and with multiple runners/streams
    // potentially consuming the shared constant buffer, a device-wide sync
    // (not just our own stream) is needed to guarantee it has completed
    // before any of them run inference against it.
    torch::cuda::synchronize();

    auto initial_outputs = loader.run({x}, stream_handle);
    // run() executed on our stream asynchronously; synchronize it before
    // reading the results on the CPU below.
    stream.synchronize();
    TORCH_CHECK(
        torch::allclose(initial_outputs[0], orig_output, /*rtol=*/1e-5,
                        /*atol=*/1e-6),
        "Initial AOTI output does not match the expected PyTorch output.");
    std::cout << "Initial AOTI inference succeeded and matches PyTorch output."
              << std::endl;

    // 2. Hot weight update: the model could already be serving inference at
    // this point, so load into the inactive buffer, atomically swap it in,
    // and free the now-inactive (old) buffer instead of overwriting the
    // active buffer in place.
    auto updated_weights =
        load_tensor_dict(artifacts_dir + "/weights_updated/data.pt");
    loader.load_constants(updated_weights,
                          /*use_inactive=*/true,
                          /*check_full_update=*/true,
                          /*user_managed=*/false,
                          /*allow_h2d_copy=*/true);
    runner->swap_constant_buffer();
    runner->free_inactive_constant_buffer();
    // Same reasoning as above: synchronize the whole device, not just our
    // stream, before running inference against the swapped-in buffer.
    torch::cuda::synchronize();

    auto updated_outputs = loader.run({x}, stream_handle);
    // Same reasoning as above: wait for our stream before reading results.
    stream.synchronize();
    TORCH_CHECK(
        torch::allclose(updated_outputs[0], updated_orig_output, /*rtol=*/1e-5,
                        /*atol=*/1e-6),
        "Updated AOTI output does not match the expected PyTorch output.");
    std::cout << "Weight update verified successfully! Updated AOTI output "
                 "matches PyTorch."
              << std::endl;

    return 0;
}

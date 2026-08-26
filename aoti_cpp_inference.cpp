// AOTI C++ inference example.
//
// This program loads an AOTInductor-compiled model package together with
// externally-dumped weights (produced by export_aoti_artifacts.py) and runs
// inference, demonstrating how to update the model's constants at runtime.
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
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

    torch::inductor::AOTIModelPackageLoader loader(package_path);

    auto io_data = load_tensor_dict(artifacts_dir + "/io_data/data.pt");
    const at::Tensor& x = io_data.at("x");
    const at::Tensor& orig_output = io_data.at("orig_output");
    const at::Tensor& updated_orig_output = io_data.at("updated_orig_output");

    // 1. Load the initial weights and verify the AOTI inference output.
    auto initial_weights =
        load_tensor_dict(artifacts_dir + "/weights_initial/data.pt");
    loader.load_constants(initial_weights,
                          /*use_inactive=*/false,
                          /*check_full_update=*/true,
                          /*user_managed=*/false,
                          /*allow_h2d_copy=*/true);

    auto initial_outputs = loader.run({x});
    TORCH_CHECK(
        torch::allclose(initial_outputs[0], orig_output, /*rtol=*/1e-5,
                        /*atol=*/1e-6),
        "Initial AOTI output does not match the expected PyTorch output.");
    std::cout << "Initial AOTI inference succeeded and matches PyTorch output."
              << std::endl;

    // 2. Load the updated weights and re-verify the AOTI inference output.
    auto updated_weights =
        load_tensor_dict(artifacts_dir + "/weights_updated/data.pt");
    loader.load_constants(updated_weights,
                          /*use_inactive=*/false,
                          /*check_full_update=*/true,
                          /*user_managed=*/false,
                          /*allow_h2d_copy=*/true);

    auto updated_outputs = loader.run({x});
    TORCH_CHECK(
        torch::allclose(updated_outputs[0], updated_orig_output, /*rtol=*/1e-5,
                        /*atol=*/1e-6),
        "Updated AOTI output does not match the expected PyTorch output.");
    std::cout << "Weight update verified successfully! Updated AOTI output "
                 "matches PyTorch."
              << std::endl;

    return 0;
}

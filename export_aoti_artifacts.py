import os

import torch
import torch._inductor

DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA is required to run this script."


# Define a PyTorch Module
class SampleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "aoti_package")


# torch::pickle_load in C++ hardcodes the archive's internal folder name to
# "data", which only happens when the saved file itself is named "data.<ext>".
def save_state_dict(state_dict, subdir):
    directory = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(directory, exist_ok=True)
    torch.save(state_dict, os.path.join(directory, "data.pt"))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = SampleModel().to(DEVICE).eval()
    x = torch.randn(2, 4, device=DEVICE)

    with torch.no_grad():
        orig_output = model(x)

    # Export the model and compile it ahead-of-time with AOTInductor.
    exported_program = torch.export.export(model, (x, ))
    package_path = torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path=os.path.join(OUTPUT_DIR, "model.pt2"),
        inductor_configs={
            "aot_inductor.package_constants_in_so": True,
            "freezing": False,
            # Without this, small constants like the bias get folded into
            # the graph and are not exposed as updatable constants.
            "always_keep_tensor_constants": True,
        },
    )

    # Dump the initial weights so the C++ side can load them via load_constants.
    weights_dict = {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }
    save_state_dict(weights_dict, "weights_initial")

    # Verify the compiled AOTI package matches PyTorch before handing it off
    # to the C++ side.
    compiled_model = torch._inductor.aoti_load_package(package_path)
    print(f"AOTI engine constant FQNs: {compiled_model.get_constant_fqns()}")
    compiled_model.load_constants(weights_dict,
                                  check_full_update=True,
                                  allow_h2d_copy=True)
    aoti_output = compiled_model(x)
    assert torch.allclose(orig_output, aoti_output, rtol=1e-5, atol=1e-6)
    print("✅ Initial AOTI inference succeeded and matches PyTorch output.")

    # Update the weights dynamically and dump them too, to demonstrate
    # updating the constants of an already-loaded AOTI model from C++.
    with torch.no_grad():
        model.fc.weight.add_(1.0)
        model.fc.bias.add_(0.5)
        updated_orig_output = model(x)

    updated_weights_dict = {
        **dict(model.named_parameters()),
        **dict(model.named_buffers()),
    }
    save_state_dict(updated_weights_dict, "weights_updated")

    # Verify the updated constants also match PyTorch.
    compiled_model.load_constants(updated_weights_dict,
                                  check_full_update=True,
                                  allow_h2d_copy=True)
    updated_aoti_output = compiled_model(x)
    assert torch.allclose(updated_orig_output,
                          updated_aoti_output,
                          rtol=1e-5,
                          atol=1e-6)
    print(
        "✅ Weight update verified successfully! Updated AOTI output matches PyTorch."
    )

    # Dump the sample input and expected outputs so the C++ side can verify
    # its AOTI inference results without needing PyTorch's eager model.
    save_state_dict(
        {
            "x": x,
            "orig_output": orig_output,
            "updated_orig_output": updated_orig_output
        },
        "io_data",
    )

    print(f"AOTI package and weights saved under: {OUTPUT_DIR}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'model.pt2')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'weights_initial', 'data.pt')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'weights_updated', 'data.pt')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'io_data', 'data.pt')}")


if __name__ == "__main__":
    main()

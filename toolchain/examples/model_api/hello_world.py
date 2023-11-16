import argparse
import torch
from turnkeyml import benchmark_model

torch.manual_seed(0)


# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 1000
output_size = 500
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Benchmark a PyTorch model on a specified device."
    )

    # Add the arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["x86", "nvidia"],
        default="x86",
        help="The device to benchmark on (x86 or nvidia)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Instantiate model and generate inputs
    torch.manual_seed(0)
    input_size = 1000
    output_size = 500
    pytorch_model = SmallModel(input_size, output_size)
    inputs = {"x": torch.rand(input_size)}

    # Benchmark the model on the specified device
    print(f"Benchmarking on {args.device}...")
    benchmark_model(
        pytorch_model,
        inputs,
        build_name="hello_api_world",
        device=args.device,
    )


if __name__ == "__main__":
    main()

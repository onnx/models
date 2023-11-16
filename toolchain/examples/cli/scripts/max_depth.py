# labels: name::max_depth author::turnkey test_group::b
"""
This model script contains a model, pytorch_model, which has
two sub-modules, fc and fc2. You can use it to experiment with
the --max-depth option, which analyzes/builds/benchmarks sub-modules
of any modules in the top-level script.

You can try it with:

turnkey max_depth.py --max-depth 1

You should see data for pytorch_model, fc, and fc2.

Meanwhile, if you were to run with the command:

turnkey max_depth.py --max-depth 0

Then you will only see data for pytorch_model.

"""

import torch

torch.manual_seed(0)

# Define model class
class TwoLayerModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayerModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        output = self.fc2(output)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5
pytorch_model = TwoLayerModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Print results
print(f"pytorch_outputs: {pytorch_outputs}")

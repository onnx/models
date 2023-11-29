# labels: name::hello_world author::turnkey test_group::a
"""
Hello, world! This is an example script that can be used to 
demonstrate the file-benchmarking API, `benchmark_files()`.

To see this in action, write a Python script that calls:

    benchmark_files(input_files=[path_to_this_file])

You should see the analysis phase pick up the SmallModel instance
and then benchmark it.
"""

import torch

torch.manual_seed(1)


# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 9
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Print results
print(f"pytorch_outputs: {pytorch_outputs}")

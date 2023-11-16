# labels: name::hello_world author::turnkey test_group::a
"""
Hello, world! This is the most basic turnkey cli example.
To try it out, run the following command:

turnkey hello_world.py

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
input_size = 11
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Print results
print(f"pytorch_outputs: {pytorch_outputs}")

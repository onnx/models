# labels: name::twolayer author::turnkey test_group::b
"""
This example demonstrates what happens when your script contains
two models. In this case, pytorch_model (of class SmallModel), and
another_pytorch_model (also of class SmallModel).

To try it, run:

turnkey two_models.py

You should see date printed to the screen for both pytorch_model and
another_pytorch_model.
"""

import torch

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
input_size = 10
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Instantiate model and generate inputs
another_input_size = 50
another_output_size = 10
another_pytorch_model = SmallModel(another_input_size, another_output_size)
more_inputs = {"x": torch.rand(another_input_size)}

more_pytorch_outputs = another_pytorch_model(**more_inputs)

# Print results
print(f"pytorch_outputs: {pytorch_outputs}")
print(f"more_pytorch_outputs: {more_pytorch_outputs}")

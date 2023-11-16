# labels: name::multiple_invocations
"""
This example demonstrates what happens when your script contains
a model that is invoked multiple times with different input shapes

To try it, run:

turnkey multiple_invocations.py

You should see the two unique invocations being identified.
"""
import torch

torch.manual_seed(1)

# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_features, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_size)

    def forward(self, x):
        # x has shape (batch_size, input_features)
        # Set the batch size dimension to -1 to allow for flexibility
        x = x.view(-1, x.size(1))

        output = self.fc(x)

        # Reshape the output to restore the original batch size dimension
        output = output.view(-1, output_size)
        return output


# Instantiate model and generate inputs
input_features = 11
output_size = 5
pytorch_model = SmallModel(input_features, output_size)

# Create 3 sets of inputs
batch_size = 1
inputs1 = {"x": torch.rand(batch_size, input_features)}
inputs2 = {"x": torch.rand(batch_size, input_features)}
inputs3 = {"x": torch.rand(batch_size + 1, input_features)}

pytorch_model(**inputs1)
pytorch_model(**inputs2)
pytorch_model(**inputs3)

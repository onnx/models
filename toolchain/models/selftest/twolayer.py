# labels: name::twolayer author::selftest
import torch

torch.manual_seed(0)


class TwoLayerTestModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayerTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        output = self.fc2(output)
        return output


# Instantiate model and generate inputs
input_features = 10
output_features = 5

# Model and input configurations
model = TwoLayerTestModel(input_features, output_features)
inputs = {"x": torch.rand(input_features)}

# Call model
model(**inputs)

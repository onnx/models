# labels: name::linear author::selftest test_group::selftest
import torch

torch.manual_seed(0)


class LinearTestModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        output = self.fc(x)
        return output


input_features = 10
output_features = 10

# Model and input configurations
model = LinearTestModel(input_features, output_features)
inputs = {"x": torch.rand(input_features)}


# Call model
model(**inputs)

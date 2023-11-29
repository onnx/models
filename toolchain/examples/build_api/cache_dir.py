"""
    This example demonstrates how to set the location of the build cache
    directory, using the cache_dir argument. The default value for
    cache_dir is `~/.cache/turnkey`.

    To specify a different cache directory than the default set cache_dir to
    your location of choice.

    Note 1: To change the cache directory for every build, a global default can be
    set with the `TURNKEY_CACHE_DIR` environment variable:
    export TURNKEY_CACHE_DIR=/path_of_your_choosing

    Note 2: Setting the cache_dir argument within build_model() will override the
    `TURNKEY_CACHE_DIR' setting.
"""

import torch
from turnkeyml import build_model

torch.manual_seed(0)


# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate PyTorch model and generate inputs
input_size = 10
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

# Build pytorch_model and set the cache_dir
# We also set the build_name to make the build easy to identify
my_local_cache = "local_cache"
build_model(
    pytorch_model, inputs, cache_dir=my_local_cache, build_name="my_cache_dir_build"
)

print(
    f"\nCheck out the cache created in the local directory by running 'ls {my_local_cache}'"
)

print("Example cache_dir.py finished")

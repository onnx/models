
import os
import shutil
from typing import Dict
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.build as build


# We generate a corpus on to the filesystem during the test
# to get around how weird bake tests are when it comes to
# filesystem access
test_scripts_dot_py = {
    "linear.py": """# labels: name::linear author::turnkey license::mit test_group::a task::test
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

output = model(**inputs)

""",
    "linear2.py": """# labels: name::linear2 author::turnkey license::mit test_group::b task::test
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
print(f"Pytorch_outputs: {pytorch_outputs}")
""",
    "crash.py": """# labels: name::crash author::turnkey license::mit task::test
import torch
import sys

torch.manual_seed(0)

# The purpose of this script is to intentionally crash
# so that we can test --resume
# Any test that doesn't supply the crash signal will treat this
# as a normal input script that runs a small model
if len(sys.argv) > 1:
    if sys.argv[1] == "crash!":
        assert False

class LinearTestModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        output = self.fc(x)
        return output


input_features = 5
output_features = 5

# Model and input configurations
model = LinearTestModel(input_features, output_features)
inputs = {"x": torch.rand(input_features)}

output = model(**inputs)
""",
}


def create_test_dir(key:str, test_scripts: Dict = None):
    # Define paths to be used
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "generated", f"{key}_cache_dir")
    corpus_dir = os.path.join(base_dir, "generated", f"test_corpus")
    
    # Delete folders if they exist and 
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    if os.path.isdir(corpus_dir):
        shutil.rmtree(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)

    # Populate corpus dir
    if test_scripts is None:
        test_scripts = test_scripts_dot_py
    for key, value in test_scripts.items():
        model_path = os.path.join(corpus_dir, key)
        with open(model_path, "w", encoding="utf") as f:
            f.write(value)

    return cache_dir, corpus_dir

def strip_dot_py(test_script_file: str) -> str:
    return test_script_file.split(".")[0]

def get_stats_and_state(
    test_script: str,
    cache_dir: str,
) -> int:
    # Figure out the build name by surveying the build cache
    builds = filesystem.get_all(cache_dir)
    test_script_name = strip_dot_py(test_script)

    for build_state_file in builds:
        if test_script_name in build_state_file:
            build_state = build.load_state(state_path=build_state_file)
            stats = filesystem.Stats(
                build_state.cache_dir,
                build_state.config.build_name,
                build_state.stats_id,
            )
            return stats.build_stats, build_state

    raise Exception(f"Stats not found for {test_script}")
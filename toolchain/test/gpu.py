"""
GPU tests
"""

import os
import unittest
from unittest.mock import patch
import sys
import shutil
from turnkeyml.cli.cli import main as turnkeycli
import turnkeyml.common.filesystem as filesystem
from cli import assert_success_of_builds, flatten

test_scripts_dot_py = {
    "linear.py": """# labels: name::linear author::turnkey license::mit test_group::a
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

"""
}


# Create a test directory and make it the CWD
test_dir = os.path.join(os.path.dirname(__file__), "generated", "gpu_test_dir")
cache_dir = os.path.join(os.path.dirname(__file__), "generated", "cache-dir")
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)
os.makedirs(test_dir)
os.chdir(test_dir)

corpus_dir = os.path.join(os.getcwd(), "test_corpus")
extras_dir = os.path.join(corpus_dir, "extras")
os.makedirs(extras_dir, exist_ok=True)

for key, value in test_scripts_dot_py.items():
    model_path = os.path.join(corpus_dir, key)

    with open(model_path, "w", encoding="utf") as f:
        f.write(value)


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        filesystem.rmdir(cache_dir)

        return super().setUp()

    def test_basic(self):
        test_script = "linear.py"
        # Benchmark with Pytorch
        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--device",
            "nvidia",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        assert_success_of_builds(
            [test_script], cache_dir, check_perf=True, runtime="trt"
        )


if __name__ == "__main__":
    unittest.main()

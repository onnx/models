"""
Tests focused on the analysis capabilities of turnkey CLI
"""

import os
import unittest
from pathlib import Path
import shutil
import glob
import subprocess
import numpy as np
from contextlib import redirect_stdout
from unittest.mock import patch
import io
import sys
import platform
from turnkeyml.cli.cli import main as turnkeycli
import turnkeyml.common.labels as labels
from turnkeyml.parser import parse
import turnkeyml.common.filesystem as filesystem
from helpers import common

try:
    # pylint: disable=unused-import
    import transformers
    import timm
except ImportError as e:
    raise ImportError(
        "The Huggingface transformers and timm libraries are required for running this test. "
        "Install them with `pip install transformers timm`"
    )


# We generate a corpus on to the filesystem during the test
# to get around how weird bake tests are when it comes to
# filesystem access

test_scripts_dot_py = {
    "linear_pytorch.py": """
# labels: test_group::selftest license::mit framework::pytorch tags::selftest,small
import torch
import argparse

torch.manual_seed(0)

# Receive command line arg
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--my-arg",
)
args = parser.parse_args()
print(f"Received arg {args.my_arg}")

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
unexecuted_model = LinearTestModel(input_features+1, output_features)
inputs = {"x": torch.rand(input_features)}
output = model(**inputs)

""",
    "pipeline.py": """
import os
from transformers import (
    TextClassificationPipeline,
    BertForSequenceClassification,
    BertConfig,
    PreTrainedTokenizerFast,
)

tokenizer_file = os.path.join(os.path.dirname(__file__),"tokenizer.json")
class MyPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        configuration = BertConfig()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        super().__init__(
            model=BertForSequenceClassification(configuration), tokenizer=tokenizer
        )


my_pipeline = MyPipeline()
my_pipeline("This restaurant is awesome")
""",
    "activation.py": """
import torch
m = torch.nn.GELU()
input = torch.randn(2)
output = m(input)
""",
    "turnkey_parser.py": """
from turnkeyml.parser import parse

parsed_args = parse(["height", "width", "num_channels"])

print(parsed_args)

""",
    "two_executions.py": """
import torch
import timm
from turnkeyml.parser import parse

# Creating model and set it to evaluation mode
model = timm.create_model("mobilenetv2_035", pretrained=False)
model.eval()

# Creating inputs
inputs1 = torch.rand((1, 3, 28, 28))
inputs2 = torch.rand((1, 3, 224, 224))

# Calling model
model(inputs1)
model(inputs2)
model(inputs1)
""",
}
minimal_tokenizer = """
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[UNK]": 0
    }
  }
}"""

# Create a test directory
cache_dir, corpus_dir = common.create_test_dir("analysis", test_scripts_dot_py)

with open(os.path.join(corpus_dir, "tokenizer.json"), "w", encoding="utf") as f:
    f.write(minimal_tokenizer)


def cache_is_lean(cache_dir, build_name):
    files = list(glob.glob(f"{cache_dir}/{build_name}/**/*", recursive=True))
    is_lean = len([x for x in files if ".onnx" in x]) == 0
    metadata_found = len([x for x in files if ".txt" in x]) > 0
    return is_lean and metadata_found


def run_cli(args):
    with redirect_stdout(io.StringIO()) as f:
        with patch.object(sys, "argv", args):
            turnkeycli()

            return f.getvalue()


def run_analysis(args):
    output = run_cli(args)

    # Process outputs
    output = output[output.rfind("Models discovered") :]
    models_executed = output.count("(executed")
    models_built = output.count("Model successfully built!")
    return models_executed, 0, models_built


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        filesystem.rmdir(cache_dir)
        return super().setUp()

    def test_01_basic(self):
        pytorch_output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "linear_pytorch.py"),
                "--analyze-only",
            ]
        )
        assert np.array_equal(pytorch_output, (1, 0, 0))

    def test_03_depth(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "linear_pytorch.py"),
                "--max-depth",
                "1",
                "--analyze-only",
            ]
        )
        assert np.array_equal(output, (2, 0, 0))

    def test_04_build(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "linear_pytorch.py::76af2f62"),
                "--max-depth",
                "1",
                "--build-only",
                "--cache-dir",
                cache_dir,
            ]
        )
        assert np.array_equal(output, (2, 0, 1))

    def test_05_cache(self):
        model_hash = "76af2f62"
        run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, f"linear_pytorch.py::{model_hash}"),
                "--max-depth",
                "1",
                "--cache-dir",
                cache_dir,
                "--lean-cache",
                "--build-only",
            ]
        )
        build_name = f"linear_pytorch_{model_hash}"
        labels_found = labels.load_from_cache(cache_dir, build_name) != {}
        assert cache_is_lean(cache_dir, build_name) and labels_found

    def test_06_generic_args(self):
        output = run_cli(
            [
                "turnkey",
                os.path.join(corpus_dir, "linear_pytorch.py"),
                "--max-depth",
                "1",
                "--script-args",
                "--my-arg test_arg",
                "--analyze-only",
            ]
        )
        assert "Received arg test_arg" in output

    # TODO: Investigate why this test is only failing on Windows
    @unittest.skipIf(
        platform.system() == "Windows",
        "Potential turnkeyml windows bug"
        "The ouputs do match, but fails due to misinterpretation",
    )
    def test_07_valid_turnkey_args(self):
        height, width, num_channels = parse(["height", "width", "num_channels"])
        cmd = [
            "turnkey",
            os.path.join(corpus_dir, "turnkey_parser.py"),
            "--script-args",
            f"--num_channels {num_channels+1}",
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        output = stdout.decode("utf-8")
        expected_output = str([height, width, num_channels + 1])
        assert expected_output in output, f"Got {output} but expected {expected_output}"

    def test_08_invalid_turnkey_args(self):
        cmd = [
            "turnkey",
            os.path.join(corpus_dir, "turnkey_parser.py"),
            "--script-args",
            "--invalid_arg 123",
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        assert "error: unrecognized argument" in stderr.decode("utf-8")

    def test_09_pipeline(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "pipeline.py"),
                "--analyze-only",
            ]
        )
        assert np.array_equal(output, (1, 0, 0))

    def test_10_activation(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "activation.py"),
                "--analyze-only",
            ]
        )
        assert np.array_equal(output, (0, 0, 0))

    def test_11_analyze_only(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "linear_pytorch.py"),
                "--analyze-only",
            ]
        )
        assert np.array_equal(output, (1, 0, 0))

    def test_12_turnkey_hashes(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "linear_pytorch.py::76af2f62"),
                "--build-only",
                "--max-depth",
                "1",
                "--cache-dir",
                cache_dir,
            ]
        )
        assert np.array_equal(output, (2, 0, 1))

    def test_13_clean_cache(self):
        model_hash = "76af2f62"
        run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, f"linear_pytorch.py::{model_hash}"),
                "--max-depth",
                "1",
                "--cache-dir",
                cache_dir,
                "--build-only",
            ]
        )
        build_name = f"linear_pytorch_{model_hash}"

        cmd = [
            "turnkey",
            "cache",
            "clean",
            build_name,
            "--cache-dir",
            cache_dir,
        ]
        subprocess.run(cmd, check=True)

        assert cache_is_lean(cache_dir, build_name)

    def test_14_same_model_different_input_shapes(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "two_executions.py"),
                "--analyze-only",
            ]
        )
        assert np.array_equal(output, (2, 0, 0))

    def test_15_same_model_different_input_shapes_maxdepth(self):
        output = run_analysis(
            [
                "turnkey",
                os.path.join(corpus_dir, "two_executions.py"),
                "--analyze-only",
                "--max-depth",
                "1",
            ]
        )
        assert np.array_equal(output, (6, 0, 0))


if __name__ == "__main__":
    unittest.main()

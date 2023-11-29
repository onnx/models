"""
Tests focused on the command-level functionality of turnkey CLI
"""

import os
import glob
import csv
from typing import List, Tuple, Any, Union, Optional
import unittest
from unittest.mock import patch
import sys
import io
from pathlib import Path
from contextlib import redirect_stdout
import yaml
import onnx
import platform
import torch
from turnkeyml.cli.cli import main as turnkeycli
import turnkeyml.cli.report as report
import turnkeyml.common.filesystem as filesystem
from turnkeyml.run.onnxrt.runtime import OnnxRT
from turnkeyml.run.tensorrt.runtime import TensorRT
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.exceptions as exceptions
import turnkeyml.build.export as export
from turnkeyml.cli.parser_helpers import decode_args, encode_args
from helpers import common

# Create a cache directory a directory with test models
cache_dir, corpus_dir = common.create_test_dir("cli")

extras_dot_py = {
    "compiled.py": """
# labels: name::linear author::selftest test_group::selftest task::test
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

# Compiled model
model = LinearTestModel(input_features, output_features)
model = torch.compile(model)
inputs = {"x": torch.rand(input_features)}
model(**inputs)

# Non-compiled model
model2 = LinearTestModel(input_features * 2, output_features)
inputs2 = {"x": torch.rand(input_features * 2)}
model2(**inputs2)
""",
    "selected_models.txt": f"""
{os.path.join(corpus_dir,"linear.py")}
{os.path.join(corpus_dir,"linear2.py")}
""",
    "timeout.py": """# labels: name::timeout author::turnkey license::mit test_group::a task::test
import torch

torch.manual_seed(0)


class LinearTestModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        output = self.fc(x)
        return output


input_features = 500000
output_features = 1000

# Model and input configurations
model = LinearTestModel(input_features, output_features)
inputs = {"x": torch.rand(input_features)}

output = model(**inputs)

""",
}

extras_dir = os.path.join(corpus_dir, "extras")
os.makedirs(extras_dir, exist_ok=True)

for key, value in extras_dot_py.items():
    file_path = os.path.join(extras_dir, key)

    with open(file_path, "w", encoding="utf") as f:
        f.write(value)


def bash(cmd: str) -> List[str]:
    """
    Emulate behavior of bash terminal when listing files
    """
    return glob.glob(cmd)


def flatten(lst: List[Union[str, List[str]]]) -> List[str]:
    """
    Flatten List[Union[str, List[str]]] into a List[str]
    """
    flattened = []
    for element in lst:
        if isinstance(element, list):
            flattened.extend(element)
        else:
            flattened.append(element)
    return flattened


def assert_success_of_builds(
    test_script_files: List[str],
    cache_dir: str,
    info_property: Tuple[str, Any] = None,
    check_perf: bool = False,
    check_opset: Optional[int] = None,
    check_iteration_count: Optional[int] = None,
    runtime: str = "ort",
    check_onnx_file_count: Optional[int] = None,
) -> int:
    # Figure out the build name by surveying the build cache
    # for a build that includes test_script_name in the name
    # TODO: simplify this code when
    # https://github.com/aig-bench/onnxmodelzoo/issues/16
    # is done
    builds = filesystem.get_all(cache_dir)
    builds_found = 0

    for test_script in test_script_files:
        test_script_name = common.strip_dot_py(test_script)
        script_build_found = False

        for build_state_file in builds:
            if test_script_name in build_state_file:
                build_state = build.load_state(state_path=build_state_file)
                stats = filesystem.Stats(
                    build_state.cache_dir,
                    build_state.config.build_name,
                    build_state.stats_id,
                )
                assert build_state.build_status == build.Status.SUCCESSFUL_BUILD
                script_build_found = True
                builds_found += 1

                if info_property is not None:
                    assert (
                        build_state.info.__dict__[info_property[0]] == info_property[1]
                    ), f"{build_state.info.__dict__[info_property[0]]} == {info_property[1]}"

                if check_perf:
                    assert stats.build_stats["mean_latency"] > 0
                    assert stats.build_stats["throughput"] > 0

                if check_iteration_count:
                    iterations = stats.build_stats["iterations"]
                    assert iterations == check_iteration_count

                if check_opset:
                    onnx_model = onnx.load(build_state.results[0])
                    model_opset = getattr(onnx_model.opset_import[0], "version", None)
                    assert model_opset == check_opset

                if check_onnx_file_count:
                    onnx_dir = export.onnx_dir(build_state)
                    assert len(os.listdir(onnx_dir)) == check_onnx_file_count

        assert script_build_found

    # Returns the total number of builds found
    return builds_found


class SmallPytorchModel(torch.nn.Module):
    def __init__(self):
        super(SmallPytorchModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        output = self.fc(x)
        return output


# Define pytorch model and inputs
pytorch_model = SmallPytorchModel()
inputs = {"x": torch.rand(10)}
inputs_2 = {"x": torch.rand(5)}
input_tensor = torch.rand(10)


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        filesystem.rmdir(cache_dir)

        return super().setUp()

    def test_001_cli_single(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir)

    def test_002_search_multiple(self):
        # Test the first model in the corpus
        test_scripts = list(common.test_scripts_dot_py.keys())

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_scripts[0]),
            os.path.join(corpus_dir, test_scripts[1]),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_scripts[0], test_scripts[1]], cache_dir)

    def test_003_cli_build_dir(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        test_scripts = common.test_scripts_dot_py.keys()

        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        assert_success_of_builds(test_scripts, cache_dir)

    def test_021_cli_report(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        test_scripts = common.test_scripts_dot_py.keys()

        # Build the test corpus so we have builds to report
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        testargs = [
            "turnkey",
            "cache",
            "report",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Read generated CSV file
        summary_csv_path = report.get_report_name()
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Check if csv file contains all expected rows and columns
        expected_cols = [
            "model_name",
            "author",
            "class",
            "parameters",
            "hash",
            "runtime",
            "device_type",
            "device",
            "mean_latency",
            "throughput",
            "all_build_stages",
            "completed_build_stages",
        ]
        linear_summary = summary[1]
        assert len(summary) == len(test_scripts)
        assert all(
            elem in linear_summary for elem in expected_cols
        ), f"Looked for each of {expected_cols} in {linear_summary.keys()}"

        # Check whether all rows we expect to be populated are actually populated
        assert (
            linear_summary["model_name"] == "linear2"
        ), f"Wrong model name found {linear_summary['model_name']}"
        assert (
            linear_summary["author"] == "turnkey"
        ), f"Wrong author name found {linear_summary['author']}"
        assert (
            linear_summary["class"] == "TwoLayerModel"
        ), f"Wrong class found {linear_summary['model_class']}"
        assert (
            linear_summary["hash"] == "80b93950"
        ), f"Wrong hash found {linear_summary['hash']}"
        assert (
            linear_summary["runtime"] == "ort"
        ), f"Wrong runtime found {linear_summary['runtime']}"
        assert (
            linear_summary["device_type"] == "x86"
        ), f"Wrong device type found {linear_summary['device_type']}"
        assert (
            float(linear_summary["mean_latency"]) > 0
        ), f"latency must be >0, got {linear_summary['x86_latency']}"
        assert (
            float(linear_summary["throughput"]) > 100
        ), f"throughput must be >100, got {linear_summary['throughput']}"

        # Make sure the report.get_dict() API works
        result_dict = report.get_dict(
            summary_csv_path, ["all_build_stages", "completed_build_stages"]
        )
        for result in result_dict.values():
            # All of the models should have exported to ONNX, so the "onnx_exported" value
            # should be True for all of them
            assert "export_pytorch" in yaml.safe_load(result["all_build_stages"])
            assert (
                "export_pytorch"
                in yaml.safe_load(result["completed_build_stages"]).keys()
            )
            assert (
                yaml.safe_load(result["completed_build_stages"])["export_pytorch"] > 0
            )

    def test_005_cli_list(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to list
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        for test_script in common.test_scripts_dot_py.keys():
            script_name = common.strip_dot_py(test_script)
            assert script_name in f.getvalue()

    def test_006_cli_delete(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus
        # - test_cli_list

        # Build the test corpus so we have builds to delete
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        for test_script in common.test_scripts_dot_py.keys():
            script_name = common.strip_dot_py(test_script)
            assert script_name in f.getvalue()

        # Delete the builds
        testargs = [
            "turnkey",
            "cache",
            "delete",
            "--all",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Make sure the builds are gone
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        for test_script in common.test_scripts_dot_py.keys():
            script_name = common.strip_dot_py(test_script)
            assert script_name not in f.getvalue()

    def test_007_cli_stats(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to print
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Make sure we can print the builds in the cache
        for test_script in common.test_scripts_dot_py.keys():
            test_script_path = os.path.join(corpus_dir, test_script)
            builds, script_name = filesystem.get_builds_from_file(
                cache_dir, test_script_path
            )

            for build_name in builds:
                # Make sure each build can be accessed with `turnkey cache stats`
                with redirect_stdout(io.StringIO()) as f:
                    testargs = [
                        "turnkey",
                        "cache",
                        "stats",
                        build_name,
                        "--cache-dir",
                        cache_dir,
                    ]
                    with patch.object(sys, "argv", testargs):
                        turnkeycli()

                    assert script_name in f.getvalue()

                # Make sure the stats YAML file contains the fields
                # required for producing a report
                stats_file = os.path.join(
                    build.output_dir(cache_dir, build_name), "turnkey_stats.yaml"
                )
                with open(stats_file, "r", encoding="utf8") as stream:
                    stats_dict = yaml.load(stream, Loader=yaml.FullLoader)

                assert isinstance(stats_dict["hash"], str), stats_dict["hash"]
                assert isinstance(stats_dict["parameters"], int), stats_dict[
                    "parameters"
                ]
                assert isinstance(
                    stats_dict["onnx_input_dimensions"], dict
                ), stats_dict["onnx_input_dimensions"]
                assert isinstance(
                    stats_dict["onnx_model_information"], dict
                ), stats_dict["onnx_model_information"]
                assert isinstance(stats_dict["onnx_ops_counter"], dict), stats_dict[
                    "onnx_ops_counter"
                ]
                assert isinstance(stats_dict["system_info"], dict), stats_dict[
                    "system_info"
                ]

                # Make sure the turnkey_stats has the expected ONNX opset
                assert (
                    stats_dict["onnx_model_information"]["opset"]
                    == build.DEFAULT_ONNX_OPSET
                ), stats_dict["onnx_model_information"]["opset"]

                # Make sure the turnkey_stats has the necessary fields used in the onnx model zoo
                assert isinstance(stats_dict["author"], str), stats_dict["author"]
                assert isinstance(stats_dict["model_name"], str), stats_dict[
                    "model_name"
                ]
                assert isinstance(stats_dict["task"], str), stats_dict["task"]

    def test_008_cli_version(self):
        # Get the version number
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "turnkey",
                "version",
            ]
            with patch.object(sys, "argv", testargs):
                turnkeycli()

        # Make sure we get back a 3-digit number
        assert len(f.getvalue().split(".")) == 3

    def test_009_cli_turnkey_args(self):
        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        # Set as many turnkey args as possible
        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--rebuild",
            "always",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir)

    # TODO: Investigate why this test is failing only on Windows CI failing
    @unittest.skipIf(platform.system() == "Windows", "Windows CI only failure")
    def test_011_cli_benchmark(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir, None, check_perf=True)

    # TODO: Investigate why this test is non-deterministically failing
    @unittest.skip("Flaky test")
    def test_013_cli_labels(self):
        # Only build models labels with test_group::a
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--labels",
            "test_group::a",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        state_files = [Path(p).stem for p in filesystem.get_all(cache_dir)]
        assert state_files == ["linear_d5b1df11_state"]

        # Delete the builds
        testargs = [
            "turnkey",
            "cache",
            "delete",
            "--all",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert filesystem.get_all(cache_dir) == []

        # Only build models labels with test_group::a and test_group::b
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/*.py"),
            "--labels",
            "test_group::a,b",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        state_files = [Path(p).stem for p in filesystem.get_all(cache_dir)]
        assert state_files == ["linear_d5b1df11_state", "linear2_80b93950_state"]

    @unittest.skip("Needs re-implementation")
    def test_014_report_on_failed_build(self):
        testargs = [
            "turnkey",
            bash(f"{corpus_dir}/linear.py"),
            "--device",
            "reimplement_me",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Ensure test failed
        build_state = build.load_state(state_path=filesystem.get_all(cache_dir)[0])
        assert build_state.build_status != build.Status.SUCCESSFUL_BUILD

        # Generate report
        testargs = [
            "turnkey",
            "cache",
            "report",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Read generated CSV file
        summary_csv_path = report.get_report_name()
        summary = None
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Ensure parameters and hash have been saved despite crash
        assert (
            len(summary) == 1
        ), "Report must contain only one row, but got {len(summary)}"
        assert (
            summary[0]["params"] == "110"
        ), "Wrong number of parameters found in report"
        assert summary[0]["hash"] == "d5b1df11", "Wrong hash found in report"

    def test_015_runtimes(self):
        # Attempt to benchmark using an invalid runtime
        with self.assertRaises(exceptions.ArgError):
            testargs = [
                "turnkey",
                "benchmark",
                bash(f"{corpus_dir}/linear.py"),
                "--cache-dir",
                cache_dir,
                "--device",
                "x86",
                "--runtime",
                "trt",
            ]
            with patch.object(sys, "argv", flatten(testargs)):
                turnkeycli()

        # Benchmark with Pytorch
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/linear.py"),
            "--cache-dir",
            cache_dir,
            "--device",
            "x86",
            "--runtime",
            "torch-eager",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        # Benchmark with Onnx Runtime
        testargs = [
            "turnkey",
            "benchmark",
            bash(f"{corpus_dir}/linear.py"),
            "--cache-dir",
            cache_dir,
            "--device",
            "x86",
            "--runtime",
            "ort",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

    # TODO: Investigate why this test is only failing on Windows CI
    @unittest.skipIf(platform.system() == "Windows", "Windows CI only failure")
    def test_016_cli_onnx_opset(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        user_opset = 15
        assert user_opset != build.DEFAULT_ONNX_OPSET

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--onnx-opset",
            str(user_opset),
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds(
            [test_script], cache_dir, None, check_perf=True, check_opset=user_opset
        )

    def test_016_cli_iteration_count(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        test_iterations = 123
        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--iterations",
            str(test_iterations),
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds(
            [test_script],
            cache_dir,
            None,
            check_perf=True,
            check_iteration_count=test_iterations,
        )

    def test_017_cli_process_isolation(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
            "--process-isolation",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir, None, check_perf=True)

    @unittest.skipIf(
        platform.system() == "Windows",
        "Skipping, as torch.compile is not supported on Windows"
        "Revisit when torch.compile for Windows is supported",
    )
    def test_018_skip_compiled(self):
        test_script = "compiled.py"
        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(extras_dir, test_script),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        builds_found = assert_success_of_builds([test_script], cache_dir)

        # Compile.py contains two Pytorch models.
        # One of those is compiled and should be skipped.
        assert builds_found == 1

    def test_019_invalid_file_type(self):
        # Ensure that we get an error when running turnkey with invalid input_files
        with self.assertRaises(exceptions.ArgError):
            testargs = ["turnkey", "gobbledegook"]
            with patch.object(sys, "argv", flatten(testargs)):
                turnkeycli()

    def test_020_cli_export_only(self):
        # Test the first model in the corpus
        test_script = list(common.test_scripts_dot_py.keys())[0]

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--sequence",
            "onnx-fp32",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([test_script], cache_dir, check_onnx_file_count=1)

    def test_022_cli_onnx_model(self):
        """
        Manually export an ONNX file, then feed it into the CLI
        """
        build_name = "receive_onnx"
        onnx_file = os.path.join(corpus_dir, f"{build_name}.onnx")

        # Create ONNX file
        torch.onnx.export(
            pytorch_model,
            input_tensor,
            onnx_file,
            opset_version=build.DEFAULT_ONNX_OPSET,
            input_names=["input"],
            output_names=["output"],
        )

        testargs = [
            "turnkey",
            "benchmark",
            onnx_file,
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([build_name], cache_dir)

    def test_023_cli_onnx_model_opset(self):
        """
        Manually export an ONNX file with a non-defualt opset, then feed it into the CLI
        """
        build_name = "receive_onnx_opset"
        onnx_file = os.path.join(corpus_dir, f"{build_name}.onnx")
        user_opset = build.MINIMUM_ONNX_OPSET

        # Make sure we are using an non-default ONNX opset
        assert user_opset != build.DEFAULT_ONNX_OPSET

        # Create ONNX file
        torch.onnx.export(
            pytorch_model,
            input_tensor,
            onnx_file,
            opset_version=user_opset,
            input_names=["input"],
            output_names=["output"],
        )

        testargs = [
            "turnkey",
            "benchmark",
            onnx_file,
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        assert_success_of_builds([build_name], cache_dir)

    def test_024_args_encode_decode(self):
        """
        Test the encoding and decoding of arguments that follow the
        ["arg1::[value1,value2]","arg2::value1","flag_arg"]' format
        """
        encoded_value = ["arg1::[value1,value2]", "arg2::value1", "flag_arg"]
        decoded_value = decode_args(encoded_value)
        reencoded_value = encode_args(decoded_value)
        assert (
            reencoded_value == encoded_value
        ), f"input: {encoded_value}, decoded: {decoded_value}, reencoded_value: {reencoded_value}"

    def test_025_benchmark_non_existent_file(self):
        # Ensure we get an error when benchmarking a non existent file
        with self.assertRaises(exceptions.ArgError):
            filename = "thou_shall_not_exist.py"
            with redirect_stdout(io.StringIO()) as f:
                testargs = ["turnkey", "benchmark", filename]
                with patch.object(sys, "argv", testargs):
                    turnkeycli()

    def test_026_benchmark_non_existent_file_prefix(self):
        # Ensure we get an error when benchmarking a non existent file
        with self.assertRaises(exceptions.ArgError):
            file_prefix = "non_existent_prefix_*.py"
            with redirect_stdout(io.StringIO()) as f:
                testargs = ["turnkey", "benchmark", file_prefix]
                with patch.object(sys, "argv", testargs):
                    turnkeycli()

    def test_027_input_text_file(self):
        """
        Ensure that we can intake .txt files
        """

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(extras_dir, "selected_models.txt"),
            "--cache-dir",
            cache_dir,
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        builds_found = assert_success_of_builds(["linear.py", "linear2.py"], cache_dir)
        assert (
            builds_found == 3
        ), f"Expected 3 builds (1 for linear.py, 2 for linear2.py), but got {builds_found}."

    def test_028_cli_timeout(self):
        """
        Make sure that the --timeout option and its associated reporting features work.

        timeout.py is designed to take a long time to export, which gives us the
        oportunity to kill it with a timeout.

        NOTE: this test can become flakey if:
         - exporting timeout.py takes less time than the timeout
         - the timeout kills the process before it has a chance to create a stats.yaml file
        """

        testargs = [
            "turnkey",
            "benchmark",
            os.path.join(extras_dir, "timeout.py"),
            "--cache-dir",
            cache_dir,
            "--process-isolation",
            "--timeout",
            "10",
        ]
        with patch.object(sys, "argv", flatten(testargs)):
            turnkeycli()

        testargs = [
            "turnkey",
            "cache",
            "report",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            turnkeycli()

        # Read generated CSV file and make sure the build was killed by the timeout
        summary_csv_path = report.get_report_name()
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary = list(csv.DictReader(summary_csv))

        # Check the summary for "killed", but also accept the edge case that
        # the build timed out before the stats.yaml was created
        try:
            timeout_summary = summary[0]

            assert timeout_summary["benchmark_status"] == "killed", timeout_summary[
                "benchmark_status"
            ]
        except IndexError:
            # Edge case where the CSV is empty because the build timed out before
            # the stats.yaml was created, which in turn means the CSV is empty
            pass


if __name__ == "__main__":
    unittest.main()

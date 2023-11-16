import os
import unittest
import torch
import shutil
import onnx
import platform
import turnkeyml.build.stage as stage
import turnkeyml.common.filesystem as filesystem
import turnkeyml.build.export as export
import turnkeyml.common.build as build
from turnkeyml import benchmark_model
from helpers import common


class SmallPytorchModel(torch.nn.Module):
    def __init__(self):
        super(SmallPytorchModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        output = self.fc(x)
        return output


class AnotherSimplePytorchModel(torch.nn.Module):
    def __init__(self):
        super(AnotherSimplePytorchModel, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(x)
        return output


# Define pytorch model and inputs
pytorch_model = SmallPytorchModel()
tiny_pytorch_model = AnotherSimplePytorchModel()
inputs = {"x": torch.rand(10)}
inputs_2 = {"x": torch.rand(5)}
input_tensor = torch.rand(10)

# Create a test directory
cache_dir, _ = common.create_test_dir("cli")


def get_build_state(cache_dir, build_name):
    return build.load_state(cache_dir=cache_dir, build_name=build_name)


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        filesystem.rmdir(cache_dir)
        return super().setUp()

    def test_001_build_pytorch_model(self):
        build_name = "build_pytorch_model"
        benchmark_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            build_only=True,
            cache_dir=cache_dir,
            runtime="ort",
        )
        state = get_build_state(cache_dir, build_name)
        assert state.build_status == build.Status.SUCCESSFUL_BUILD

    def test_002_custom_stage(self):
        build_name = "custom_stage"

        class MyCustomStage(stage.Stage):
            def __init__(self, funny_saying):
                super().__init__(
                    unique_name="funny_stage",
                    monitor_message="Funny Stage",
                )

                self.funny_saying = funny_saying

            def fire(self, state):
                print(f"funny message: {self.funny_saying}")
                state.build_status = build.Status.SUCCESSFUL_BUILD
                return state

        my_custom_stage = MyCustomStage(
            funny_saying="Is a fail whale a fail at all if it makes you smile?"
        )
        my_sequence = stage.Sequence(
            unique_name="my_sequence",
            monitor_message="Running My Sequence",
            stages=[
                export.ExportPytorchModel(),
                export.OptimizeOnnxModel(),
                my_custom_stage,
            ],
        )

        benchmark_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            sequence=my_sequence,
            build_only=True,
            cache_dir=cache_dir,
            runtime="ort",
        )

        state = get_build_state(cache_dir, build_name)
        return state.build_status == build.Status.SUCCESSFUL_BUILD

    # TODO: Investigate why this test is only failing on Windows CI failing
    @unittest.skipIf(platform.system() == "Windows", "Windows CI only failure")
    def test_003_local_benchmark(self):
        build_name = "local_benchmark"
        perf = benchmark_model(
            pytorch_model,
            inputs,
            device="x86",
            build_name=build_name,
            rebuild="always",
            cache_dir=cache_dir,
            lean_cache=True,
            runtime="ort",
        )
        state = get_build_state(cache_dir, build_name)
        assert state.build_status == build.Status.SUCCESSFUL_BUILD
        assert os.path.isfile(
            os.path.join(cache_dir, build_name, "x86_benchmark/outputs.json")
        )
        assert perf.mean_latency > 0
        assert perf.throughput > 0

    # TODO: Investigate why this test is only failing on Windows CI failing
    @unittest.skipIf(platform.system() == "Windows", "Windows CI only issue")
    def test_004_onnx_opset(self):
        """
        Make sure we can successfully benchmark a model with a user-defined ONNX opset
        """

        build_name = "onnx_opset"

        user_opset = 15
        assert user_opset != build.DEFAULT_ONNX_OPSET

        perf = benchmark_model(
            pytorch_model,
            inputs,
            device="x86",
            build_name=build_name,
            rebuild="always",
            cache_dir=cache_dir,
            onnx_opset=user_opset,
            runtime="ort",
        )
        state = get_build_state(cache_dir, build_name)
        assert state.build_status == build.Status.SUCCESSFUL_BUILD
        assert os.path.isfile(
            os.path.join(cache_dir, build_name, "x86_benchmark/outputs.json")
        )
        assert perf.mean_latency > 0
        assert perf.throughput > 0

        onnx_model = onnx.load(state.results[0])
        model_opset = getattr(onnx_model.opset_import[0], "version", None)
        assert user_opset == model_opset


if __name__ == "__main__":
    unittest.main()

import os
import unittest
import torch
import onnx
import tensorflow as tf
import numpy as np
import sklearn.ensemble
import sklearn.neighbors
import xgboost  # pylint: disable=import-error
import lightgbm  # pylint: disable=import-error
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import save_model
from onnxmltools.utils import load_model
from turnkeyml import build_model
import turnkeyml.build.export as export
import turnkeyml.build.stage as stage
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
import turnkeyml.build.sequences as sequences


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


class SmallKerasModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self):
        super(SmallKerasModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, x):  # pylint: disable=arguments-differ
        return self.dense(x)


base_dir = os.path.dirname(os.path.abspath(__file__))
cache_location = os.path.join(base_dir, "generated", "build_model_cache")

# Define pytorch model and inputs
pytorch_model = SmallPytorchModel()
tiny_pytorch_model = AnotherSimplePytorchModel()
inputs = {"x": torch.rand(10)}
inputs_2 = {"x": torch.rand(5)}
input_tensor = torch.rand(10)

# Define keras models and inputs
batch_keras_inputs = {"x": tf.random.uniform((1, 10), dtype=tf.float32)}
keras_subclass_model = SmallKerasModel()
keras_subclass_model.build(input_shape=(1, 10))
keras_sequential_model = tf.keras.Sequential()
keras_sequential_model.add(
    tf.keras.layers.InputLayer(
        batch_size=1,
        input_shape=(10),
        name="x",
    )
)
keras_sequential_model.add(tf.keras.layers.Dense(10))
keras_sequential_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# Define sklearn model and inputs
np.random.seed(0)
rf_batch_size = 320

rf_inputs = np.random.rand(rf_batch_size, 10).astype(np.float32)

rf_model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=10, max_depth=5, random_state=0
)
xgb_model = xgboost.XGBClassifier(
    n_estimators=10, max_depth=5, random_state=0, objective="binary:logistic"
)
lgbm_model = lightgbm.LGBMClassifier(n_estimators=10, max_depth=5, random_state=0)
kn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10)


# Run build_model() and get results
def full_compilation_pytorch_model():
    build_name = "full_compilation_pytorch_model"
    state = build_model(
        pytorch_model,
        inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_keras_subclass_model():
    build_name = "full_compilation_keras_subclass_model"
    state = build_model(
        keras_subclass_model,
        batch_keras_inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_keras_sequential_model():
    build_name = "full_compilation_keras_sequential_model"
    state = build_model(
        keras_sequential_model,
        batch_keras_inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_onnx_model():
    build_name = "full_compilation_onnx_model"
    torch.onnx.export(
        pytorch_model,
        input_tensor,
        "small_onnx_model.onnx",
        opset_version=build.DEFAULT_ONNX_OPSET,
        input_names=["input"],
        output_names=["output"],
    )
    state = build_model(
        "small_onnx_model.onnx",
        inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_hummingbird_rf():
    rf_model.fit(rf_inputs, np.random.randint(2, size=rf_batch_size))

    build_name = "full_compilation_hummingbird_rf"
    state = build_model(
        rf_model,
        {"input_0": rf_inputs},
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_hummingbird_xgb():
    xgb_model.fit(rf_inputs, np.random.randint(2, size=rf_batch_size))

    build_name = "full_compilation_hummingbird_xgb"
    state = build_model(
        xgb_model,
        {"input_0": rf_inputs},
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_hummingbird_lgbm():
    lgbm_model.fit(rf_inputs, np.random.randint(2, size=rf_batch_size))

    build_name = "full_compilation_hummingbird_lgbm"
    state = build_model(
        lgbm_model,
        {"input_0": rf_inputs},
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compilation_hummingbird_kn():
    kn_model.fit(rf_inputs, np.random.randint(2, size=rf_batch_size))

    build_name = "full_compilation_hummingbird_kn"
    state = build_model(
        kn_model,
        {"input_0": rf_inputs},
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def scriptmodule_functional_check():
    build_name = "scriptmodule_functional_check"
    x = torch.rand(10)
    forward_input = x
    input_dict = {"forward": forward_input}
    pytorch_module = torch.jit.trace_module(pytorch_model, input_dict)
    state = build_model(
        pytorch_module,
        inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )
    return state.build_status == build.Status.SUCCESSFUL_BUILD


def full_compile_individual_stages():
    build_name = "full_compile_individual_stages"
    build_model(
        pytorch_model,
        inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        sequence=stage.Sequence(
            "ExportPytorchModel_seq", "", [export.ExportPytorchModel()]
        ),
        cache_dir=cache_location,
    )
    build_model(
        build_name=build_name,
        sequence=stage.Sequence("OptimizeModel_seq", "", [export.OptimizeOnnxModel()]),
        cache_dir=cache_location,
    )
    build_model(
        build_name=build_name,
        sequence=stage.Sequence("Fp16Conversion_seq", "", [export.ConvertOnnxToFp16()]),
        cache_dir=cache_location,
    )
    state = build_model(
        build_name=build_name,
        sequence=stage.Sequence("SuccessStage_seq", "", [export.SuccessStage()]),
        cache_dir=cache_location,
    )

    return state.build_status == build.Status.SUCCESSFUL_BUILD


def custom_stage():
    build_name = "custom_stage"

    class MyCustomStage(stage.Stage):
        def __init__(self, funny_saying):
            super().__init__(
                unique_name="funny_fp16_convert",
                monitor_message="Funny FP16 conversion",
            )

            self.funny_saying = funny_saying

        def fire(self, state):
            input_onnx = state.intermediate_results[0]
            output_onnx = os.path.join(export.onnx_dir(state), "custom.onnx")
            fp32_model = load_model(input_onnx)
            fp16_model = convert_float_to_float16(fp32_model)
            save_model(fp16_model, output_onnx)

            print(f"funny message: {self.funny_saying}")

            state.intermediate_results = [output_onnx]

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
            export.SuccessStage(),
        ],
    )

    state = build_model(
        pytorch_model,
        inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        sequence=my_sequence,
        cache_dir=cache_location,
    )

    return state.build_status == build.Status.SUCCESSFUL_BUILD


class FullyCustomStage(stage.Stage):
    def __init__(self, saying, name):
        super().__init__(
            unique_name=name,
            monitor_message=f"Running {name}",
        )

        self.saying = saying

    def fire(self, state):
        print(self.saying)

        return state


def custom_sequence():
    build_name = "custom_sequence"
    stage_1_name = "Stage1"
    stage_2_name = "Stage2"
    stage_3_name = "Stage3"
    stage_1_msg = "Developer Velocity is"
    stage_2_msg = "Innovating"
    stage_3_msg = "Faster than ever"

    stage_1 = FullyCustomStage(stage_1_msg, stage_1_name)
    stage_2 = FullyCustomStage(stage_2_msg, stage_2_name)
    stage_3 = FullyCustomStage(stage_3_msg, stage_3_name)

    my_sequence = stage.Sequence(
        "my_stage", "Running my Sequence", stages=[stage_1, stage_2, stage_3]
    )

    build_model(
        build_name=build_name,
        monitor=False,
        rebuild="always",
        sequence=my_sequence,
        cache_dir=cache_location,
    )

    log_1_path = os.path.join(cache_location, build_name, f"log_{stage_1_name}.txt")
    log_2_path = os.path.join(cache_location, build_name, f"log_{stage_2_name}.txt")
    log_3_path = os.path.join(cache_location, build_name, f"log_{stage_3_name}.txt")

    with open(log_1_path, "r", encoding="utf8") as f:
        log_1 = f.readlines()[1]

    with open(log_2_path, "r", encoding="utf8") as f:
        log_2 = f.readlines()[1]

    with open(log_3_path, "r", encoding="utf8") as f:
        log_3 = f.readlines()[1]

    return stage_1_msg in log_1 and stage_2_msg in log_2 and stage_3_msg in log_3


def rebuild_always():
    """
    This function checks to see if the build_name.yaml file has been modified.
    If rebuild="always" the build_name_state.yaml file will have been modified along with
        the rest of the files in model/build_name due to a forced rebuild.
    If rebuild="never" the build_name_state.yaml file should *not* have been modified and
        the rest of the files in model/build_name will remain untouched and the
        model will be loaded from cache.
    To pass this test:
        between build 1 and build 2 the build_name_state.yaml file will be modified and
            therefor have different file modification timestamps
        between build 2 and build 3 the build_name_state.yaml file will *not* be modified
            resulting in identical modification timestamps.
    """
    build_name = "rebuild"
    build_timestamps = {}
    build_purpose_to_rebuild_setting = {
        "initial": "always",
        "rebuild": "always",
        "load": "never",
    }

    # Build Initial model, rebuild, and load from cache
    for build_purpose, rebuild_setting in build_purpose_to_rebuild_setting.items():
        build_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild=rebuild_setting,
            monitor=False,
            cache_dir=cache_location,
        )

        yaml_file_path = build.state_file(cache_location, build_name)

        # Read the the file modification timestamp
        if os.path.isfile(yaml_file_path):
            build_timestamps[build_purpose] = os.path.getmtime(yaml_file_path)
        else:
            msg = f"""
            The rebuild_always test attempted to load a state.yaml file
            at {yaml_file_path} but couldn't find one.
            """
            raise ValueError(msg)

    # Did the second build Rebuild?
    if build_timestamps["initial"] != build_timestamps["rebuild"]:
        rebuild = True
    else:
        rebuild = False

    # Was the third build skipped and the model loaded from cache?
    if build_timestamps["rebuild"] == build_timestamps["load"]:
        load = True
    else:
        load = False

    return rebuild and load


def rebuild_if_needed():
    """
    This function checks to see if the build_name.yaml file has been modified.
    If rebuild="always" the build_name_state.yaml file will have been modified along with
        the rest of the files in model/build_name due to a forced rebuild.
    If rebuild="if_needed" the build_name_state.yaml file should *not* have been modified and
        the rest of the files in model/build_name will remain untouched and the
        model will be loaded from cache.
    To pass this test:
        between build 1 and build 2 the build_name_state.yaml file will *not* be modified
            resulting in identical modification timestamps.
    We also toss in a state.save() call to make sure that doesn't break the cache.
    """
    build_name = "rebuild"
    build_timestamps = {}
    build_purpose_to_rebuild_setting = {
        "initial": "always",
        "load": "if_needed",
    }

    # Build Initial model, rebuild, and load from cache
    for build_purpose, rebuild_setting in build_purpose_to_rebuild_setting.items():
        state = build_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild=rebuild_setting,
            monitor=False,
            cache_dir=cache_location,
        )

        if build_purpose == "initial":
            state.save()

        yaml_file_path = build.state_file(cache_location, build_name)

        # Read the the file modification timestamp
        if os.path.isfile(yaml_file_path):
            build_timestamps[build_purpose] = os.path.getmtime(yaml_file_path)
        else:
            msg = f"""
            The rebuild_always test attempted to load a state.yaml file
            at {yaml_file_path} but couldn't find one.
            """
            raise ValueError(msg)

    # Was the third build skipped and the model loaded from cache?
    if build_timestamps["initial"] == build_timestamps["load"]:
        load = True
    else:
        load = False

    return load


def illegal_onnx_opset():
    build_name = "illegal_onnx_opset"
    torch.onnx.export(
        pytorch_model,
        input_tensor,
        "illegal_onnx_opset.onnx",
        opset_version=(build.MINIMUM_ONNX_OPSET - 1),
        input_names=["input"],
        output_names=["output"],
    )
    build_model(
        "illegal_onnx_opset.onnx",
        inputs,
        build_name=build_name,
        rebuild="always",
        monitor=False,
        cache_dir=cache_location,
    )


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        filesystem.rmdir(cache_location)

        return super().setUp()

    def test_000_rebuild_always(self):
        assert rebuild_always()

    def test_001_rebuild_if_needed(self):
        assert rebuild_if_needed()

    def test_002_full_compilation_pytorch_model(self):
        assert full_compilation_pytorch_model()

    def test_003_full_compilation_keras_sequential_model(self):
        assert full_compilation_keras_sequential_model()

    def test_004_full_compilation_keras_subclass_model(self):
        assert full_compilation_keras_subclass_model()

    def test_005_full_compilation_onnx_model(self):
        assert full_compilation_onnx_model()

    def test_006_full_compilation_hummingbird_rf(self):
        assert full_compilation_hummingbird_rf()

    def test_007_full_compilation_hummingbird_xgb(self):
        assert full_compilation_hummingbird_xgb()

    def test_008_full_compile_individual_stages(self):
        assert full_compile_individual_stages()

    def test_009_custom_stage(self):
        assert custom_stage()

    def test_010_nested_sequence(self):
        build_name = "nested_sequence"
        stage_1_name = "Stage1"
        stage_2_name = "Stage2"
        stage_3_name = "Stage3"
        stage_1_msg = "Did you know"
        stage_2_msg = "sequences can go in sequences?"
        stage_3_msg = "Indeed they can!"

        stage_1 = FullyCustomStage(stage_1_msg, stage_1_name)
        stage_2 = FullyCustomStage(stage_2_msg, stage_2_name)
        stage_3 = FullyCustomStage(stage_3_msg, stage_3_name)

        inner_sequence = stage.Sequence(
            "inner_sequence", "Running my Inner Sequence", stages=[stage_1, stage_2]
        )

        outer_sequence = stage.Sequence(
            "outer_sequence",
            "Running my Outer Sequence",
            stages=[inner_sequence, stage_3],
        )

        build_model(
            build_name=build_name,
            monitor=False,
            rebuild="always",
            sequence=outer_sequence,
            cache_dir=cache_location,
        )

        log_1_path = os.path.join(cache_location, build_name, f"log_{stage_1_name}.txt")
        log_2_path = os.path.join(cache_location, build_name, f"log_{stage_2_name}.txt")
        log_3_path = os.path.join(cache_location, build_name, f"log_{stage_3_name}.txt")

        with open(log_1_path, "r", encoding="utf8") as f:
            log_1 = f.readlines()[1]

        with open(log_2_path, "r", encoding="utf8") as f:
            log_2 = f.readlines()[1]

        with open(log_3_path, "r", encoding="utf8") as f:
            log_3 = f.readlines()[1]

        assert stage_1_msg in log_1, f"{stage_1_msg} not in {log_1}"
        assert stage_2_msg in log_2, f"{stage_2_msg} not in {log_2}"
        assert stage_3_msg in log_3, f"{stage_3_msg} not in {log_3}"

    def test_011_custom_sequence(self):
        assert custom_sequence()

    def test_012_illegal_onnx_opset(self):
        self.assertRaises(exp.StageError, illegal_onnx_opset)
        if os.path.exists("illegal_onnx_opset.onnx"):
            os.remove("illegal_onnx_opset.onnx")

    def test_013_set_onnx_opset(self):
        build_name = "full_compilation_pytorch_model"

        user_opset = 15
        assert user_opset != build.DEFAULT_ONNX_OPSET

        state = build_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
            onnx_opset=user_opset,
            sequence=sequences.optimize_fp16,
        )

        assert state.build_status == build.Status.SUCCESSFUL_BUILD

        onnx_model = onnx.load(state.results[0])
        model_opset = getattr(onnx_model.opset_import[0], "version", None)
        assert user_opset == model_opset

    def test_014_export_only(self):
        build_name = "export_only"

        state = build_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
            sequence=sequences.onnx_fp32,
        )

        assert state.build_status == build.Status.SUCCESSFUL_BUILD
        assert os.path.exists(export.base_onnx_file(state))
        assert not os.path.exists(export.opt_onnx_file(state))

    def test_015_receive_onnx(self):
        """
        Manually export an ONNX file with an opset other than the default
        Then make sure that the state file correctly reflects that opset
        """
        build_name = "receive_onnx"
        onnx_file = f"{build_name} + .onnx"
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

        # Build the ONNX file
        state = build_model(
            onnx_file,
            inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
        )

        # Make sure the build was successful
        assert state.build_status == build.Status.SUCCESSFUL_BUILD

        # Get ONNX file's opset
        onnx_model = onnx.load(onnx_file)
        model_opset = getattr(onnx_model.opset_import[0], "version", None)

        # Make sure the ONNX file matches the opset we asked for
        assert user_opset == model_opset

        # Make sure the ONNX file matches the state file
        assert model_opset == state.config.onnx_opset

    def test_016_full_compilation_hummingbird_lgbm(self):
        assert full_compilation_hummingbird_lgbm()

    def test_017_inputs_conversion(self):
        custom_sequence_fp32 = stage.Sequence(
            "custom_sequence_fp32",
            "Building Pytorch Model without fp16 conversion",
            [
                export.ExportPytorchModel(),
                export.OptimizeOnnxModel(),
            ],
            enable_model_validation=True,
        )

        custom_sequence_fp16 = stage.Sequence(
            "custom_sequence_fp16",
            "Building Pytorch Model with fp16 conversion",
            [
                export.ExportPytorchModel(),
                export.OptimizeOnnxModel(),
                export.ConvertOnnxToFp16(),
            ],
            enable_model_validation=True,
        )

        # Build model using fp32 inputs
        build_name = "custom_sequence_fp32"
        build_model(
            pytorch_model,
            inputs,
            build_name=build_name,
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
            sequence=custom_sequence_fp32,
        )

        inputs_path = os.path.join(cache_location, build_name, "inputs.npy")
        assert np.load(inputs_path, allow_pickle=True)[0]["x"].dtype == np.float32

        # Build model using fp16 inputs
        build_name = "custom_sequence_fp16"
        build_model(
            pytorch_model,
            inputs,
            build_name="custom_sequence_fp16",
            rebuild="always",
            monitor=False,
            cache_dir=cache_location,
            sequence=custom_sequence_fp16,
        )

        inputs_path = os.path.join(cache_location, build_name, "inputs.npy")
        assert np.load(inputs_path, allow_pickle=True)[0]["x"].dtype == np.float16

    def test_018_full_compilation_hummingbird_kn(self):
        assert full_compilation_hummingbird_kn()


if __name__ == "__main__":
    unittest.main()

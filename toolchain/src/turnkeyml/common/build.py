import os
import logging
import sys
import pathlib
import copy
import traceback
import platform
import subprocess
import enum
from typing import Optional, Any, List, Dict, Union, Type
from collections.abc import Collection
import dataclasses
import hashlib
import pkg_resources
import psutil
import yaml
import torch
import numpy as np
import sklearn.base
import turnkeyml.common.exceptions as exp
import turnkeyml.common.tf_helpers as tf_helpers
import turnkeyml.run.plugin_helpers as plugin_helpers
from turnkeyml.version import __version__ as turnkey_version


UnionValidModelInstanceTypes = Union[
    None,
    str,
    torch.nn.Module,
    torch.jit.ScriptModule,
    "tf.keras.Model",
    sklearn.base.BaseEstimator,
]

if os.environ.get("TURNKEY_ONNX_OPSET"):
    DEFAULT_ONNX_OPSET = int(os.environ.get("TURNKEY_ONNX_OPSET"))
else:
    DEFAULT_ONNX_OPSET = 14

MINIMUM_ONNX_OPSET = 11

DEFAULT_REBUILD_POLICY = "if_needed"
REBUILD_OPTIONS = ["if_needed", "always", "never"]


class ModelType(enum.Enum):
    PYTORCH = "pytorch"
    PYTORCH_COMPILED = "pytorch_compiled"
    KERAS = "keras"
    ONNX_FILE = "onnx_file"
    HUMMINGBIRD = "hummingbird"
    UNKNOWN = "unknown"


# Indicates that the build should take take any specific device into account
DEFAULT_DEVICE = "default"


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise exp.IOError(
                f"Failed while trying to open {file_path}."
                f"The exception that triggered this was:\n{e}"
            )


def output_dir(cache_dir, build_name):
    path = os.path.join(cache_dir, build_name)
    return path


def state_file(cache_dir, build_name):
    state_file_name = f"{build_name}_state.yaml"
    path = os.path.join(output_dir(cache_dir, build_name), state_file_name)
    return path


def hash_model(model, model_type: ModelType, hash_params: bool = True):
    # If the model is a path to a file, hash the file
    if model_type == ModelType.ONNX_FILE:
        # TODO: Implement a way of hashing the models but not the parameters
        # of ONNX inputs.
        if not hash_params:
            msg = "hash_params must be True for model_type ONNX_FILE"
            raise ValueError(msg)
        if os.path.isfile(model):
            with open(model, "rb") as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        else:
            raise ValueError(
                "hash_model received str model that doesn't correspond to a file"
            )

    elif model_type in [ModelType.PYTORCH, ModelType.PYTORCH_COMPILED]:
        # Convert model parameters and topology to string
        hashable_params = {}
        for name, param in model.named_parameters():
            hashable_params[name] = param.data
        if hash_params:
            hashable_model = (str(model) + str(hashable_params)).encode()
        else:
            hashable_model = str(model).encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    elif model_type == ModelType.KERAS:
        # Convert model parameters and topology to string
        summary_list = []  # type: List[str]

        # pylint: disable=unnecessary-lambda
        model.summary(print_fn=lambda x: summary_list.append(x))

        summary_str = " ".join(summary_list)
        hashable_params = {}
        for layer in model.layers:
            hashable_params[layer.name] = layer.weights
        if hash_params:
            hashable_model = (summary_str + str(hashable_params)).encode()
        else:
            hashable_model = summary_str.encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    elif model_type == ModelType.HUMMINGBIRD:
        import pickle

        return hashlib.sha256(pickle.dumps(model)).hexdigest()

    else:
        msg = f"""
        model_type "{model_type}" unsupported by this hash_model function
        """
        raise ValueError(msg)


class Status(enum.Enum):
    NOT_STARTED = "not_started"
    PARTIAL_BUILD = "partial_build"
    BUILD_RUNNING = "build_running"
    SUCCESSFUL_BUILD = "successful_build"
    FAILED_BUILD = "failed_build"


# Create a unique ID from this run by hashing pid + process start time
def unique_id():
    pid = os.getpid()
    p = psutil.Process(pid)
    start_time = p.create_time()
    return hashlib.sha256(f"{pid}{start_time}".encode()).hexdigest()


def get_shapes_and_dtypes(inputs: dict):
    """
    Return the shape and data type of each value in the inputs dict
    """
    shapes = {}
    dtypes = {}
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(
            value,
            (list, tuple),
        ):
            for v, i in zip(value, range(len(value))):
                subkey = f"{key}[{i}]"
                shapes[subkey] = np.array(v).shape
                dtypes[subkey] = np.array(v).dtype.name
        elif torch.is_tensor(value):
            shapes[key] = np.array(value.detach()).shape
            dtypes[key] = np.array(value.detach()).dtype.name
        elif tf_helpers.is_keras_tensor(value):
            shapes[key] = np.array(value).shape
            dtypes[key] = np.array(value).dtype.name
        elif isinstance(value, np.ndarray):
            shapes[key] = value.shape
            dtypes[key] = value.dtype.name
        elif isinstance(value, (bool, int, float)):
            shapes[key] = (1,)
            dtypes[key] = type(value).__name__
        elif value is None:
            pass
        else:
            raise exp.Error(
                "One of the provided inputs contains the unsupported "
                f' type {type(value)} at key "{key}".'
            )

    return shapes, dtypes


@dataclasses.dataclass(frozen=True)
class Config:
    """
    User-provided build configuration. Instances of Config should not be modified
    once they have been instantiated (frozen=True enforces this).

    Note: modifying this struct can create a breaking change that
    requires users to rebuild their models. Increment the minor
    version number of the turnkey package if you do make a build-
    breaking change.
    """

    build_name: str
    auto_name: bool
    sequence: List[str]
    onnx_opset: int
    device: Optional[str]


@dataclasses.dataclass
class State:
    # User-provided args that influence the generated model
    config: Config

    # User-provided args that do not influence the generated model
    monitor: bool = False
    rebuild: str = ""
    cache_dir: str = ""
    stats_id: str = ""

    # User-provided args that will not be saved as part of state.yaml
    model: UnionValidModelInstanceTypes = None
    inputs: Optional[Dict[str, Any]] = None

    # Member variable that helps the code know if State has called
    # __post_init__ yet
    save_when_setting_attribute: bool = False

    # All of the following are critical aspects of the build,
    # including properties of the tool and choices made
    # while building the model, which determine the outcome of the build.
    # NOTE: adding or changing a member name in this struct can create
    # a breaking change that requires users to rebuild their models.
    # Increment the minor version number of the turnkey package if you
    # do make a build-breaking change.

    turnkey_version: str = turnkey_version
    model_type: ModelType = ModelType.UNKNOWN
    uid: Optional[int] = None
    model_hash: Optional[int] = None
    build_status: Status = Status.NOT_STARTED
    expected_input_shapes: Optional[Dict[str, list]] = None
    expected_input_dtypes: Optional[Dict[str, list]] = None
    expected_output_names: Optional[List] = None

    # Whether or not inputs must be downcasted during inference
    downcast_applied: bool = False

    # The results of the most recent stage that was executed
    current_build_stage: str = None
    intermediate_results: Any = None

    # Results of a successful build
    results: Any = None

    quantization_samples: Optional[Collection] = None

    def __post_init__(self):
        if self.uid is None:
            self.uid = unique_id()
        if self.inputs is not None:
            (
                self.expected_input_shapes,
                self.expected_input_dtypes,
            ) = get_shapes_and_dtypes(self.inputs)
        if self.model is not None and self.model_type != ModelType.UNKNOWN:
            self.model_hash = hash_model(self.model, self.model_type)

        self.save_when_setting_attribute = True

    def __setattr__(self, name, val):
        super().__setattr__(name, val)

        # Always automatically save the state.yaml whenever State is modified
        # But don't bother saving until after __post_init__ is done (indicated
        # by the save_when_setting_attribute flag)
        # Note: This only works when elements of the state are set directly.
        if self.save_when_setting_attribute and name != "save_when_setting_attribute":
            self.save()

    @property
    def original_inputs_file(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "inputs.npy"
        )

    def prepare_file_system(self):
        # Create output folder if it doesn't exist
        os.makedirs(output_dir(self.cache_dir, self.config.build_name), exist_ok=True)

    def prepare_state_dict(self) -> Dict:
        state_dict = {
            key: value
            for key, value in vars(self).items()
            if not key == "inputs"
            and not key == "model"
            and not key == "save_when_setting_attribute"
        }

        # Special case for saving objects
        state_dict["config"] = copy.deepcopy(vars(self.config))

        state_dict["model_type"] = self.model_type.value
        state_dict["build_status"] = self.build_status.value

        # During actual execution, quantization_samples in the state
        # stores the actual quantization samples.
        # However, we do not save quantization samples
        # Instead, we save a boolean to indicate whether the model
        # stored has been quantized by some samples.
        if self.quantization_samples:
            state_dict["quantization_samples"] = True
        else:
            state_dict["quantization_samples"] = False

        return state_dict

    def save_yaml(self, state_dict: Dict):
        with open(
            state_file(self.cache_dir, self.config.build_name), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(state_dict, outfile)

    def save(self):
        self.prepare_file_system()

        state_dict = self.prepare_state_dict()

        self.save_yaml(state_dict)


def load_state(
    cache_dir=None,
    build_name=None,
    state_path=None,
    state_type: Type = State,
) -> State:
    if state_path is not None:
        file_path = state_path
    elif build_name is not None and cache_dir is not None:
        file_path = state_file(cache_dir, build_name)
    else:
        raise ValueError(
            "This function requires either build_name and cache_dir to be set, "
            "or state_path to be set, not both or neither"
        )

    state_dict = load_yaml(file_path)

    # Get the type of Config and Info in case they have been overloaded
    field_types = {field.name: field.type for field in dataclasses.fields(state_type)}
    config_type = field_types["config"]

    try:
        # Special case for loading enums
        state_dict["model_type"] = ModelType(state_dict["model_type"])
        state_dict["build_status"] = Status(state_dict["build_status"])
        state_dict["config"] = config_type(**state_dict["config"])

        state = state_type(**state_dict)

    except (KeyError, TypeError) as e:
        if state_path is not None:
            path_suggestion = pathlib.Path(state_path).parent
        else:
            path_suggestion = output_dir(cache_dir, build_name)
        msg = f"""
        The cached build of this model was built with an
        incompatible older version of the tool.

        Suggested solution: delete the build with
        rm -rf {path_suggestion}

        The underlying code raised this exception:
        {e}
        """
        raise exp.StateError(msg)

    return state


class Logger:
    """
    Redirects stdout to to file (and console if needed)
    """

    def __init__(
        self,
        initial_message: str,
        log_path: str = None,
    ):
        self.debug = os.environ.get("TURNKEY_BUILD_DEBUG") == "True"
        self.terminal = sys.stdout
        self.terminal_err = sys.stderr
        self.log_path = log_path

        # Create the empty logfile
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"{initial_message}\n")

        # Disable any existing loggers so that we can capture all
        # outputs to a logfile
        self.root_logger = logging.getLogger()
        self.handlers = [handler for handler in self.root_logger.handlers]
        for handler in self.handlers:
            self.root_logger.removeHandler(handler)

        # Send any logger outputs to the logfile
        if not self.debug:
            self.file_handler = logging.FileHandler(filename=log_path)
            self.file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self.file_handler.setFormatter(formatter)
            self.root_logger.addHandler(self.file_handler)

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, _exc_type, _exc_value, _exc_tb):
        # Ensure we also capture the traceback as part of the logger when exceptions happen
        if _exc_type:
            traceback.print_exception(_exc_type, _exc_value, _exc_tb)

        # Stop redirecting stdout/stderr
        sys.stdout = self.terminal
        sys.stderr = self.terminal_err

        # Remove the logfile logging handler
        if not self.debug:
            self.file_handler.close()
            self.root_logger.removeHandler(self.file_handler)

            # Restore any pre-existing loggers
            for handler in self.handlers:
                self.root_logger.addHandler(handler)

    def write(self, message):
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message)
        if self.debug or self.log_path is None:
            self.terminal.write(message)
            self.terminal.flush()
            self.terminal_err.write(message)
            self.terminal_err.flush()

    def flush(self):
        # needed for python 3 compatibility.
        pass


def logged_subprocess(
    cmd: List[str],
    cwd: str = os.getcwd(),
    env: Optional[Dict] = None,
    log_file_path: Optional[str] = None,
    log_to_std_streams: bool = True,
    log_to_file: bool = True,
) -> None:
    """
    This function calls a subprocess and sends the logs to either a file, stdout/stderr, or both.

    cmd             Command that will run o a sbprocess
    cwd             Working directory from where the subprocess should run
    env             Evironment to be used by the subprocess (useful for passing env vars)
    log_file_path   Where logs will be stored
    log_to_file     Whether or not to store the subprocess's stdout/stderr into a file
    log_to_std      Whether or not to print subprocess's stdout/stderr to the screen
    """
    if env is None:
        env = os.environ.copy()
    if log_to_file and log_file_path is None:
        raise ValueError("log_file_path must be set when log_to_file is True")

    log_stdout = ""
    log_stderr = ""
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            env=env,
            capture_output=True,
            cwd=cwd,
        )
    except Exception as e:  # pylint: disable=broad-except
        log_stdout = e.stdout.decode("utf-8")  # pylint: disable=no-member
        log_stderr = e.stderr.decode("utf-8")  # pylint: disable=no-member
        raise plugin_helpers.CondaError(
            f"Exception {e} encountered, \n\nstdout was: "
            f"\n{log_stdout}\n\n and stderr was: \n{log_stderr}"
        )
    else:
        log_stdout = proc.stdout.decode("utf-8")
        log_stderr = proc.stderr.decode("utf-8")
    finally:
        if log_to_std_streams:
            # Print log to stdout
            # This might be useful when this subprocess is being logged externally
            print(log_stdout, file=sys.stdout)
            print(log_stderr, file=sys.stdout)
        if log_to_file:
            log = f"{log_stdout}\n{log_stderr}"
            with open(
                log_file_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(log)


def get_system_info():
    os_type = platform.system()
    info_dict = {}

    # Get OS Version
    try:
        info_dict["OS Version"] = platform.platform()
    except Exception as e: # pylint: disable=broad-except
        info_dict["Error OS Version"] = str(e)

    if os_type == "Windows":
        # Get Processor Information
        try:
            proc_info = (
                subprocess.check_output("wmic cpu get name", shell=True)
                .decode()
                .split("\n")[1]
                .strip()
            )
            info_dict["Processor"] = proc_info
        except Exception as e: # pylint: disable=broad-except
            info_dict["Error Processor"] = str(e)

        # Get OEM System Information
        try:
            oem_info = (
                subprocess.check_output("wmic computersystem get model", shell=True)
                .decode()
                .split("\n")[1]
                .strip()
            )
            info_dict["OEM System"] = oem_info
        except Exception as e: # pylint: disable=broad-except
            info_dict["Error OEM System"] = str(e)

        # Get Physical Memory in GB
        try:
            mem_info_bytes = (
                subprocess.check_output(
                    "wmic computersystem get TotalPhysicalMemory", shell=True
                )
                .decode()
                .split("\n")[1]
                .strip()
            )
            mem_info_gb = round(int(mem_info_bytes) / (1024**3), 2)
            info_dict["Physical Memory"] = f"{mem_info_gb} GB"
        except Exception as e: # pylint: disable=broad-except
            info_dict["Error Physical Memory"] = str(e)

    elif os_type == "Linux":
        # WSL has to be handled differently compared to native Linux
        if "microsoft" in str(platform.release()):
            try:
                oem_info = (
                    subprocess.check_output(
                        'powershell.exe -Command "wmic computersystem get model"',
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
                oem_info = (
                    oem_info.replace("\r", "")
                    .replace("\n", "")
                    .split("Model")[-1]
                    .strip()
                )
                info_dict["OEM System"] = oem_info
            except Exception as e: # pylint: disable=broad-except
                info_dict["Error OEM System (WSL)"] = str(e)

        else:
            # Get OEM System Information
            try:
                oem_info = (
                    subprocess.check_output(
                        "sudo dmidecode -s system-product-name",
                        shell=True,
                    )
                    .decode()
                    .strip()
                    .replace("\n", " ")
                )
                info_dict["OEM System"] = oem_info
            except Exception as e: # pylint: disable=broad-except
                info_dict["Error OEM System"] = str(e)

        # Get CPU Information
        try:
            cpu_info = subprocess.check_output("lscpu", shell=True).decode()
            for line in cpu_info.split("\n"):
                if "Model name:" in line:
                    info_dict["Processor"] = line.split(":")[1].strip()
                    break
        except Exception as e: # pylint: disable=broad-except
            info_dict["Error Processor"] = str(e)

        # Get Memory Information
        try:
            mem_info = (
                subprocess.check_output("free -m", shell=True)
                .decode()
                .split("\n")[1]
                .split()[1]
            )
            mem_info_gb = round(int(mem_info) / 1024, 2)
            info_dict["Memory Info"] = f"{mem_info_gb} GB"
        except Exception as e: # pylint: disable=broad-except
            info_dict["Error Memory Info"] = str(e)

    else:
        info_dict["Error"] = "Unsupported OS"

    # Get Python Packages
    try:
        installed_packages = pkg_resources.working_set
        info_dict["Python Packages"] = [
            f"{i.key}=={i.version}" for i in installed_packages # pylint: disable=not-an-iterable
        ]
    except Exception as e: # pylint: disable=broad-except
        info_dict["Error Python Packages"] = str(e)

    return info_dict

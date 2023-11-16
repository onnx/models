"""
Utilities for spawning new turnkey calls, in both local processes and on Slurm
"""

import os
import subprocess
import pathlib
import time
import shlex
import platform
import getpass
from typing import List, Optional, Dict, Union
from enum import Enum
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.printing as printing
from turnkeyml.cli.parser_helpers import encode_args

if os.environ.get("TURNKEY_TIMEOUT_SECONDS"):
    timeout_env_var = os.environ.get("TURNKEY_TIMEOUT_SECONDS")
    SECONDS_IN_A_DAY = 60 * 60 * 24
    if timeout_env_var > SECONDS_IN_A_DAY:
        raise ValueError(
            f"Value of TURNKEY_TIMEOUT_SECONDS must be less than 1 day, got {timeout_env_var}"
        )
    DEFAULT_TIMEOUT_SECONDS = int(timeout_env_var)
else:
    DEFAULT_TIMEOUT_SECONDS = 3600


class Target(Enum):
    SLURM = "slurm"
    LOCAL_PROCESS = "local_process"


def slurm_jobs_in_queue(job_name=None) -> List[str]:
    """Return the set of slurm jobs that are currently pending/running"""
    user = getpass.getuser()
    if job_name is None:
        output = subprocess.check_output(["squeue", "-u", user])
    else:
        output = subprocess.check_output(["squeue", "-u", user, "--name", job_name])
    output = output.split(b"\n")
    output = [s.decode("utf").split() for s in output]

    # Remove headers
    output.pop(0)

    # Remove empty line at the end
    output.pop(-1)

    # Get just the job names
    if len(output) > 0:
        name_index_in_squeue = 2
        output = [s[name_index_in_squeue] for s in output]

    return output


def arg_format(name: str):
    name_underscores = name.replace("_", "-")
    return f"--{name_underscores}"


def list_arg(key: str, values: List):
    if values is not None:
        result = " ".join(values)
        return f"{key} {result}"
    else:
        return ""


def value_arg(key: str, value: Union[str, int]):
    if value is not None:
        return f'{key}="{value}"'
    else:
        return ""


def bool_arg(key: str, value: bool):
    if value:
        return f"{key}"
    else:
        return ""


def dict_arg(key: str, value: Dict):
    if value:
        return f"{key} {' '.join(encode_args(value))}"
    else:
        return ""


def run_turnkey(
    op: str,
    file_name: str,
    target: Target,
    timeout: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
    working_dir: str = os.getcwd(),
    ml_cache_dir: Optional[str] = os.environ.get("SLURM_ML_CACHE"),
    max_jobs: int = 50,
    **kwargs,
):
    """
    Run turnkey on a single input file in a separate process (e.g., Slurm, subprocess).
    Any arguments that should be passed to the new turnkey process must be provided via kwargs.

    kwargs must also match the following format:
      The key must be the snake_case version of the CLI argument (e.g, build_only for --build-only)
    """

    type_to_formatter = {
        str: value_arg,
        int: value_arg,
        bool: bool_arg,
        list: list_arg,
        dict: dict_arg,
    }

    invocation_args = f"{op} {file_name}"

    for key, value in kwargs.items():
        if value is not None:
            arg_str = type_to_formatter[type(value)](arg_format(key), value)
            invocation_args = invocation_args + " " + arg_str

    if target == Target.SLURM:
        # Change args into the format expected by Slurm
        slurm_args = " ".join(shlex.split(invocation_args))

        # Remove the .py extension from the build name
        job_name = filesystem.clean_file_name(file_name)

        # Put the timeout into format days-hh:mm:ss
        hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(timeout))
        slurm_format_timeout = f"00-{hh_mm_ss}"

        while len(slurm_jobs_in_queue()) >= max_jobs:
            print(
                f"Waiting: Your number of jobs running ({len(slurm_jobs_in_queue())}) "
                "matches or exceeds the maximum "
                f"concurrent jobs allowed ({max_jobs}). "
                f"The jobs in queue are: {slurm_jobs_in_queue()}"
            )
            time.sleep(5)

        shell_script = os.path.join(
            pathlib.Path(__file__).parent.resolve(), "run_slurm.sh"
        )

        slurm_command = ["sbatch", "-c", "1"]
        if os.environ.get("TURNKEY_SLURM_USE_DEFAULT_MEMORY") != "True":
            slurm_command.append("--mem=128000")
        slurm_command.extend(
            [
                f"--time={slurm_format_timeout}",
                f"--job-name={job_name}",
                shell_script,
                "turnkey",
                slurm_args,
                working_dir,
            ]
        )
        if ml_cache_dir is not None:
            slurm_command.append(ml_cache_dir)

        print(f"Submitting job {job_name} to Slurm")
        subprocess.check_call(slurm_command)
    elif target == Target.LOCAL_PROCESS:
        command = "turnkey " + invocation_args
        printing.log_info(f"Starting process with command: {command}")

        # Linux and Windows want to handle some details differently
        if platform.system() != "Windows":
            command = shlex.split(command)

        # Launch a subprocess for turnkey to evaluate the script
        try:
            subprocess.check_call(command, stderr=subprocess.STDOUT, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            printing.log_error(
                "Process was terminated with the error shown below. "
                f"turnkey will move on to the next input.\n{e}"
            )

            if "Signals.SIGKILL: 9" in str(e):
                printing.log_info(
                    "It is possible your computer ran out of memory while attempting to evaluate "
                    f"{file_name}. You can check this by repeating the experiment "
                    "while using `top` to monitor memory utilization."
                )
    else:
        raise ValueError(f"Unsupported value for target: {target}.")

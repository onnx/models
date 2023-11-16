import subprocess
import logging
import os

TIMEOUT = 900


class SubprocessError(Exception):
    pass


class CondaError(Exception):
    """
    Triggered when execution within the Conda environment goes wrong
    """


def run_subprocess(cmd):
    """Run a subprocess with the given command and log the output."""
    if isinstance(cmd, str):
        cmd_str = cmd
        shell_flag = True
    else:
        cmd_str = " ".join(cmd)
        shell_flag = False

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            shell=shell_flag,
        )
        return result
    except subprocess.TimeoutExpired:
        logging.error(f"{cmd_str} timed out after {TIMEOUT} seconds")
        raise SubprocessError("TimeoutExpired")
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Subprocess failed with command: {cmd_str} and error message: {e.stderr}"
        )
        raise SubprocessError("CalledProcessError")
    except (OSError, ValueError) as e:
        logging.error(
            f"Subprocess failed with command: {cmd_str} and error message: {str(e)}"
        )
        raise SubprocessError(str(e))


def get_python_path(conda_env_name):
    try:
        conda_path = os.getenv("CONDA_EXE")
        cmd = [
            conda_path,
            "run",
            "--name",
            conda_env_name,
            "python",
            "-c",
            "import sys; print(sys.executable)",
        ]
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        python_path = result.stdout.decode().strip()

        return python_path
    except subprocess.CalledProcessError as e:
        raise EnvironmentError(
            f"An error occurred while getting Python path for {conda_env_name} environment"
            f"{e.stderr.decode()}"
        )

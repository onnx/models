import argparse
import os
import sys
import copy
from difflib import get_close_matches
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exceptions
import turnkeyml.common.filesystem as filesystem
import turnkeyml.cli.report as report
import turnkeyml.cli.parser_helpers as parser_helpers
from turnkeyml.files_api import benchmark_files
from turnkeyml.version import __version__ as turnkey_version
from turnkeyml.run.devices import SUPPORTED_DEVICES, SUPPORTED_RUNTIMES
from turnkeyml.build.sequences import SUPPORTED_SEQUENCES
from turnkeyml.cli.spawn import DEFAULT_TIMEOUT_SECONDS


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)

    def print_cache_help(self):
        print("Error: a cache command is required")
        self.print_help()
        sys.exit(2)


def print_version(_):
    """
    Print the package version number
    """
    print(turnkey_version)


def print_stats(args):
    state_path = build.state_file(args.cache_dir, args.build_name)
    filesystem.print_yaml_file(state_path, "build state")

    filesystem.print_yaml_file(
        filesystem.Stats(args.cache_dir, args.build_name).file, "stats"
    )


def benchmark_command(args):
    """
    Map the argparse args into benchmark_files() arguments

    Assumes the following rules:
    -   All args passed to a "benchmark" command should be forwarded to the benchmark_files()
        API, except as explicitly handled below.
    -   The "dest" names of all CLI args must exactly match the names of the corresponding API arg
    """

    api_args = copy.deepcopy(vars(args))

    # Remove the function ID because it was only used to get us into this method
    api_args.pop("func")

    # Decode CLI arguments before calling the API
    api_args["rt_args"] = parser_helpers.decode_args(api_args["rt_args"])

    benchmark_files(**api_args)


def main():
    """
    Parses arguments passed by user and forwards them into a
    command function
    """

    parser = MyParser(
        description="TurnkeyML benchmarking command line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # We use sub-parsers to keep the help info neatly organized for each command
    # Sub-parses also allow us to set command-specific help on options like --cache-dir
    # that are used in multiple commands

    subparsers = parser.add_subparsers(
        title="command",
        help="Choose one of the following commands:",
        metavar="COMMAND",
        required=True,
    )

    #######################################
    # Parser for the "benchmark" command
    #######################################

    def check_extension(choices, file_name):
        _, extension = os.path.splitext(file_name.split("::")[0])
        if extension[1:].lower() not in choices:
            raise exceptions.ArgError(
                f"input_files must end with .py, .onnx, or .txt (got '{file_name}')"
            )
        return file_name

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark the performance of one or more models"
    )
    benchmark_parser.set_defaults(func=benchmark_command)

    benchmark_parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more script (.py), ONNX (.onnx), or input list (.txt) files to be benchmarked",
        type=lambda file: check_extension(("py", "onnx", "txt"), file),
    )

    slurm_or_processes_group = benchmark_parser.add_mutually_exclusive_group()

    slurm_or_processes_group.add_argument(
        "--use-slurm",
        dest="use_slurm",
        help="Execute on Slurm instead of using local compute resources",
        action="store_true",
    )

    slurm_or_processes_group.add_argument(
        "--process-isolation",
        dest="process_isolation",
        help="Isolate evaluating each input into a separate process",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--lean-cache",
        dest="lean_cache",
        help="Delete all build artifacts except for log files when the command completes",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Build cache directory where the resulting build directories will "
        f"be stored (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    benchmark_parser.add_argument(
        "--labels",
        dest="labels",
        help="Only benchmark the scripts that have the provided labels",
        nargs="*",
        default=[],
    )

    benchmark_parser.add_argument(
        "--sequence",
        choices=SUPPORTED_SEQUENCES.keys(),
        dest="sequence",
        help="Name of a build sequence that will define the model-to-model transformations, "
        "used to build the models. Each runtime has a default sequence that it uses.",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--rebuild",
        choices=build.REBUILD_OPTIONS,
        dest="rebuild",
        help=f"Sets the cache rebuild policy (defaults to {build.DEFAULT_REBUILD_POLICY})",
        required=False,
        default=build.DEFAULT_REBUILD_POLICY,
    )

    benchmark_default_device = "x86"
    benchmark_parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICES,
        dest="device",
        help="Type of hardware device to be used for the benchmark "
        f'(defaults to "{benchmark_default_device}")',
        required=False,
        default=benchmark_default_device,
    )

    benchmark_parser.add_argument(
        "--runtime",
        choices=SUPPORTED_RUNTIMES.keys(),
        dest="runtime",
        help="Software runtime that will be used to collect the benchmark. "
        "Must be compatible with the selected device. "
        "Automatically selects a sequence if `--sequence` is not used."
        "If this argument is not set, the default runtime of the selected device will be used.",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=100,
        help="Number of execution iterations of the model to capture\
              the benchmarking performance (e.g., mean latency)",
    )

    benchmark_parser.add_argument(
        "--analyze-only",
        dest="analyze_only",
        help="Stop this command after the analysis phase",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--build-only",
        dest="build_only",
        help="Stop this command after the build phase",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--script-args",
        dest="script_args",
        type=str,
        help="Arguments to pass into the target script(s)",
    )

    benchmark_parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=0,
        help="Maximum depth to analyze within the model structure of the target script(s)",
    )

    benchmark_parser.add_argument(
        "--onnx-opset",
        dest="onnx_opset",
        type=int,
        default=None,
        help=f"ONNX opset used when creating ONNX files (default={build.DEFAULT_ONNX_OPSET}). "
        "Not applicable when input model is already a .onnx file.",
    )

    benchmark_parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Build timeout, in seconds, after which a build will be canceled "
        f"(default={DEFAULT_TIMEOUT_SECONDS}). Only "
        "applies when --process-isolation or --use-slurm is also used.",
    )

    benchmark_parser.add_argument(
        "--rt-args",
        dest="rt_args",
        type=str,
        nargs="*",
        help="Optional arguments provided to the runtime being used",
    )

    #######################################
    # Subparser for the "cache" command
    #######################################

    cache_parser = subparsers.add_parser(
        "cache",
        help="Commands for managing the build cache",
    )

    cache_subparsers = cache_parser.add_subparsers(
        title="cache",
        help="Commands for managing the build cache",
        required=True,
        dest="cache_cmd",
    )

    #######################################
    # Parser for the "cache report" command
    #######################################

    report_parser = cache_subparsers.add_parser(
        "report", help="Generate reports in CSV format"
    )
    report_parser.set_defaults(func=report.summary_spreadsheets)

    report_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dirs",
        help=(
            "One or more build cache directories to generate the report "
            f"(defaults to {filesystem.DEFAULT_CACHE_DIR})"
        ),
        default=[filesystem.DEFAULT_CACHE_DIR],
        nargs="*",
    )

    report_parser.add_argument(
        "-r",
        "--report-dir",
        dest="report_dir",
        help="Path to folder where report will be saved (defaults to current working directory)",
        required=False,
        default=os.getcwd(),
    )

    #######################################
    # Parser for the "cache list" command
    #######################################

    list_parser = cache_subparsers.add_parser(
        "list", help="List all builds in a target cache"
    )
    list_parser.set_defaults(func=filesystem.print_available_builds)

    list_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="The builds in this build cache directory will printed to the terminal "
        f" (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    #######################################
    # Parser for the "cache stats" command
    #######################################

    stats_parser = cache_subparsers.add_parser(
        "stats", help="Print stats about a build in a target cache"
    )
    stats_parser.set_defaults(func=print_stats)

    stats_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="The stats of a build in this build cache directory will printed to the terminal "
        f" (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    stats_parser.add_argument(
        "build_name",
        help="Name of the specific build whose stats are to be printed, within the cache directory",
    )

    #######################################
    # Parser for the "cache delete" command
    #######################################

    delete_parser = cache_subparsers.add_parser(
        "delete", help="Delete one or more builds in a build cache"
    )
    delete_parser.set_defaults(func=filesystem.delete_builds)

    delete_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Search path for builds " f"(defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    delete_group = delete_parser.add_mutually_exclusive_group(required=True)

    delete_group.add_argument(
        "build_name",
        nargs="?",
        help="Name of the specific build to be deleted, within the cache directory",
    )

    delete_group.add_argument(
        "--all",
        dest="delete_all",
        help="Delete all builds in the cache directory",
        action="store_true",
    )

    #######################################
    # Parser for the "cache clean" command
    #######################################

    clean_parser = cache_subparsers.add_parser(
        "clean",
        help="Remove the build artifacts from one or more builds in a build cache",
    )
    clean_parser.set_defaults(func=filesystem.clean_builds)

    clean_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Search path for builds " f"(defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    clean_group = clean_parser.add_mutually_exclusive_group(required=True)

    clean_group.add_argument(
        "build_name",
        nargs="?",
        help="Name of the specific build to be cleaned, within the cache directory",
    )

    clean_group.add_argument(
        "--all",
        dest="clean_all",
        help="Clean all builds in the cache directory",
        action="store_true",
    )

    #######################################
    # Parser for the "cache location" command
    #######################################

    cache_location_parser = cache_subparsers.add_parser(
        "location",
        help="Print the location of the default build cache directory",
    )
    cache_location_parser.set_defaults(func=filesystem.print_cache_dir)

    #######################################
    # Subparser for the "models" command
    #######################################

    models_parser = subparsers.add_parser(
        "models",
        help="Commands for managing the models",
    )

    # Design note: the `models` command is simple right now, however some additional ideas
    #    are documented in https://github.com/aig-bench/onnxmodelzoo/issues/247

    models_subparsers = models_parser.add_subparsers(
        title="models",
        help="Commands for managing the models",
        required=True,
        dest="models_cmd",
    )

    models_location_parser = models_subparsers.add_parser(
        "location",
        help="Print the location of the models directory",
    )
    models_location_parser.set_defaults(func=filesystem.print_models_dir)

    models_location_parser.add_argument(
        "--quiet",
        dest="verbose",
        help="Command output will only include the directory path",
        required=False,
        action="store_false",
    )

    #######################################
    # Parser for the "version" command
    #######################################

    version_parser = subparsers.add_parser(
        "version",
        help="Print the package version number",
    )
    version_parser.set_defaults(func=print_version)

    #######################################
    # Execute the command
    #######################################

    # The default behavior of this CLI is to run the build command
    # on a target script. If the user doesn't provide a command,
    # we alter argv to insert the command for them.

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg not in subparsers.choices.keys() and "-h" not in first_arg:
            if "." in first_arg:
                sys.argv.insert(1, "benchmark")
            else:
                # Check how close we are from each of the valid options
                valid_options = list(subparsers.choices.keys())
                close_matches = get_close_matches(first_arg, valid_options)

                error_msg = f"Unexpected positional argument `turnkey {first_arg}`. "
                if close_matches:
                    error_msg += f"Did you mean `turnkey {close_matches[0]}`?"
                else:
                    error_msg += (
                        "The first positional argument must either be "
                        "an input file with the .py or .onnx file extension or "
                        f"one of the following commands: {valid_options}."
                    )
                raise exceptions.ArgError(error_msg)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()

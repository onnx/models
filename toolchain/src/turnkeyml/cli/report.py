import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import yaml
import pandas as pd
import turnkeyml.common.printing as printing
import turnkeyml.common.filesystem as fs


def get_report_name(prefix: str = "") -> str:
    """
    Returns the name of the .csv report
    """
    day = datetime.now().day
    month = datetime.now().month
    year = datetime.now().year
    date_key = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    return f"{prefix}{date_key}.csv"


def _good_get(
    dict: Dict, key: str, return_keys: bool = False, return_values: bool = False
):
    if key in dict:
        if return_keys:
            return list(dict[key].keys())
        elif return_values:
            return list(dict[key].values())
        else:
            return dict[key]
    else:
        return "-"


def summary_spreadsheets(args) -> None:
    # Input arguments from CLI
    cache_dirs = [os.path.expanduser(dir) for dir in args.cache_dirs]
    cache_dirs = fs.expand_inputs(cache_dirs)
    report_dir = os.path.expanduser(args.report_dir)

    # Name report file
    report_path = os.path.join(report_dir, get_report_name())

    # Create report dict
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    report: List[Dict] = []
    all_build_stats = []

    # Add results from all user-provided cache folders
    for cache_dir in cache_dirs:
        # Check if this is a valid cache directory
        fs.check_cache_dir(cache_dir)

        # List all yaml files available
        all_model_stats_yamls = fs.get_all(
            path=cache_dir, file_type="turnkey_stats.yaml"
        )
        all_model_stats_yamls = sorted(all_model_stats_yamls)

        # Bring all of the stats for all of the models into memory
        for model_stats_yaml in all_model_stats_yamls:
            with open(model_stats_yaml, "r", encoding="utf8") as stream:
                try:
                    # load the yaml into a dict
                    model_stats = yaml.load(stream, Loader=yaml.FullLoader)

                    # create a separate dict for each build
                    for build in model_stats[fs.Keys.BUILDS].values():
                        build_stats = {}

                        # Copy all of the stats for the model that are common across builds
                        for key, value in model_stats.items():
                            if key != fs.Keys.BUILDS:
                                build_stats[key] = value

                        # Copy the build-specific stats
                        for key, value in build.items():
                            # Break each value in "completed build stages" into its own column
                            # to make analysis easier
                            if key == fs.Keys.COMPLETED_BUILD_STAGES:
                                for subkey, subvalue in value.items():
                                    build_stats[subkey] = subvalue

                            # If a build is still marked as "running" at reporting time, it
                            # must have been killed by a time out, out-of-memory (OOM), or some
                            # other uncaught exception
                            if (
                                key == fs.Keys.BENCHMARK_STATUS
                                and value == fs.BenchmarkStatus.RUNNING
                            ):
                                value = fs.BenchmarkStatus.KILLED

                            build_stats[key] = value

                        all_build_stats.append(build_stats)
                except yaml.scanner.ScannerError:
                    continue

        # Scan the build stats to determine the set of columns for the CSV file.
        # The CSV will have one column for every key in any build stats dict.
        column_headers = []
        for build_stats in all_build_stats:
            # Add any key that isn't already in column_headers
            for header in build_stats.keys():
                if header not in column_headers:
                    column_headers.append(header)

        # Add each build to the report
        for build_stats in all_build_stats:
            # Start with a dictionary where all of the values are "-". If a build
            # has a value for each key we will fill it in, and otherwise the "-"
            # will indicate that no value was available
            result = {k: "-" for k in column_headers}

            for key in column_headers:
                result[key] = _good_get(build_stats, key)

            report.append(result)

    # Populate results spreadsheet
    with open(report_path, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)
        writer.writerow(column_headers)
        for build in report:
            writer.writerow([build[col] for col in column_headers])

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(report_path), printing.Colors.OKGREEN)

    # Save the unique errors and counts to a file
    errors = []
    for build_stats in all_build_stats:
        if (
            "compilation_error" in build_stats.keys()
            and "compilation_error_id" in build_stats.keys()
        ):
            error = build_stats["compilation_error"]
            id = build_stats["compilation_error_id"]
            if id != "":
                unique_error = True
                for reported_error in errors:
                    if reported_error["id"] == id:
                        unique_error = False
                        reported_error["count"] = reported_error["count"] + 1
                        reported_error["models_impacted"] = reported_error[
                            "models_impacted"
                        ] + [build_stats["model_name"]]

                if unique_error:
                    reported_error = {
                        "id": id,
                        "count": 1,
                        "models_impacted": [build_stats["model_name"]],
                        "example": error,
                    }
                    errors.append(reported_error)

    if len(errors) > 0:
        errors_path = os.path.join(report_dir, get_report_name("errors-"))
        with open(errors_path, "w", newline="", encoding="utf8") as spreadsheet:
            writer = csv.writer(spreadsheet)
            error_headers = errors[0].keys()
            writer.writerow(error_headers)
            for unique_error in errors:
                writer.writerow([unique_error[col] for col in error_headers])

        printing.log("Compilation errors spreadsheet saved at ")
        printing.logn(str(errors_path), printing.Colors.OKGREEN)
    else:
        printing.logn(
            "No compilation errors in any cached build, skipping errors spreadsheet."
        )


def get_dict(report_csv: str, columns: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Returns a dictionary where the keys are model names and the values are dictionaries.
    Each dictionary represents a model with column names as keys and their corresponding values.
    args:
     - report_csv: path to a report.csv file generated by turnkey CLI
     - columns: list of column names in the report.csv file whose values will be used to
        populate the dictionary
    """

    # Load the report as a dataframe
    dataframe = pd.read_csv(report_csv)

    # Create a nested dictionary with model_name as keys and another
    # dictionary of {column: value} pairs as values
    result = {
        row[0]: row[1].to_dict()
        for row in dataframe.set_index("model_name")[columns].iterrows()
    }

    return result

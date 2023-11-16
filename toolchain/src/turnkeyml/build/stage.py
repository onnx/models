import abc
import sys
import time
import os
from typing import List, Tuple
from multiprocessing import Process
import psutil
import turnkeyml.common.printing as printing
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as fs


def _spinner(message):
    parent_pid = os.getppid()
    while psutil.pid_exists(parent_pid):
        for cursor in ["   ", ".  ", ".. ", "..."]:
            time.sleep(0.5)
            status = f"      {message}{cursor}\r"
            sys.stdout.write(status)
            sys.stdout.flush()


def _name_is_file_safe(name: str):
    """
    Make sure the name can be used in a filename
    """

    allowed_in_unique_name = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    )

    if len(name) == 0:
        msg = """
        Stage __init__() was passed a unique_name with no length. A
        uniquely identifying unique_name is required.
        """
        raise ValueError(msg)

    for char in name:
        if char not in allowed_in_unique_name:
            msg = f"""
            Stage __init__() was passed a unique_name:
            {name}
            with illegal characters. The unique_name must be safe to
            use in a filename, meaning it can only use characters: {allowed_in_unique_name}
            """
            raise ValueError(msg)


class Stage(abc.ABC):
    def status_line(self, successful, verbosity):
        """
        Print a line of status information for this Stage into the monitor.
        """
        if verbosity:
            # Only use special characters when the terminal encoding supports it
            if sys.stdout.encoding == "utf-8":
                success_tick = "✓"
                fail_tick = "×"
            else:
                success_tick = "+"
                fail_tick = "x"

            if successful is None:
                # Initialize the message
                printing.logn(f"      {self.monitor_message}   ")
            elif successful:
                # Print success message
                printing.log(f"    {success_tick} ", c=printing.Colors.OKGREEN)
                printing.logn(self.monitor_message + "   ")
            else:
                # successful == False, print failure message
                printing.log(f"    {fail_tick} ", c=printing.Colors.FAIL)
                printing.logn(self.monitor_message + "   ")

    def __init__(
        self,
        unique_name,
        monitor_message,
    ):
        _name_is_file_safe(unique_name)

        self.unique_name = unique_name
        self.monitor_message = monitor_message
        self.progress = None
        self.logfile_path = None
        self.stages = None

    @abc.abstractmethod
    def fire(self, state: build.State) -> build.State:
        """
        Developer-defined function to fire the stage.
        In less punny terms, this is the function that
        build_model() will run to implement a model-to-model
        transformation on the flow to producing a Model.
        """

    def fire_helper(self, state: build.State) -> Tuple[build.State, int]:
        """
        Wraps the user-defined .fire method with helper functionality.
        Specifically:
            - Provides a path to a log file
            - Redirects the stdout of the stage to that log file
            - Monitors the progress of the stage on the command line,
                including in the event of an exception
        """

        # Set the build status to BUILD_RUNNING to indicate that a Stage
        # started running. This allows us to test whether the Stage exited
        # unexpectedly, before it was able to set FAILED_BUILD, SUCCESSFUL_BUILD,
        # or PARTIAL_BUILD
        state.build_status = build.Status.BUILD_RUNNING

        self.logfile_path = os.path.join(
            build.output_dir(state.cache_dir, state.config.build_name),
            f"log_{self.unique_name}.txt",
        )

        if state.monitor:
            self.progress = Process(target=_spinner, args=[self.monitor_message])
            self.progress.start()

        try:
            # Execute the build stage
            with build.Logger(self.monitor_message, self.logfile_path):
                state = self.fire(state)

        except exp.StageError:
            self.status_line(
                successful=False,
                verbosity=state.monitor,
            )
            state.build_status = build.Status.FAILED_BUILD
            raise

        else:
            self.status_line(successful=True, verbosity=state.monitor)

            # Set the build status PARTIAL_BUILD, indicating that the stage
            # ran successfully, unless the stage set SUCCESSFUL_BUILD, in which
            # case leave the build status alone.
            if state.build_status != build.Status.SUCCESSFUL_BUILD:
                state.build_status = build.Status.PARTIAL_BUILD

        finally:
            if state.monitor:
                self.progress.terminate()

        return state

    def get_names(self) -> List[str]:
        """
        Sequence uses self.names() to recursively get the names of all
        Stages in the Sequence. An individual Stage just needs to return
        its own name.
        """
        if self.stages is None:
            return [self.unique_name]
        else:
            result = []
            for stage in self.stages:
                result = result + stage.get_names()

            return result

    def get_depth(self) -> int:
        """
        Sequence needs to know the depth of each Stage within the Sequence in order
        to properly update the terminal UI. An individual Stage just needs to return
        the value 1.
        """
        if self.stages is None:
            return 1
        else:
            count = 0
            for stage in self.stages:
                count = count + stage.get_depth()
            return count


def _rewind_stdout(lines: int):
    """
    Helper function for the command line monitor. Moves the cursor up a
    certain number of lines in the terminal, corresponding to the
    status line for a Stage, so that we can update the status of
    that Stage.
    """
    rewind_stdout_one_line = "\033[1A"
    rewind_multiple_lines = rewind_stdout_one_line * lines
    print(rewind_multiple_lines, end="")


def unroll_stages(stages):
    """
    Recursively goes through all sequences and returns list of stages
    """

    unrolled_stages = []
    for stage in stages:
        if isinstance(stage, Sequence):
            unrolled_stages += unroll_stages(stage.stages)
        else:
            unrolled_stages += [stage]
    return unrolled_stages


class Sequence(Stage):
    def __init__(
        self,
        unique_name,
        monitor_message,
        stages: List[Stage],
        enable_model_validation=False,
    ):
        super().__init__(unique_name, monitor_message)

        # The `stages` argument can be a nested Sequence (ie, Sequence of Sequence of Stage).
        # Unroll the stages to make the Sequence easier to deal with
        self.stages = unroll_stages(stages)

        # Follow default model validation steps in ignition.model_intake()
        self.enable_model_validation = enable_model_validation

        # Make sure all the stage names are unique
        stage_names = self.get_names()

        if len(stage_names) != len(set(stage_names)):
            msg = f"""
            All Stages in a Sequence must have unique unique_names, however Sequence
            received duplicates in the list of names: {stage_names}
            """
            raise ValueError(msg)

    def show_monitor(self, config: build.Config, verbosity: bool):
        """
        Displays the monitor on the terminal. The purpose of the monitor
        is to show the status of each stage (success, failure, not started yet,
        or in-progress).
        """

        if verbosity:
            print("\n\n")

            printing.logn(
                f'Building "{config.build_name}"',
                c=printing.Colors.BOLD,
            )

            for stage in self.stages:
                stage.status_line(successful=None, verbosity=True)

            _rewind_stdout(self.get_depth())

    def launch(self, state: build.State) -> build.State:
        """
        Executes a launch sequence.
        In less punny terms, this method is called by the top-level
        build_model() function to iterate over all of the Stages required for a build.
        Builds are defined by self.stages in a top-level Sequence, and self.stages
        can include both Stages and Sequences (ie, sequences can be nested).
        """

        if state.build_status == build.Status.NOT_STARTED:
            state.build_status = build.Status.PARTIAL_BUILD
        elif state.build_status == build.Status.SUCCESSFUL_BUILD:
            msg = """
            build_model() is running a build on a model that already built successfully, which
            should not happen because the build should have loaded from cache or rebuilt from scratch.
            If you are using custom Stages and Sequences then you have some debugging to do. Otherwise,
            please file an issue at https://github.com/aig-bench/onnxmodelzoo/issues
            """
            raise exp.Error(msg)

        # Collect telemetry for the build
        stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
        stats.add_build_stat(
            fs.Keys.ALL_BUILD_STAGES,
            self.get_names(),
        )

        # Run the build
        try:
            for stage in self.stages:
                # Collect telemetry about the stage
                state.current_build_stage = stage.unique_name
                start_time = time.time()

                # Run the stage
                state = stage.fire_helper(state)

                # Collect telemetry about the stage
                execution_time = time.time() - start_time

                stats.add_build_sub_stat(
                    parent_key=fs.Keys.COMPLETED_BUILD_STAGES,
                    key=stage.unique_name,
                    value=execution_time,
                )

        except exp.StageError as e:
            # Advance the cursor below the monitor so
            # we can print an error message
            stage_depth_in_sequence = self.get_depth() - self.get_names().index(
                stage.unique_name
            )
            stdout_lines_to_advance = stage_depth_in_sequence - 2
            cursor_down = "\n" * stdout_lines_to_advance

            print(cursor_down)

            printing.log_error(e)
            raise

        else:
            state.current_build_stage = None
            return state

    def status_line(self, successful, verbosity):
        """
        This override of status_line simply propagates status_line()
        to every Stage in the Sequence
        FIXME: A cleaner implementation of Stage/Sequence might not need this
        """
        for stage in self.stages:
            stage.status_line(successful=None, verbosity=verbosity)

    def fire(self, state: build.State) -> build.State:
        """
        This override of fire simply propagates fire()
        to every Stage in the Sequence
        FIXME: A cleaner implementation of Stage/Sequence might not need this
        """
        for stage in self.stages:
            state = stage.fire_helper(state)

        return state

    def fire_helper(self, state: build.State) -> build.State:
        """
        Sequence doesn't need any help calling self.fire(), so it's fire_helper
        is just to call self.fire()
        FIXME: A cleaner implementation of Stage/Sequence might not need this
        """
        return self.fire(state)

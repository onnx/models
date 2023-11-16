import turnkeyml.common.printing as printing


class Error(Exception):
    """
    Indicates something has gone wrong while running the tools
    """

    def __init__(self, msg):
        super().__init__(msg)
        printing.log_error(msg)


class CacheError(Error):
    """
    Indicates ambiguous behavior from when a build already exists in the cache,
    but the model, inputs, or args have changed thereby invalidating
    the cached copy of the model.
    """


class EnvError(Error):
    """
    Indicates to the user that the required tools are not
    available on their PATH.
    """


class ArgError(Error):
    """
    Indicates to the user that they provided invalid arguments
    """


class StageError(Exception):
    """
    Let the user know that something went wrong while
    firing off a Stage.

    Note: not overloading __init__() so that the
    attempt to print to stdout isn't captured into
    the Stage's log file.
    """


class StateError(Exception):
    """
    Raised when something goes wrong with State
    """


class IntakeError(Exception):
    """
    Let the user know that something went wrong during the
    initial intake process of analyzing a model.
    """


class IOError(Error):
    """
    Indicates to the user that an input/output operation failed,
    such trying to open a file.
    """


class ModelArgError(Error):
    """
    Indicates to the user that values provided to a Model instance method
    were not allowed.
    """


class ModelRuntimeError(Error):
    """
    Indicates to the user that attempting to invoke a Model instance failed.
    """


class BenchmarkException(Exception):
    """
    Indicates a failure during benchmarking
    """


class HardwareError(Error):
    """
    Indicates that the hardware used is faulty or unavailable.
    """


class SkipBuild(Exception):
    """
    Indicates that an exception is deliberately being raised to skip a build
    """

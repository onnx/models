"""
This script is an example of a sequence.py file for Sequence Plugin. Such a sequence.py 
can be used to redefine the build phase of the turnkey CLI, benchmark_files(),
and benchmark_model() to have any custom behavior.

In this example sequence.py file we are setting the build sequence to simply
export from pytorch to ONNX. This differs from the default build sequence, which
exports to ONNX, optimizes, and converts to float16.

After you install the plugin, you can tell `turnkey` to use this sequence with:

    turnkey benchmark INPUT_SCRIPTS --sequence exampleseq
"""

from turnkeyml.build.stage import Sequence, Stage
import turnkeyml.common.build as build
import turnkeyml.build.export as export


example_seq_name = "example-seq"


class ExampleStage(Stage):
    """
    This is an empty Stage that we include in our example Sequence. Its purpose
    is to display the monitor_message during the build so that you can see that the
    example Sequence is really running.
    """

    def __init__(self):
        super().__init__(
            unique_name="teaching_by_example",
            monitor_message="Teaching by example",
        )

    def fire(self, state: build.State):
        return state


example_sequence = Sequence(
    unique_name="example_sequence",
    monitor_message="Example sequence for a plugin",
    stages=[
        export.ExportPlaceholder(),
        ExampleStage(),
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

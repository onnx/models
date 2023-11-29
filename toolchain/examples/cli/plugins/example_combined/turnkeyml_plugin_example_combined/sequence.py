from turnkeyml.build.stage import Sequence, Stage
import turnkeyml.common.build as build
import turnkeyml.build.export as export

combined_seq_name = "example-combined-seq"


class CombinedExampleStage(Stage):
    """
    This is an empty Stage that we include in our example that provides both
    a sequence and a runtime in a single plugin package.
    """

    def __init__(self):
        super().__init__(
            unique_name="combined_example",
            monitor_message="Special step expected by CombinedExampleRT",
        )

    def fire(self, state: build.State):
        return state


combined_example_sequence = Sequence(
    unique_name="combined_example_sequence",
    monitor_message="Example sequence expected by CombinedExampleRT",
    stages=[
        export.ExportPlaceholder(),
        CombinedExampleStage(),
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

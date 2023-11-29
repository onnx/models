import turnkeyml.build.export as export
import turnkeyml.build.stage as stage
import turnkeyml.common.plugins as plugins

optimize_fp16 = stage.Sequence(
    "optimize_fp16",
    "Optimized FP16 ONNX file",
    [
        export.ExportPlaceholder(),
        export.OptimizeOnnxModel(),
        export.ConvertOnnxToFp16(),
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

optimize_fp32 = stage.Sequence(
    "optimize_fp32",
    "Optimized FP32 ONNX File",
    [
        export.ExportPlaceholder(),
        export.OptimizeOnnxModel(),
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

onnx_fp32 = stage.Sequence(
    "onnx_fp32",
    "Base Sequence",
    [
        export.ExportPlaceholder(),
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

pytorch_with_quantization = stage.Sequence(
    "pytorch_export_sequence_with_quantization",
    "Exporting PyTorch Model and Quantizing Exported ONNX",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        export.QuantizeONNXModel(),
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

# Plugin interface for sequences
discovered_plugins = plugins.discover()

# Populated supported sequences dict with builtin sequences
SUPPORTED_SEQUENCES = {
    "optimize-fp16": optimize_fp16,
    "optimize-fp32": optimize_fp32,
    "onnx-fp32": onnx_fp32,
}

# Add sequences from plugins to supported sequences dict
for module in discovered_plugins.values():
    if "sequences" in module.implements.keys():
        for seq_name, seq_info in module.implements["sequences"].items():
            if seq_name in SUPPORTED_SEQUENCES:
                raise ValueError(
                    f"Your turnkeyml installation has two sequences named '{seq_name}' "
                    "installed. You must uninstall one of your plugins that includes "
                    f"{seq_name}. Your imported sequence plugins are: {SUPPORTED_SEQUENCES}\n"
                    f"This error was thrown while trying to import {module}"
                )

            SUPPORTED_SEQUENCES[seq_name] = seq_info["sequence_instance"]

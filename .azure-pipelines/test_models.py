import onnx
import numpy as np
from pathlib import Path

# identify list of onnx models in model Zoo
model_list = []

for path in Path('text').rglob('*.onnx'):
    model_list.append(path.name)

for path in Path('vision').rglob('*.onnx'):
    model_list.append(path.name)

# run checker on each model
for model_name in model_list:
    model = onnx.load(model_name)
    onnx.checker.check_model(model)

    print("Model ", model_name, "has been successfully checked!")

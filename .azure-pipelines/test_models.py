import onnx
from pathlib import Path

# identify list of onnx models in model Zoo
model_list = []

for path in Path('text').rglob('*.onnx'):
    if path.stat().st_size >= 200:
        model_list.append(str(path))

for path in Path('vision').rglob('*.onnx'):
    if path.stat().st_size >= 200:
        model_list.append(str(path))

# run checker on each model
for model_name in model_list:
    model = onnx.load(model_name)
    onnx.checker.check_model(model)

    print("Model ", model_name, "has been successfully checked!")

print(len(model_list), " models checked.")

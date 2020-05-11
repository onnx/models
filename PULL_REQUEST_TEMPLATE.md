# Model Name

## Description
Description of model - What task does it address (i.e. object detection, image classification)? What is the main advantage or feature of its architecture?

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Model Name       | Relative link to ONNX Model with size  | tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |

Please submit new models with Git LFS (commit directly to the repository, and use relative links (i.e. ***model/alexnet-7.onnx***) in the table above. In this file name example, ***alexnet*** is the name of the model and ***7*** is the opset number.

### Source
Source Framework ==> ONNX model
i.e. (Caffe2 DenseNet-121 ==> ONNX DenseNet)

All ONNX models must pass the ONNX model checker before contribution. The snippet of code below can be used to perform the check. If any errors are encountered, it implies the check has failed.

```
import onnx
from onnx import checker
model_proto = onnx.load("path to .onnx file")
checker.check_model(model_proto)
```

## Inference
Step by step instructions on how to use the pretrained model and link to an example notebook/code. This section should ideally contain:

### Input
Input to network (Example: 224x224 pixels in RGB)

### Preprocessing
Preprocessing required

### Output
Output of network

### Postprocessing
Post processing and meaning of output

## Model Creation

### Dataset (Train and validation)
This section should discuss datasets and any preparation steps if required.
<hr>

### Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional).

Also clarify in case the model is not trained from scratch and include the source/process used to obtain the ONNX model.

### Validation accuracy
Validation script/notebook used to obtain accuracy reported above along with details of how to use it and reproduce accuracy. Details of experiments leading to accuracy from the reference paper.
<hr>

### References
Link to paper or references.
<hr>

## Contributors
Contributors' name
<hr>

## License
Add license information - on default, Apache 2.0
<hr>

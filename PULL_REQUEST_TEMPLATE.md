# Model Name

## Use-cases
Model use cases

## Description
Description of model

## Model

 |Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Model Name       | Relative link to ONNX Model with size  | tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |

Please submit new models with Git LFS (commit directly to the repository, and use relative links (i.e. ***model/alexnet-7.onnx***) in the table above. In this file name example, ***alexnet*** is the name of the model and ***7*** is the opset number.

All ONNX models must pass the ONNX model checker before contribution. The snippet of code below can be used to perform the check. If any errors are encountered, it implies the check has failed.

```
import onnx
from onnx import checker
model_proto = onnx.load("path to .onnx file")
checker.check_model(model_proto)
```

## Inference
Step by step instructions on how to use the pretrained model and link to an example notebook/code.

**Conversion Path**: i.e. PyTorch -> ONNX

This section should ideally contain:

### Input
Input to network (Example: 224x224 pixels in RGB)

### Preprocessing
Preprocessing required

### Output
Output of network

### Postprocessing
Post processing and meaning of output


## Dataset (Train and validation)
This section should discuss datasets and any preparation steps if required.
<hr>

### Validation accuracy
Validation script/notebook used to obtain accuracy reported above along with details of how to use it and reproduce accuracy. Details of experiments leading to accuracy and comparison with the reference paper.
<hr>

### Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional).

Also clarify in case the model is not trained from scratch and include the source/process used to obtain the ONNX model.

## References
Link to references
<hr>

## Contributors
Contributors' name
<hr>

## License
Add license information - on default, Apache 2.0
<hr>

# Model Name

## Use-cases
Model use cases

## Description
Description of model

## Model

 |Model        |Download  |Checksum| Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Model Name       |ONNX Model download link with size  | MD5 checksum for the ONNX model| tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |

All ONNX models must pass the ONNX model checker before contribution. The snippet of code below can be used to perform the check. If any errors are encountered, it implies the check has failed.

```
import onnx
from onnx import checker
model_proto = onnx.load("path to .onnx file")
checker.check_model(model_proto)
```

<hr>

## Inference
Step by step instructions on how to use the pretrained model and link to an example notebook/code. This section should ideally contain:

### Input to model
Input to network (Example: 224x224 pixels in RGB)

### Preprocessing steps
Preprocessing required

### Output of model
Output of network

### Postprocessing steps
Post processing and meaning of output
<hr>

## Dataset (Train and validation)
This section should discuss datasets and any preparation steps if required.
<hr>

## Validation accuracy
Details of experiments leading to accuracy and comparison with the reference paper.
<hr>

## Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional). 

Also clarify in case the model is not trained from scratch and include the source/process used to obtain the ONNX model.
<hr>

## Validation
Validation script/notebook used to obtain accuracy reported above along with details of how to use it and reproduce accuracy.
<hr>

## References
Link to references
<hr>

## Contributors
Contributors' name
<hr>

## License
Add license information
<hr>

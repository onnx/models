# Model

|Model        |Download  |ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|
|Model Name       | Relative link to ONNX Model with size  | ONNX version used for conversion| Opset version used for conversion|Accuracy values |
|Example (VGG 19)| [548.1 MB](classification/vgg/model/vgg19-7.onnx) | 1.2.1  |7 | 73.72     |

Please submit new models with Git LFS by committing directly to the repository, and using relative links (i.e. ***model/vgg19-7.onnx***) in the table above. In this file name example, ***vgg19*** is the name of the model and ***7*** is the opset number.

## Source
Source Framework ==> ONNX model  
i.e. Caffe2 DenseNet-121 ==> ONNX DenseNet

# Inference
Step by step instructions on how to use the pretrained model and link to an example notebook/code. This section should ideally contain:

## Input
Input to network (Example: 224x224 pixels in RGB)

## Preprocessing
Preprocessing required

## Output
Output of network

## Postprocessing
Post processing and meaning of output

# Model Creation

## Dataset (Train and validation)
This section should discuss datasets and any preparation steps if required.

## Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional).

Also clarify in case the model is not trained from scratch and include the source/process used to obtain the ONNX model.

## Validation accuracy
Validation script/notebook used to obtain accuracy reported above along with details of how to use it and reproduce accuracy. Details of experiments leading to accuracy from the reference paper.

## References
Link to paper or references.

# Contributors
Contributors' names

# License
Add license information - on default, Apache 2.0

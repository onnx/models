# How to Contribute to the Model Zoo
To contribute a new model, create a pull request with the following template:
<hr>

# Model Name
Description of model
## Model

 |Model        |ONNX Model  | Model archives(optional)|Accuracy |
|-------------|:--------------|:--------------|:--------------|
|Model-Name       |ONNX Model link with size  |  [ Model archive](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_from_onnx.md) link with size|Accuracy values |
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

### Inference with Model Server (if applicable)
Details of inference with model server using model archives if applicable

<hr>

## Dataset (Train and validation)
This section should discuss datasets and any preparation steps if required.
<hr>

## Validation accuracy
Details of experiments leading to accuracy and comparison with the reference paper.
<hr>

## Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional)
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


## Keyword
Keywords/tags related to models
<hr>

<!-- add link to the pull request/issue page-->
 View an [example submission](./models/image_classification/resnet/README.md)

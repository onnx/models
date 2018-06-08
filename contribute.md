# How to Contribute to the Model Zoo
To contribute a new model, create a pull request with the following template:

# Model Name
Description of model
## Model

 |Model        |ONNX Model  | Model archives(optional)|Accuracy |
|-------------|:--------------|:--------------|:--------------|
|Model-Name       |ONNX Model link with size  |  [ Model archive](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_from_onnx.md) link with size.|Accuracy values |
## Inference
Step by step instructions on how to use the pretrained model and link to an example notebook/code. This section should ideally contain:

### Input to model
Input to network
### Preprocessing steps
Preprocessing required
### Output of model
Output of network
### Postprocessing steps
Post processing and meaning of output
### Inference with Model Server(if applicable)
Details of inference with model server using model archives if applicable
## Dataset(Train and validation)
This section should discuss datasets and any preparation steps if required.

## Validation accuracy
Details of experiments leading to accuracy and comparison with the refernce paper.


## Training
Training details (preprocessing, hyperparameters, resources and environment) alongwith link to a training notebook (optional)
## Validation
Validation script/notebook used to obtain accuracy reported above alongwith details of how to use it and reproduce accuracy.

## References
Link to references

## Contributors
Contributors name

## Keyword
Keywords/tags related to models


<!-- add link to the pull request/issue page-->
 View an [example submission](./models/resnet/README.md)

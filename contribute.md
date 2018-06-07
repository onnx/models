# How to Contribute to the Model Zoo
To contribute a new model, create a pull request with the following information:
<!-- add link to the pull request/issue page-->
* ONNX model link
* Model archive link (optional) - See an [example](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_from_onnx.md) on how to create Model archive for ONNX models.
* Dataset details as well as preparation (if applicable)
* Accuracy of trained models (~1-2% of reference paper)
* Link to references
* List of dependencies to train and test the model
* Step by step instructions on how to use the pretrained model OR link to an example notebook
* Training details (preprocessing, hyperparameters, resources and environment) OR link to a training notebook (optional)
* Step by step instructions on how to validate accuracy of the model alongwith validation datasets OR link to an example validation notebook

See an example submission [here](./models/resnet/README.md)

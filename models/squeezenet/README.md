# SqueezeNet

##Model

|Model        |ONNX Model  | Model archives|
|-------------|:--------------|:--------------|
|SqueezeNet1.0|    [4.8 MB](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.0/squeezenet1.0.onnx)    |  [4.8 MB](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.0/squeezenet1.0.model)     |
|SqueezeNet1.1|    [4.7 MB](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx)    |  [4.7 MB](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.model)     |


## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). 

Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset. 
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook
## Training notebook
Use the notebook [train_squeezenet](train_squeezenet.ipynb) to train the model. It contains the hyperparameters and network details used for training the above models
## Validation accuracy
The accuracies obtained by the models on the validation set are shown in the table below: 

|Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|SqueezeNet1.0|     56.52     |     79.07     |
|SqueezeNet1.1|     56.34     |     79.12     |

Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.
## References
* **SqueezeNet1.0**  
Model from the paper [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
* **SqueezeNet1.1**   
Model from [Official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1). SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.
## Keywords



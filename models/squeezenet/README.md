# Squeezenet
## ONNX model
* [Squeezenet1.0]() ; Size: 5MB
* [Squeezenet1.1]() ; Size: 5MB
## MMS archive
* [Squeezenet1.0]() ; Size: 5MB
* [Squeezenet1.1]() ; Size: 5MB
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). 

Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset. 
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook
## Training notebook
Use the notebook [train_squeezenet](train_squeezenet.ipynb) to train the model. It contains the hyperparameters and network details used for training the above models
## Validation accuracy
The accuracies obtained by the models on the validation set are shown in the table below: 

|Model        |Top-1 error (%)|Top-5 error (%)|
|-------------|:--------------|:--------------|
|Squeezenet1.0|     43.48     |     20.93     |
|Squeezenet1.1|     43.66     |     20.88     |

Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.
## References
* **Squeezenet1.0**  
Model from the paper [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
* **Squeezenet1.1**   
Model from [Official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1). SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.
## Tags


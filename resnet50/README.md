# ResNet-50

Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/resnet50.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/resnet50.tar.gz
- release 1.2: https://s3.amazonaws.com/download.onnx/models/opset_7/resnet50.tar.gz
- release 1.3: https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz

Model size: 103 MB

## Description
ResNet-50 is a deep convolutional networks for classification.

### Paper
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### Dataset
[ILSVRC2015](http://www.image-net.org/challenges/LSVRC/2015/)

### Demo
[Demo implemented by ONNX.js (model file: resnet50 V1.2)](https://microsoft.github.io/onnxjs-demo/#/resnet50)

## Source
Caffe2 ResNet-50 ==> ONNX ResNet-50

## Model input and output
### Preprocessing
* Mean: 128
* Std: 128
* Scale: 256
* Reference: [the training script](https://github.com/pytorch/pytorch/blob/master/caffe2/python/examples/resnet50_trainer.py#L61)
### Input
```
gpu_0/data_0: float[1, 3, 224, 224]
```
### Output
```
gpu_0/softmax_1: float[1, 1000]
```
### Pre-processing steps
### Post-processing steps
### Sample test data
random generated sampe test data:
- test_data_0.npz
- test_data_1.npz
- test_data_2.npz
- test_data_set_0
- test_data_set_1
- test_data_set_2

## Results/accuracy on test set

## License
MIT

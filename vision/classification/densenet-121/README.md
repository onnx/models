# DenseNet-121

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|DenseNet-121| [32 MB](model/densenet-3.onnx)  |  [33 MB](model/densenet-3.tar.gz) |  1.1 | 3|
|DenseNet-121| [32 MB](model/densenet-6.onnx)  |  [33 MB](model/densenet-6.tar.gz) |  1.1.2 | 6|
|DenseNet-121| [32 MB](model/densenet-7.onnx)  |  [33 MB](model/densenet-7.tar.gz) |  1.2 | 7|
|DenseNet-121| [32 MB](model/densenet-8.onnx)  |  [33 MB](model/densenet-8.tar.gz) |  1.3 | 8|
|DenseNet-121| [32 MB](model/densenet-9.onnx)  |  [33 MB](model/densenet-9.tar.gz) |  1.4 | 9|

## Description
DenseNet-121 is a convolutional neural network for classification.

### Paper
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 DenseNet-121 ==> ONNX DenseNet

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
fc6_1: float[1, 1000, 1, 1]
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

# Inception v1

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|Inception-1| [28 MB](model/inception-v1-3.onnx)  |  [31 MB](model/inception-v1-3.tar.gz) |  1.1 | 3|
|Inception-1| [28 MB](model/inception-v1-6.onnx)  |  [31 MB](model/inception-v1-6.tar.gz) |  1.1.2 | 6|
|Inception-1| [28 MB](model/inception-v1-7.onnx)  |  [31 MB](model/inception-v1-7.tar.gz) |  1.2 | 7|
|Inception-1| [28 MB](model/inception-v1-8.onnx)  |  [31 MB](model/inception-v1-8.tar.gz) |  1.3 | 8|
|Inception-1| [28 MB](model/inception-v1-9.onnx)  |  [31 MB](model/inception-v1-9.tar.gz) |  1.4 | 9|

## Description
Inception v1 is a reproduction of GoogLeNet.

### Paper
[Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 Inception v1 ==> ONNX Inception v1

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
prob_1: float[1, 1000]
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

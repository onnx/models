# GoogleNet
Download:
- release 1.1 https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_googlenet.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_googlenet.tar.gz

Model size: 28 MB

## Description
GoogLeNet is the name of a convolutional neural network for classification,
which competed in the ImageNet Large Scale Visual Recognition Challenge in 2014.

Differences:
- not training with the relighting data-augmentation;
- not training with the scale or aspect-ratio data-augmentation;
- uses "xavier" to initialize the weights instead of "gaussian";

### Paper
[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)

### Dataset
[ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/)

## Source
Caffe BVLC GoogLeNet ==> Caffe2 GoogLeNet ==> ONNX GoogLeNet

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
softmaxout_1: float[1, 1000]
```
### Pre-processing steps
### Post-processing steps
### Sample test data
random generated sampe test data:
- test_data_set_0
- test_data_set_1
- test_data_set_2
- test_data_set_3
- test_data_set_4
- test_data_set_5

## Results/accuracy on test set
This bundled model obtains a top-1 accuracy 68.7% (31.3% error) and
a top-5 accuracy 88.9% (11.1% error) on the validation set, using
just the center crop. (Using the average of 10 crops,
(4 + 1 center) * 2 mirror, should obtain a bit higher accuracy.)

## License
[BSD-3](LICENSE)

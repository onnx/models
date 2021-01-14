<!--- SPDX-License-Identifier: Apache-2.0 -->

# ShuffleNet

## Use cases
Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.

## Description
ShuffleNet is a deep convolutional network for image classification. [ShuffleNetV2](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/) is an improved architecture that is the state-of-the-art in terms of speed and accuracy tradeoff used for image classification.

Caffe2 ShuffleNet-v1 ==> ONNX ShuffleNet-v1
PyTorch ShuffleNet-v2 ==> ONNX ShuffleNet-v2

## Model

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|ShuffleNet-v1| [5.3 MB](model/shufflenet-3.onnx)  |  [7 MB](model/shufflenet-3.tar.gz) |  1.1 | 3|
|ShuffleNet-v1| [5.3 MB](model/shufflenet-6.onnx)  |  [9 MB](model/shufflenet-6.tar.gz) |  1.1.2 | 6|
|ShuffleNet-v1| [5.3 MB](model/shufflenet-7.onnx)  |  [9 MB](model/shufflenet-7.tar.gz) |  1.2 | 7|
|ShuffleNet-v1| [5.3 MB](model/shufflenet-8.onnx)  |  [9 MB](model/shufflenet-8.tar.gz) |  1.3 | 8|
|ShuffleNet-v1| [5.3 MB](model/shufflenet-9.onnx)  |  [9 MB](model/shufflenet-9.tar.gz) |  1.4 | 9|

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|Top-1 error |Top-5 error |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ShuffleNet-v2 |[9.2MB](model/shufflenet-v2-10.onnx) |  [8.7MB](model/shufflenet-v2-10.tar.gz) | 1.6 | 10 | 30.64 | 11.68|

## Inference
[This script](ShufflenetV2-export.py) converts the ShuffleNetv2 model from PyTorch to ONNX and uses ONNX Runtime for inference.

### Input to model
Input to the model are 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
```
data_0: float[1, 3, 224, 224]
```

### Preprocessing steps
All pre-trained models expect input images normalized in the same way. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

```python
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
```
Create a mini-batch as expected by the model.
```python
input_batch = input_tensor.unsqueeze(0)
```

### Output of model

Output of this model is tensor of shape 1000, with confidence scores over ImageNet's 1000 classes.
```
softmax_1: float[1, 1000]
```
<hr>

## Dataset (Train and Validation)
Models are pretrained on ImageNet.
For training we use train+valset in COCO except for 5000 images from minivalset, and use the minivalset to test.
Details of performance on COCO object detection are provided in [this paper](https://arxiv.org/pdf/1807.11164v1.pdf)
<hr>

## References
Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design. 2018.

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
<hr>

## Contributors
Ksenija Stanojevic
<hr>

## License
BSD 3-Clause License
<hr>

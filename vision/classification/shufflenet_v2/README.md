# ShuffleNetV2

## Use cases
Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.

## Description
[ShuffleNetV2](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/) is an architecture that is the state-of-the-art in terms of speed and accuracy tradeoff used for image classification.

## Model

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|Top-1 error |Top-5 error |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ShuffleNetV2 |[9.2MB](model/shufflenet-v2-10.onnx) |  [8.7MB](model/shufflenet-v2-10.tar.gz) | 1.6 | 10 | 30.64 | 11.68|

## Inference
[This script](https://github.com/onnx/models/blob/master/vision/classification/shufflenet_v2/ShufflenetV2-export.py) converts the model from PyTorch to ONNX and uses ONNX Runtime for inference.

### Input to model
Input to the model are 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.

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

Output of this model is tensor of shape 1000, with confidence scores over Imagenet's 1000 classes.

<hr>

## Dataset (Train and Validation)
Models are pretrained on ImageNet.
For training we use train+valset in COCO except for 5000 images from minivalset, and use the minivalset to test.
Details of performance on COCO object detection are provided in [this paper](https://arxiv.org/pdf/1807.11164v1.pdf)
<hr>

## References
Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design. 2018.
<hr>

## Contributors
Ksenija Stanojevic
<hr>

## License
BSD 3-Clause License
<hr>

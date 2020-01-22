# ShuffleNetV2

## Use-cases
Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.

## Description
[ShuffleNetV2](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/) is an architecture that is the state-of-the-art in terms of speed and accuracy tradeoff.

## Model

|Model        |Download  |Checksum| Download (with sample test data)|ONNX version|Opset version|Top-1 error |Top-5 error |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ShuffleNetV2 |[5.28MB](https://github.com/onnx/models/blob/master/text/machine_comprehension/shufflenet_v2/model/model.onnx) | [MD5](https://github.com/onnx/models/blob/master/text/machine_comprehension/shufflenet_v2/model/shufflenetv2-md5.txt) | [5.13MB](https://github.com/onnx/models/blob/master/text/machine_comprehension/shufflenet_v2/model/model.zip) | 1.6 | 10 | 30.64 | 11.68| 

## Inference
The script for ONNX model conversion and ONNXRuntime inference is [here]

### Input to model
Input to a model are 3-channel RGB images of shape (3 x H x W).

### Preprocessing steps
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

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
Crete a mini-batch as expected by the model
```python
input_batch = input_tensor.unsqueeze(0)
```
<hr>

## Dataset (Train and Validation)
Models are pretrainedon ImageNet.
For training we usetrain+valset in COCO except for 5000 images from minivalset, and use the minivalset to test.
<hr>

## Validation accuracy
Details of performance on COCO object detection are provided [here](https://arxiv.org/pdf/1807.11164v1.pdf)
<hr>

## Publication/Attribution
Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design. 2018.
<hr>

## Contributors
Ksenija Stanojevic
<hr>

## Licence
Apache 2.0 License
<hr>

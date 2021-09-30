<!--- SPDX-License-Identifier: BSD-3-Clause -->

# ShuffleNet

## Use cases
Computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power.

## Description
ShuffleNet is a deep convolutional network for image classification. [ShuffleNetV2](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/) is an improved architecture that is the state-of-the-art in terms of speed and accuracy tradeoff used for image classification.

Caffe2 ShuffleNet-v1 ==> ONNX ShuffleNet-v1

PyTorch ShuffleNet-v2 ==> ONNX ShuffleNet-v2

ONNX ShuffleNet-v2 ==> Quantized ONNX ShuffleNet-v2

ONNX ShuffleNet-v2 ==> Quantized ONNX ShuffleNet-v2

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
|ShuffleNet-v2-fp32 |[8.79MB](model/shufflenet-v2-12.onnx) |[8.69MB](model/shufflenet-v2-12.tar.gz) |1.9 |12 |33.65 |13.43|
|ShuffleNet-v2-int8 |[2.28MB](model/shufflenet-v2-12-int8.onnx) |[2.37MB](model/shufflenet-v2-10-int8.tar.gz) |1.9 |12 |33.85 |13.66 |
> Compared with the fp32 ShuffleNet-v2, int8 ShuffleNet-v2's Top-1 error rising ratio is 0.59%, Top-5 error rising ratio is 1.71% and performance improvement is 1.62x.
>
> Note the performance depends on the test hardware. 
> 
> Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

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

## Quantization
ShuffleNet-v2-int8 is obtained by quantizing ShuffleNet-v2-fp32 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/onnx_model_zoo/shufflenet/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.7.0 
onnxruntime: 1.6.0+

### Prepare model
```shell
wget https://github.com/onnx/models/tree/master/vision/classification/shufflenet/model/shufflenet-v2-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=shufflenetv2.yaml \
                   --output_model=path/to/save
```

### Model inference
We use onnxruntime to perform Resnet50_fp32 and Resnet50_int8 inference. View the notebook [onnxrt_inference](../onnxrt_inference.ipynb) to understand how to use these 2 models for doing inference as well as which preprocess and postprocess we use.

## References
* Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design. 2018.

* huffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
<hr>

## Contributors
* Ksenija Stanojevic
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)
<hr>

## License
BSD 3-Clause License
<hr>

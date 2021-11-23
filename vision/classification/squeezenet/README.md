<!--- SPDX-License-Identifier: Apache-2.0 -->

# SqueezeNet

## Use cases
SqueezeNet models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. SqueezeNet models are highly efficient in terms of size and speed while providing good accuracies. This makes them ideal for platforms with strict constraints on size.

## Description
SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with 50x fewer parameters. SqueezeNet requires less communication across servers during distributed training, less bandwidth to export a new model from the cloud to an autonomous car and more feasible to deploy on FPGAs and other hardware with limited memory.

## Model
Squeezenet 1.0 gives AlexNet level of accuracy with 50X fewer parameters. [Run SqueezeNet 1.0 in browser](https://microsoft.github.io/onnxjs-demo/#/squeezenet) - implemented by ONNX.js.
SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|SqueezeNet 1.1|    [9 MB](model/squeezenet1.1-7.onnx) | [6 MB](model/squeezenet1.1-7.tar.gz) |1.2.1  |7 |56.34     |     79.12     |
|SqueezeNet 1.0| [5 MB](model/squeezenet1.0-3.onnx)  |  [6 MB](model/squeezenet1.0-3.tar.gz) |  1.1 | 3|
|SqueezeNet 1.0| [5 MB](model/squeezenet1.0-6.onnx)  |  [6 MB](model/squeezenet1.0-6.tar.gz) |  1.1.2 | 6|
|SqueezeNet 1.0| [5 MB](model/squeezenet1.0-7.onnx)  |  [11 MB](model/squeezenet1.0-7.tar.gz) |  1.2 | 7|
|SqueezeNet 1.0| [5 MB](model/squeezenet1.0-8.onnx)  |  [11 MB](model/squeezenet1.0-8.tar.gz) |  1.3 | 8|
|SqueezeNet 1.0| [5 MB](model/squeezenet1.0-9.onnx)  |  [11 MB](model/squeezenet1.0-9.tar.gz) |  1.4 | 9|
|SqueezeNet 1.0| [233 MB](model/squeezenet1.0-12.onnx)  |  [216 MB](model/squeezenet1.0-12.tar.gz) |  1.9 | 12|56.85|79.87|
|SqueezeNet 1.0-int8| [58 MB](model/squeezenet1.0-12-int8.onnx)  |  [39 MB](model/squeezenet1.0-12-int8.tar.gz) |  1.9 | 12|56.48|79.76|
> Compared with the fp32 SqueezeNet 1.0, int8 SqueezeNet 1.0's Top-1 accuracy drop ratio is 0.65%, Top-5 accuracy drop ratio is 0.14% and performance improvement is 1.31x.
>
> **Note** 
>
> The performance depends on the test hardware.
> 
> Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Inference
We used MXNet as framework with gluon APIs to perform inference for SqueezeNet 1.1. View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook.

SqueezeNet 1.0 is converted from Caffe2 -> ONNX.

### Input
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size, and H and W are expected to be at least 224. The inference was done using a jpeg image.

``float[1, 3, 224, 224]``

### Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing. Check [imagenet_preprocess.py](../imagenet_preprocess.py) for code.

### Output
The model outputs image scores for each of the [1000 classes of ImageNet](../synset.txt).

``softmaxout_1: float[1, 1000, 1, 1]``

### Postprocessing
The post-processing involves calculating the softmax probablility scores for each class and sorting them to report the most probable classes. Check [imagenet_postprocess.py](../imagenet_postprocess.py) for code.

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#squeezenet_v1.1_onnx).

## Dataset
Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.

## Validation accuracy
The accuracies obtained by the model on the validation set are mentioned above. The accuracies have been calculated on center cropped images with a maximum deviation of 1.2% (top-5 accuracy) from the paper.

## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_squeezenet.ipynb) to understand details for parameters and network for each of the above variants of SqueezeNet.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.

## Quantization
SqueezeNet 1.0-int8 is obtained by quantizing fp32 SqueezeNet 1.0 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/examples/onnxrt/onnx_model_zoo/squeezenet/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/blob/master/vision/classification/squeezenet/model/squeezenet1.0-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=squeezenet.yaml \
                   --data_path=/path/to/imagenet \
                   --label_path=/path/to/imagenet/label \
                   --output_model=path/to/save
```

## References
* **SqueezeNet1.1**
SqueezeNet1.1 presented in the [Official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) is an improved version of SqueezeNet1.0 from the paper [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

* [MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

## Contributors
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
Apache 2.0

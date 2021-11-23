<!--- SPDX-License-Identifier: BSD-3-Clause -->

# AlexNet

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|AlexNet| [238 MB](model/bvlcalexnet-3.onnx)  |  [225 MB](model/bvlcalexnet-3.tar.gz) |  1.1 | 3| | |
|AlexNet| [238 MB](model/bvlcalexnet-6.onnx)  |  [225 MB](model/bvlcalexnet-6.tar.gz) |  1.1.2 | 6| | |
|AlexNet| [238 MB](model/bvlcalexnet-7.onnx)  |  [226 MB](model/bvlcalexnet-7.tar.gz) |  1.2 | 7| | |
|AlexNet| [238 MB](model/bvlcalexnet-8.onnx)  |  [226 MB](model/bvlcalexnet-8.tar.gz) |  1.3 | 8| | |
|AlexNet| [238 MB](model/bvlcalexnet-9.onnx)  |  [226 MB](model/bvlcalexnet-9.tar.gz) |  1.4 | 9| | |
|AlexNet| [233 MB](model/bvlcalexnet-12.onnx)  |  [216 MB](model/bvlcalexnet-12.tar.gz) |  1.9 | 12|54.80|78.23|
|AlexNet-int8| [58 MB](model/bvlcalexnet-12-int8.onnx)  |  [39 MB](model/bvlcalexnet-12-int8.tar.gz) |  1.9 | 12|54.68|78.23|
> Compared with the fp32 AlextNet, int8 AlextNet's Top-1 accuracy drop ratio is 0.22%, Top-5 accuracy drop ratio is 0.05% and performance improvement is 2.26x.
>
> **Note** 
>
> Different preprocess methods will lead to different accuracies, the accuracy in table depends on this specific [preprocess method](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/onnx_model_zoo/alexnet/main.py).
> 
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Description
AlexNet is the name of a convolutional neural network for classification,
which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.

Differences:
- not training with the relighting data-augmentation;
- initializing non-zero biases to 0.1 instead of 1 (found necessary for training, as initialization to 1 gave flat loss).

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe BVLC AlexNet ==> Caffe2 AlexNet ==> ONNX AlexNet

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
Randomly generated sample test data:
- test_data_0.npz
- test_data_1.npz
- test_data_2.npz
- test_data_set_0
- test_data_set_1
- test_data_set_2

## Results/accuracy on test set
The bundled model is the iteration 360,000 snapshot.
The best validation performance during training was iteration
358,000 with validation accuracy 57.258% and loss 1.83948.
This model obtains a top-1 accuracy 57.1% and a top-5 accuracy
80.2% on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror,
should obtain a bit higher accuracy.)

## Quantization
AlexNet-int8 is obtained by quantizing fp32 AlexNet model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/onnx_model_zoo/alexnet/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/blob/master/vision/classification/alexnet/model/bvlcalexnet-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=alexnet.yaml \
                   --data_path=/path/to/imagenet \
                   --label_path=/path/to/imagenet/label \
                   --output_model=path/to/save
```

## References
* [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
[BSD-3](LICENSE)

<!--- SPDX-License-Identifier: BSD-3-Clause -->

# CaffeNet

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- | ------------- |
|CaffeNet| [238 MB](model/caffenet-3.onnx)  |  [244 MB](model/caffenet-3.tar.gz) |  1.1 | 3| | |
|CaffeNet| [238 MB](model/caffenet-6.onnx)  |  [244 MB](model/caffenet-6.tar.gz) |  1.1.2 | 6| | |
|CaffeNet| [238 MB](model/caffenet-7.onnx)  |  [244 MB](model/caffenet-7.tar.gz) |  1.2 | 7| | |
|CaffeNet| [238 MB](model/caffenet-8.onnx)  |  [244 MB](model/caffenet-8.tar.gz) |  1.3 | 8| | |
|CaffeNet| [238 MB](model/caffenet-9.onnx)  |  [244 MB](model/caffenet-9.tar.gz) |  1.4 | 9| | |
|CaffeNet| [233 MB](model/caffenet-12.onnx)  |  [216 MB](model/caffenet-12.tar.gz) |  1.9 | 12|56.27 |79.52 |
|CaffeNet-int8| [58 MB](model/caffenet-12-int8.onnx)  |  [39 MB](model/caffenet-12-int8.tar.gz) |  1.9 | 12| 56.22|79.52 |
> Compared with the fp32 CaffeNet, int8 CaffeNet's Top-1 accuracy drop ratio is 0.09%, Top-5 accuracy drop ratio is 0.13% and performance improvement is 3.08x.
>
> **Note** 
>
> Different preprocess methods will lead to different accuracies, the accuracy in table depends on this specific [preprocess method](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/onnx_model_zoo/caffenet/main.py).
> 
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Description
CaffeNet a variant of AlexNet.
AlexNet is the name of a convolutional neural network for classification,
which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.

Differences:
- not training with the relighting data-augmentation;
- the order of pooling and normalization layers is switched (in CaffeNet, pooling is done before normalization).

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe BVLC CaffeNet ==> Caffe2 CaffeNet ==> ONNX CaffeNet

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
- test_data_set_0
- test_data_set_1
- test_data_set_2
- test_data_set_3
- test_data_set_4
- test_data_set_5

## Results/accuracy on test set
This model is snapshot of iteration 310,000.
The best validation performance during training was iteration
313,000 with validation accuracy 57.412% and loss 1.82328.
This model obtains a top-1 accuracy 57.4% and a top-5 accuracy
80.4% on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror,
should obtain a bit higher accuracy still.)

## Quantization
CaffeNet-int8 is obtained by quantizing fp32 CaffeNet model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/onnx_model_zoo/caffenet/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/blob/master/vision/classification/caffenet/model/caffenet-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=caffenet.yaml \
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

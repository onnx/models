<!--- SPDX-License-Identifier: MIT -->

# ZFNet-512

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|ZFNet-512| [341 MB](model/zfnet512-3.onnx)  |  [320 MB](model/zfnet512-3.tar.gz) |  1.1 | 3| | |
|ZFNet-512| [341 MB](model/zfnet512-6.onnx)  |  [320 MB](model/zfnet512-6.tar.gz) |  1.1.2 | 6| | |
|ZFNet-512| [341 MB](model/zfnet512-7.onnx)  |  [320 MB](model/zfnet512-7.tar.gz) |  1.2 | 7| | |
|ZFNet-512| [341 MB](model/zfnet512-8.onnx)  |  [318 MB](model/zfnet512-8.tar.gz) |  1.3 | 8| | |
|ZFNet-512| [341 MB](model/zfnet512-9.onnx)  |  [318 MB](model/zfnet512-9.tar.gz) |  1.4 | 9| | |
|ZFNet-512| [333 MB](model/zfnet512-12.onnx)  |  [309 MB](model/zfnet512-12.tar.gz) |  1.9 | 12|55.97|79.41|
|ZFNet-512-int8| [83 MB](model/zfnet512-12-int8.onnx)  |  [48 MB](model/zfnet512-12-int8.tar.gz) |  1.9 | 12|55.84|79.33|
> Compared with the fp32 ZFNet-512, int8 ZFNet-512's Top-1 accuracy drop ratio is 0.23%, Top-5 accuracy drop ratio is 0.10% and performance improvement is 1.78x.
>
> **Note** 
>
> Different preprocess methods will lead to different accuracies, the accuracy in table depends on this specific [preprocess method](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/examples/onnxrt/onnx_model_zoo/zfnet/main.py).
> 
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Description
ZFNet-512 is a deep convolutional networks for classification.
This model's 4th layer has 512 maps instead of 1024 maps mentioned in the paper.

### Dataset
[ILSVRC2013](http://www.image-net.org/challenges/LSVRC/2013/)

## Source
Caffe2 ZFNet-512 ==> ONNX ZFNet-512

## Model input and output
### Input
```
gpu_0/data_0: float[1, 3, 224, 224]
```
### Output
```
gpu_0/softmax_1: float[1, 1000]
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

## Quantization
ZFNet-512-int8 is obtained by quantizing fp32 ZFNet-512 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/image_recognition/onnx_model_zoo/zfnet/quantization/ptq/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/blob/main/vision/classification/zfnet-512/model/zfnet512-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=zfnet512.yaml \
                   --data_path=/path/to/imagenet \
                   --label_path=/path/to/imagenet/label \
                   --output_model=path/to/save
```

## References
* [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
MIT

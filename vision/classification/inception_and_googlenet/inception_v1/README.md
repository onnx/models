<!--- SPDX-License-Identifier: MIT -->

# Inception v1

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version| Top-1 accuracy (%)|
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
|Inception-1| [28 MB](model/inception-v1-3.onnx)  |  [29 MB](model/inception-v1-3.tar.gz) |  1.1 | 3| |
|Inception-1| [28 MB](model/inception-v1-6.onnx)  |  [29 MB](model/inception-v1-6.tar.gz) |  1.1.2 | 6| |
|Inception-1| [28 MB](model/inception-v1-7.onnx)  |  [29 MB](model/inception-v1-7.tar.gz) |  1.2 | 7| |
|Inception-1| [28 MB](model/inception-v1-8.onnx)  |  [29 MB](model/inception-v1-8.tar.gz) |  1.3 | 8| |
|Inception-1| [28 MB](model/inception-v1-9.onnx)  |  [29 MB](model/inception-v1-9.tar.gz) |  1.4 | 9| |
|Inception-1| [27 MB](model/inception-v1-12.onnx)  |  [25 MB](model/inception-v1-12.tar.gz) |  1.9 | 12| 67.23|
|Inception-1-int8| [10 MB](model/inception-v1-12-int8.onnx)  |  [9 MB](model/inception-v1-12-int8.tar.gz) |  1.9 | 12| 67.24|
> Compared with the fp32 Inception-1, int8 Inception-1's Top-1 accuracy drop ratio is -0.01% and performance improvement is 1.26x.
>
> **Note** 
> 
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.


## Description
Inception v1 is a reproduction of GoogLeNet.

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 Inception v1 ==> ONNX Inception v1
ONNX Inception v1 ==> Quantized ONNX Inception v1

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

## Quantization
Inception-1-int8 is obtained by quantizing fp32 Inception-1 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/image_recognition/onnx_model_zoo/inception/quantization/ptq/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/blob/main/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=inception_v1.yaml \
                   --data_path=/path/to/imagenet \
                   --label_path=/path/to/imagenet/label \
                   --output_model=path/to/save
```

## References
* [Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)


## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
MIT

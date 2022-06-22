<!--- SPDX-License-Identifier: MIT -->

# DenseNet-121

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|DenseNet-121| [32 MB](model/densenet-3.onnx)  |  [33 MB](model/densenet-3.tar.gz) |  1.1 | 3| |
|DenseNet-121| [32 MB](model/densenet-6.onnx)  |  [33 MB](model/densenet-6.tar.gz) |  1.1.2 | 6| |
|DenseNet-121| [32 MB](model/densenet-7.onnx)  |  [33 MB](model/densenet-7.tar.gz) |  1.2 | 7| |
|DenseNet-121| [32 MB](model/densenet-8.onnx)  |  [33 MB](model/densenet-8.tar.gz) |  1.3 | 8| |
|DenseNet-121| [32 MB](model/densenet-9.onnx)  |  [33 MB](model/densenet-9.tar.gz) |  1.4 | 9| |
|DenseNet-121-12| [32 MB](model/densenet-12.onnx)  |  [30 MB](model/densenet-12.tar.gz) |  1.9 | 12| 60.96 |
|DenseNet-121-12-int8| [9 MB](model/densenet-12-int8.onnx)  |  [6 MB](model/densenet-12-int8.tar.gz) |  1.9 | 12| 60.20 |
> Compared with the DenseNet-121-12, DenseNet-121-12-int8's op-1 accuracy drop ratio is 1.25% and performance improvement is 1.18x.
>
> Note the performance depends on the test hardware. 
> 
> Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Description
DenseNet-121 is a convolutional neural network for classification.

### Paper
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 DenseNet-121 ==> ONNX DenseNet

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
fc6_1: float[1, 1000, 1, 1]
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
Mask R-CNN R-50-FPN-int8 is obtained by quantizing Mask R-CNN R-50-FPN-fp32 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/image_recognition/onnx_model_zoo/densenet/quantization/ptq/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.10.0

### Prepare model
```shell
wget https://github.com/onnx/models/raw/main/vision/classification/densenet-121/model/densenet-12.onnx
```

### Model quantize
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=densenet.yaml \
                   --output_model=path/to/save
```

## References
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
MIT

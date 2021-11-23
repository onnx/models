<!--- SPDX-License-Identifier: BSD-3-Clause -->

# GoogleNet

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|GoogleNet| [28 MB](model/googlenet-3.onnx)  |  [31 MB](model/googlenet-3.tar.gz) |  1.1 | 3| | |
|GoogleNet| [28 MB](model/googlenet-6.onnx)  |  [31 MB](model/googlenet-6.tar.gz) |  1.1.2 | 6| | |
|GoogleNet| [28 MB](model/googlenet-7.onnx)  |  [31 MB](model/googlenet-7.tar.gz) |  1.2 | 7| | |
|GoogleNet| [28 MB](model/googlenet-8.onnx)  |  [31 MB](model/googlenet-8.tar.gz) |  1.3 | 8| | |
|GoogleNet| [28 MB](model/googlenet-9.onnx)  |  [31 MB](model/googlenet-9.tar.gz) |  1.4 | 9| | |
|GoogleNet| [27 MB](model/googlenet-12.onnx)  |  [25 MB](model/googlenet-12.tar.gz) |  1.9 | 12|67.78|88.34|
|GoogleNet-int8| [7 MB](model/googlenet-12-int8.onnx)  |  [5 MB](model/googlenet-12-int8.tar.gz) |  1.9 | 12|67.73|88.32|
> Compared with the fp32 GoogleNet, int8 GoogleNet's Top-1 accuracy drop ratio is 0.07%, Top-5 accuracy drop ratio is 0.02% and performance improvement is 1.27x.
>
> **Note** 
>
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Description
GoogLeNet is the name of a convolutional neural network for classification,
which competed in the ImageNet Large Scale Visual Recognition Challenge in 2014.

Differences:
- not training with the relighting data-augmentation;
- not training with the scale or aspect-ratio data-augmentation;
- uses "xavier" to initialize the weights instead of "gaussian";

### Dataset
[ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/)

## Source
Caffe BVLC GoogLeNet ==> Caffe2 GoogLeNet ==> ONNX GoogLeNet

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
prob_0: float[1, 1000]
```
### Pre-processing steps
#### Necessary Imports
```python
import imageio
from PIL import Image
```
#### Obtain and pre-process image

```python
def get_image(path):
'''
    Using path to image, return the RGB load image
'''
    img = imageio.imread(path, pilmode='RGB')
    return img

# Pre-processing function for ImageNet models using numpy
def preprocess(img):
    '''
    Preprocessing required on the images for inference with mxnet gluon
    The function takes loaded image and returns processed tensor
    '''
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img
```

### Post-processing steps
```python
def predict(path):
    # based on : https://mxnet.apache.org/versions/1.0.0/tutorials/python/predict_image.html
    img = get_image(path)
    img = preprocess(img)
    mod.forward(Batch([mx.nd.array(img)]))
    # Take softmax to generate probabilities
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    return a
```
### Sample test data
random generated sample test data:
- test_data_set_0
- test_data_set_1
- test_data_set_2
- test_data_set_3
- test_data_set_4
- test_data_set_5

## Results/accuracy on test set
This bundled model obtains a top-1 accuracy 68.7% (31.3% error) and
a top-5 accuracy 88.9% (11.1% error) on the validation set, using
just the center crop. (Using the average of 10 crops,
(4 + 1 center) * 2 mirror, should obtain a bit higher accuracy.)

## Quantization
GoogleNet-int8 is obtained by quantizing fp32 GoogleNet model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot/blob/master/examples/onnxrt/onnx_model_zoo/googlenet/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/blob/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=googlenet.yaml \
                   --data_path=/path/to/imagenet \
                   --label_path=/path/to/imagenet/label \
                   --output_model=path/to/save
```

## References
* [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
[BSD-3](LICENSE)

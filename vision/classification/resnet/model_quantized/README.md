<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNet

## Use cases
ResNet models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. ResNet models provide very high accuracies with affordable model sizes. They are ideal for cases when high accuracy of classification is required.

## Description
Deeper neural networks are more difficult to train. Residual learning framework ease the training of networks that are substantially deeper. The research explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. It also provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset the residual networks are evaluated with a depth of up to 152 layers - 8X deeper than VGG nets but still having lower complexity.

ONNX ResNet50-v1 ==> Quantized ONNX ResNet50-v1

## Model

The model below is ResNet50-v1. ResNet50_int8 is obtained by quantizing ResNet50_fp32 model. Note the latency depends on the test hardware and the displayed results are tested by Intel® Xeon® Platinum 8280 Processor. 

Compared with the fp32 model, we get an accuracy drop of 0.2% and a performance improvement of 1.9X after quantization.

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|latency (ms)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ResNet50_fp32|[97.8 MB](model/resnet50-v1-12.onnx)|[92.0 MB](model/resnet50-v1-12.tar.gz)|1.7.0|12|74.97|8.32|
|ResNet50_int8|[24.6 MB](model/resnet50-v1-12-int8.onnx)|[22.3 MB](model/resnet50-v1-12-int8.tar.gz)|1.7.0|12|74.77|4.30|

## Inference
We use onnxruntime to perform inference. View the notebook [onnxrt_inference](../../onnxrt_inference.ipynb) to understand how to use above models for doing inference.

### Input
The fp32 and int8 models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size expected to be 1, and H and W are expected to be 224.

### Preprocessing
```python
def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img
```

### Output
The model outputs image scores for each of the [1000 classes of ImageNet](../../synset.txt).

### Postprocessing
The post-processing sort the predictions to report the most probable classes.

```python
preds = np.squeeze(preds)
a = np.argsort(preds)[::-1]
print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))
```

### Dataset
Dataset used for validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../../imagenet_prep.md) for guidelines on preparing the dataset.

## Quantization
We use [Intel® Low Precision Optimization Tool (LPOT)](https://github.com/intel/lpot) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/lpot/tree/master/examples/onnxrt/onnx_model_zoo/resnet50/README.md) to understand how to use LPOT for quantization.

### Environment
onnx: 1.7.0 
onnxruntime: 1.6.0+

### Prepare model
```shell
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model_quantized/model/resnet50-v1-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=resnet50_v1_5.yaml \
                   --output_model=path/to/save
```

### References
* **ResNetv1**
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
 He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

* Intel® Low Precision Optimization Tool (LPOT): https://github.com/intel/lpot

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
Apache 2.0

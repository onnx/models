# ResNet

## Use cases
ResNet models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. ResNet models provide very high accuracies with affordable model sizes. They are ideal for cases when high accuracy of classification is required.

## Description
Deeper neural networks are more difficult to train. Residual learning framework ease the training of networks that are substantially deeper. The research explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. It also provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset the residual nets were evaluated with a depth of up to 152 layers — 8× deeper than VGG nets but still having lower complexity.

## Model

The model below are ResNet v1 and v2. ResNet models consists of residual blocks and came up to counter the effect of deteriorating accuracies with more layers due to network not learning the initial layers.
ResNet v2 uses pre-activation function whereas ResNet v1  uses post-activation for the residual blocks. The models below have 18, 34, 50, 101 and 152 layers for with ResNetv1 and ResNetv2 architecture.

* Version 1:

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ResNet-18|    [44.7 MB](model/resnet18-v1.onnx)    |[42.9 MB](model/resnet18-v1.tar.gz)    |  1.2.1  |7| 69.93         |    89.29|         
|ResNet-34|    [83.3 MB](model/resnet34-v1.onnx)    | [78.6 MB](model/resnet34-v1.tar.gz)    |  1.2.1   |7|73.73         |     91.40           |
|ResNet-50|    [97.8 MB](model/resnet50-v1.onnx)    |[92.2 MB](model/resnet50-v1.tar.gz)    |1.2.1    |7|74.93         |     92.38           |
|ResNet-101|    [170.6 MB](model/resnet101-v1.onnx)   | [159.8 MB](model/resnet101-v1.tar.gz)    |  1.2.1  |7  | 76.48         |     93.20           |
|ResNet-152|    [230.6 MB](model/resnet152-v1.onnx)    |[217.2 MB](model/resnet152-v1.tar.gz)    | 1.2.1  |7 |77.11         |     93.61           |


* Version 2:

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ResNet-18|    [44.6 MB](model/resnet18-v2.onnx)    | [42.9 MB](model/resnet18-v2.tar.gz)    | 1.2.1  |7 |    69.70         |     89.49          |
|ResNet-34|    [83.2 MB](model/resnet34-v2.onnx)    |[78.6 MB](model/resnet34-v2.tar.gz)    |  1.2.1   |7| 73.36         |     91.43           |
|ResNet-50|    [97.7 MB](model/resnet50-v2.onnx)   |[92.0 MB](model/resnet50-v2.tar.gz)    | 1.2.1 |7|75.81         |     92.82           |
|ResNet-101|    [170.4 MB](model/resnet101-v2.onnx)    |[159.4 MB](model/resnet101-v2.tar.gz)    |  1.2.1  |7 | 77.42         |     93.61           |
|ResNet-152|    [230.3 MB](model/resnet152-v2.onnx)    |[216.0 MB](model/resnet152-v2.tar.gz)    | 1.2.1   |7 | 78.20         |     94.21           |


## Inference
We used MXNet as framework with gluon APIs to perform inference. View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook.

### Input
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size, and H and W are expected to be at least 224.
The inference was done using jpeg image.

### Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing.

The following code shows how to preprocess a NCHW tensor:

```python
import numpy

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
```

Check [imagenet_preprocess.py](../imagenet_preprocess.py) for additional sample code.

### Output
The model outputs image scores for each of the [1000 classes of ImageNet](../synset.txt).

### Postprocessing
The post-processing involves calculating the softmax probablility scores for each class. You can also sort them to report the most probable classes. Check [imagenet_postprocess.py](../imagenet_postprocess.py) for code.

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#resnet_header).

## Dataset
Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.


## Validation accuracy
The accuracies obtained by the models on the validation set are mentioned above. The validation was done using center cropping of images unlike the paper which uses ten-cropping. We expect an increase of 1-2% in accuracies using ten cropping and that would lead to accuracies similar to the paper.

## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_resnet.ipynb) to understand details for parameters and network for each of the above variants of ResNet.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.

## References
* **ResNet-v1**
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
 He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

* **ResNet-v2**
[Identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.

* [MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

## Contributors
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)

## License
Apache 2.0

# ResNet
Deeper neural networks are more difficult to train. Residual learning framework ease the training of networks that are substantially deeper. The research explicitly reformulate the layers as learning residual functions with reference to the layer inputs, in- stead of learning unreferenced functions. It also provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset the residual nets were evaluated with a depth of up to 152 layers—8× deeper than VGG nets but still having lower complexity.

## Keyword
CNN, ResNet, ONNX, ImageNet, Computer Vision 

## Model
* Version 1:

 |Model        |ONNX Model  | Model archives|
|-------------|:--------------|:--------------|
|ResNet-18|    [44.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx)    |  [44.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.model)     |
|ResNet-34|    [83.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.onnx)    |  [83.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.model)     |
|ResNet-50|    [97.8 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.onnx)    |  [97.8 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.model)     |
|ResNet-101|    [170.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.onnx)    |  [170.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.model)     |
|ResNet-152|    [230.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.onnx)    |  [230.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.model)     |


* Version 2:

 |Model        |ONNX Model  | Model archives|
|-------------|:--------------|:--------------|
|ResNet-18|    [44.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx)    |  [44.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.model)     |
|ResNet-34|    [83.2 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.onnx)    |  [83.2 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.model)     |
|ResNet-50|    [97.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx)    |  [97.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.model)     |
|ResNet-101|    [170.4 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx)    |  [170.4 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.model)     |
|ResNet-152|    [230.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx)    |  [230.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.model)     |





## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.

## Example Notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook.

## Training Notebook
View the notebook that can be used for training [here](train_resnet.ipynb). The notebook contains details for parameters and network for each of the above variants of ResNet.

## Verify Validation Accuracy
Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.

## Validation accuracy
The accuracies obtained by the models on the validation set are shown in the table below: 
* Version 1:

 |Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|ResNet-18|     69.93         |     89.29           |
|ResNet-34|     73.73         |     91.40           |
|ResNet-50|     74.93         |     92.38           |
|ResNet-101|    76.48         |     93.20           |
|ResNet-152|    77.11         |     93.61           |

* Version 2:

 |Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|ResNet-18|     69.70         |     89.49          |
|ResNet-34|     73.36         |     91.43           |
|ResNet-50|     75.81         |     92.82           |
|ResNet-101|    77.42         |     93.61           |
|ResNet-152|    78.20         |     94.21           |

## References
**ResNet-v1**
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
 He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

**ResNet-v2**
[Identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.


## Keywords

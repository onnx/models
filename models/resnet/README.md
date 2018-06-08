# ResNet
Deeper neural networks are more difficult to train. Residual learning framework ease the training of networks that are substantially deeper. The research explicitly reformulate the layers as learning residual functions with reference to the layer inputs, in- stead of learning unreferenced functions. It also provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset the residual nets were evaluated with a depth of up to 152 layers—8× deeper than VGG nets but still having lower complexity.


## Model
* Version 1:

 |Model        |ONNX Model  | Model archives|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|
|ResNet-18|    [44.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx)    |  [44.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.model)     | 69.93         |     89.29           |
|ResNet-34|    [83.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.onnx)    |  [83.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.model)     |73.73         |     91.40           |
|ResNet-50|    [97.8 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.onnx)    |  [97.8 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.model)     |74.93         |     92.38           |
|ResNet-101|    [170.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.onnx)    |  [170.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.model)     | 76.48         |     93.20           |
|ResNet-152|    [230.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.onnx)    |  [230.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.model)     |77.11         |     93.61           |


* Version 2:

 |Model        |ONNX Model  | Model archives|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|
|ResNet-18|    [44.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.onnx)    |  [44.6 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.model)     |    69.70         |     89.49          |
|ResNet-34|    [83.2 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.onnx)    |  [83.2 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.model)     | 73.36         |     91.43           |
|ResNet-50|    [97.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx)    |  [97.7 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.model)     |75.81         |     92.82           |
|ResNet-101|    [170.4 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx)    |  [170.4 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.model)     | 77.42         |     93.61           |
|ResNet-152|    [230.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx)    |  [230.3 MB](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.model)     | 78.20         |     94.21           |


## Inference
We used MXNet as framework with gluon APIs to perform inference. View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook. 
### Input 
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size, and H and W are expected to be at least 224. 
### Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing.
```bash
def preprocess(img):   
    '''
    Preprocessing required on the images for inference with mxnet gluon
    ''''''
    
    transform_fn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0) # batchify
    
    return img
    
 ```
 

### Output
The model outputs image scores for each of the [1000 classes of ImageNet](../../synset.txt). 

### Postprocessing
The post-processing involves calculating the softmax probablility scores for each classes and sorting them to report the most probable 
classes

```bash
def postprocess(scores): 
    '''
    Postprocessing with mxnet gluon
    ''''''
    prob = mx.ndarray.softmax(scores).asnumpy()
    # print the top-5 inferences class
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    return a
    
 ```
### Inference with Model Server
Head on to [Quick start section of model server](https://github.com/awslabs/mxnet-model-server/blob/master/README.md#quick-start) for serving your models. 
* **Start Server**:
```bash
mxnet-model-server --models resnet18v1=https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.model
```

* **Run Prediction**:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/resnet18v1/predict -F "data=@kitten.jpeg"
```
Use the dataname as 'data' in predict call for all the above ResNet models

## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.


## Validation accuracy
The accuracies obtained by the models on the validation set are mentioned above. The validation was done using center cropping of images unlike
the paper using ten-cropping. Even with center crop, the accuracies are within 1-2% of accuracy reported by the paper.
 
<!-- * Version 1:

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

-->

## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_resnet.ipynb) to understand details for parameters and network for each of the above variants of ResNet.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.



## References
**ResNet-v1**
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
 He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

**ResNet-v2**
[Identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.

## Contributors

## Keywords
CNN, ResNet, ONNX, ImageNet, Computer Vision 

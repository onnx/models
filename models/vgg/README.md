# VGG

VGG presents the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. VGG networks have increased depth with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations achieved by pushing the depth to 16–19 weight layers. The work secured the first and the second places in the localisation and classification tracks respectively in ImageNet Challenge 2014. The representations from VGG generalise well to other datasets, where they achieve state-of-the-art results. 



## Model

The model below are variant of same network with different number of layers and use of batch normalisation.

 |Model        |ONNX Model  | Model archives|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|
|VGG 16|    [527.8 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx)    |  [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.model)     | 72.62     |      91.14     |
|VGG 16_bn|    [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.onnx)    |  [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.model)     |   72.71     |      91.21    |
|VGG 19|    [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.onnx)    |  [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.model)     | 73.72     |      91.58     |
|VGG 19_bn|    [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.onnx)    |  [548.2 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.model)     | 73.83    |      91.79     |

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
mxnet-model-server --models vgg16=https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.model
```

* **Run Prediction**:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/vgg16/predict -F "data=@kitten.jpeg"
```
Use the dataname as 'data' in predict call for all the above VGG models
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.



## Validation accuracy
The accuracies obtained by the models on the validation set as mentioned above. The accuracies has been calculate on center cropped 
images and is similar to accuracy obtained in the paper.

<!--|Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|VGG 16        |     72.62     |      91.14     |
|VGG 16_bn     |     72.71     |      91.21    |
|VGG 19        |     73.72     |      91.58     |
|VGG 19_bn     |     73.83    |      91.79     |
-->


## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_vgg.ipynb) to understand details for parameters and network for each of the above variants of VGG.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.


## References 
* **VGG 16** and **vVGG 19** are from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* **VGG 16_bn** and **VGG 19_bn** are the same models as above but with batch normalization applied after each convolution layer
## Contributors
## Keywords
CNN, VGG, ONNX, ImageNet, Computer Vision 
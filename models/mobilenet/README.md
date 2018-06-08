# MobileNet
MobileNet improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. MobileNet is based on an inverted residual structure where the shortcut connections are between the thin bottle- neck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. Additionally,  it removes non-linearities in the narrow layers in order to maintain representational power. 

## Model
* Version 2:

 |Model        |ONNX Model  | Model archives|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|
|MobileNet v2-1.0|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx)    |  [13.7 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.model)     | 70.94    |     89.99           |
|MobileNet v2-0.5|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.onnx)    |  [13.7 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.model)     |          |             |

## Inference
We used MXNet as framework with gluon APIs to perform inference. View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook. 
### Input 
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size, and H and W are expected to be at least 224. 
### Pre-processing
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

### Post-process
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
mxnet-model-server --models mobilenetv2_1_0=https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.model
```

* **Run Prediction**:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/mobilenetv2_1_0/predict -F "data=@kitten.jpeg"
```
Use the dataname as 'data' in predict call for all the above MobileNet models
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.


## Validation accuracy
The accuracies obtained by the models on the validation set as mentioned above. The accuracies has been calculate on center cropped 
images and is similar to accuracy obtained in the paper.
<!--* Version 1:

 |Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|MobileNet v2-1.0|     70.94    |     89.99           |
|MobileNet v2-0.5|              |             |

-->


## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_mobilenet.ipynb) to understand details for parameters and network for each of the above variants of MobileNet.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.


## References
**MobileNet-v2**
Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## Contributors

## Keyword
CNN, MobileNet, ONNX, ImageNet, Computer Vision 

# SqueezeNet

## Use cases
SqueezeNet models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. SqueezeNet models are highly efficient in terms of size and speed while providing good accuracies. This makes them ideal for platforms with strict constraints on size.

## Description
SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with 50x fewer parameters. SqueezeNet requires less communication across servers during distributed training, less bandwidth to export a new model from the cloud to an autonomous car and more feasible to deploy on FPGAs and other hardware with limited memory.

## Model
Squeezenet 1.1 gives AlexNet level of accuracy with 50X fewer parameters.
<!-- SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.-->

|Model        |ONNX Model  | ONNX version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|
|SqueezeNet 1.1|    [4.7 MB](https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.onnx)    |  1.2.1   |56.34     |     79.12     |

## Inference
We used MXNet as framework with gluon APIs to perform inference. View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook.

### Input 
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size, and H and W are expected to be at least 224. The inference was done using a jpeg image.

### Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing. Check [imagenet_preprocess.py](../imagenet_preprocess.py) for code.

### Output
The model outputs image scores for each of the [1000 classes of ImageNet](../synset.txt).

### Postprocessing
The post-processing involves calculating the softmax probablility scores for each class and sorting them to report the most probable classes. Check [imagenet_postprocess.py](../imagenet_postprocess.py) for code.

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#squeezenet_v1.1_onnx).

## Dataset
Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.

## Validation accuracy
The accuracy obtained by the model on the validation set is mentioned above. The accuracy has been calculate on center cropped images and is within 1-2% of the accuracy obtained in the paper.

## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_squeezenet.ipynb) to understand details for parameters and network for each of the above variants of SqueezeNet.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.

## References
* **SqueezeNet1.1**  
SqueezeNet1.1 presented in the [Official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) is an improved version of SqueezeNet1.0 from the paper [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
<!-- * **SqueezeNet1.1**   
Model from [Official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1).
-->
## Contributors
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)

## Acknowledgments
[MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

## Keywords
CNN, SqueezeNet, ONNX, ImageNet, Computer Vision

## License
Apache 2.0
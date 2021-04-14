<!--- SPDX-License-Identifier: Apache-2.0 -->

# Dense Upsampling Convolution (DUC)

## Use cases
DUC is a semantic segmentation model, i.e., for an input image the model labels each pixel in the image into a set of pre-defined categories. The model provides very good accuracy in terms of [mIOU](#metric) (mean Intersection Over Union) score and can be used in any application requiring semantic segmentation. In particular, since the model is trained on the [cityscapes dataset](#dset) which contains images from urban street scenes, it can be used effectively in self driving vehicle systems.

## Description
DUC is a CNN based model for semantic segmentation which uses an image classification network (ResNet) as a backend and achieves improved accuracy in terms of mIOU score using two novel techniques. The first technique is called Dense Upsampling Convolution (DUC) which generates pixel-level prediction by capturing and decoding more detailed information that is generally missing in bilinear upsampling. Secondly, a framework called Hybrid Dilated Convolution (HDC) is proposed in the encoding phase which enlarges the receptive fields of the network to aggregate global information. It also alleviates the checkerboard receptive field problem ("gridding") caused by the standard dilated convolution operation.

## Model
The model ResNet101_DUC_HDC uses ResNet101 as a backend network with both Dense Upsampling Convolution (DUC) and Hybrid Dilated Convolution (HDC) techniques.

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|[mIOU](#metric) (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|ResNet101_DUC_HDC|    [248.6 MB](model/ResNet101-DUC-7.onnx) | [282.0 MB](model/ResNet101-DUC-7.tar.gz) |1.2.2  |7 |81.92 |

## Inference
We used MXNet as framework to perform inference. View the notebook [duc-inference](dependencies/duc-inference.ipynb) to understand how to use above models for doing inference. A brief description of the inference process is provided below:

### Input
Since the model is trained on the cityscapes dataset which contains images of urban street scenes, the input should preferably be an image of a street scene to produce best results. There are no constraints on the size of the image. The example in the inference notebook is done using a png image.

### Preprocessing
The DUC layer has an effect of partitioning the image into d^2 subparts (d = downsampling rate). This is why the input image is extrapolated with a small border in order to obtain an accurate reshaped image after the DUC layer. After this the image is normalized using mean subtraction. Check [duc-preprocess.py](dependencies/duc-preprocess.py) for code.

### Output
The output of the network is a tensor of shape (1 X `label_num` X `H` * `W`) where `H` and `W` are the height and width of the output segmented map.

### Postprocessing
The output tensor is reshaped and resized to give the softmax map of shape (`H` X `W` X `label_num`). The raw label map is computed by doing an argmax on the softmax map. The script [cityscapes_labels.py](dependencies/cityscapes_labels.py) contains the segmentation category labels and their corresponding color map. Using this the colorized segmented images are generated. Check [duc-postprocess.py](dependencies/duc-postprocess.py) for code.

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#duc-resnet101_onnx).

## <a name="dset"></a>Dataset
Cityscapes dataset is used for training and validation. It is a large dataset that focuses on semantic understanding of urban street scenes. It contains 5000 images with fine annotations across 50 cities, different seasons, varying scene layout and background. There are a total of 30 categories in the dataset of which 19 are included for training and evaluation. The training, validation and test set contains 2975, 500 and 1525 fine images, respectively.

### Download
First, go to the [Cityscapes download page](https://www.cityscapes-dataset.com/downloads/) and register for an account (login if account already made). Next, find and download the following two files:

|Filename                 | Size  | Details|
|-------------------------|:------|:-------|
|leftImg8bit_trainvaltest.zip| 11 GB| train/val/test images|
|gtFine_trainvaltest.zip  | 241 MB| fine annotations for train and val sets|

### Setup
* Unpack the zip files into folders `leftImg8bit_trainvaltest` and `gtFine_trainvaltest`.
* Use the path to the train/val folders inside these folders for training/validation.

Please note that the dataset is under copyright. Refer to the [citation](https://www.cityscapes-dataset.com/citation/) page for details.

## Validation accuracy
The [mIOU](#metric) score obtained by the models on the validation set are mentioned above and they match with those mentioned in the paper.

## <a name="metric"></a>Validation
**mean Intersection Over Union (mIOU)** is the metric used for validation. For each class the intersection over union (IOU) of pixel labels between the output and the target segmentation maps is computed and then averaged over all classes to give us the mean intersection over union (mIOU).

We used MXNet framework to compute mIOU of the models on the validation set described above. Use the notebook [duc-validation](dependencies/duc-validation.ipynb) to verify the mIOU of the model. The scripts [cityscapes_loader.py](dependencies/cityscapes_loader.py), [cityscapes_labels.py](dependencies/cityscapes_labels.py) and [utils.py](dependencies/utils.py) are used in the notebook for data loading and processing.

## References
* All models are from the paper [Understanding Convolution for Semantic Segmentation](https://arxiv.org/abs/1702.08502).
* [TuSimple-DUC repo](https://github.com/TuSimple/TuSimple-DUC), [MXNet](http://mxnet.incubator.apache.org)

## Contributors
[abhinavs95](https://github.com/abhinavs95) (Amazon AI)

## License
Apache 2.0

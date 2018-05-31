# VGG
## ONNX model
* [vgg16]() ; Size: 
* [vgg19]() ; Size:
* [vgg16_bn]() ; Size: 
* [vgg19_bn]() ; Size:
## MMS archive
* [vgg16]() ; Size: 
* [vgg19]() ; Size:
* [vgg16_bn]() ; Size: 
* [vgg19_bn]() ; Size:
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)
### Download
First, go to the [download page](http://www.image-net.org/download-images) (you may need to register an account), and find the page for ILSVRC2012. Next, find and download the following two files:

|Filename                 | Size  |
|-------------------------|:------|
|ILSVRC2012_img_train.tar | 138 GB|
|ILSVRC2012_img_val.tar   | 6.3 GB|
### Setup
* Download helper script [extract_imagenet.py](../extract_imagenet.py) and validation image info [imagenet_val_maps.pklz](../imagenet_val_maps.pklz) and place in the same folder
* Run `python extract_imagenet.py --download-dir *path to download folder* --target-dir *path to extract folder*`
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook
## Training notebook
Use the notebook [train_vgg](train_vgg.ipynb) to train the model. It contains the hyperparameters and network details used for training the above models
## Verify validation accuracy
Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook
## References 
* **vgg16** and **vgg 19** are from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* **vgg16_bn** and **vgg19_bn** are the same models as above but with batch normalization applied after each convolution layer
## Accuracy measures
## Tags

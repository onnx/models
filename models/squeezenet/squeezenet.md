# Squeezenet
## ONNX model
* Version 1.0: <br>Download: [link]() ; Size: 5MB
* Version 1.1:<br>Download: [link]() ; Size: 5MB
## MMS archive
* Version 1.0: <br>Download: [link]() ; Size:
* Version 1.1:<br>Download: [link]() ; Size:
## Example notebook
## Training notebook
View the notebook that can be used for training [here](train_notebook_squeezenet.ipynb)
## Paper
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size - [link](https://arxiv.org/abs/1602.07360)
## Dataset and Preparation
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)
### Preparation:
* Run [extract_imagenet.py](../extract_imagenet.py). Make sure to have [imagenet_val_maps.pklz](../imagenet_val_maps.pklz) file in the same folder.
## Accuracy measures
## Tags
## Verify validation accuracy
Use the notebook [test_script.ipynb](../test_script.ipynb) to verify the accuracy of the model on the validation set

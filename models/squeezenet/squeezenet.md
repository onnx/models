# Squeezenet
## ONNX model
* [Squeezenet1.0]() ; Size: 5MB
* [Squeezenet1.1]() ; Size: 5MB
## MMS archive
* [Squeezenet1.0]() ; Size: 5MB
* [Squeezenet1.1]() ; Size: 5MB
## Example notebook
## Training notebook
Use the notebook [train_squeezenet](train_squeezenet.ipynb) to train the model. It contains the hyperparameters and network details used for training the above models
## References
* Squeezenet1.0 <br>[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
## Dataset and Preparation
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)
### Preparation:
* Download the .tar files of the dataset 
* Download [extract_imagenet.py](../extract_imagenet.py) and [imagenet_val_maps.pklz](../imagenet_val_maps.pklz) and place in the same folder
* Run `python extract_imagenet.py --download-dir *path to download folder* --target-dir *path to extract folder*`
## Accuracy measures
## Tags
## Verify validation accuracy
Use the notebook [test_script](../test_script.ipynb) to verify the accuracy of the model on the validation set

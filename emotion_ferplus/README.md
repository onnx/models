# FER+ Emotion Recognition

Download:
- release 1.1: https://www.cntk.ai/OnnxModels/opset_2/emotion_ferplus.tar.gz  
- master: https://www.cntk.ai/OnnxModels/opset_7/emotion_ferplus.tar.gz 

Model size: 34 MB

## Description
This model is a deep convolutional neural network for emotion recognition in faces. 

### Paper
"Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution" [arXiv:1608.01041](https://arxiv.org/abs/1608.01041)

### Dataset
The model is trained on the FER+ annotations for the standard Emotion FER [dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), as described in the above paper.

## Source
The model is trained in CNTK, using the cross entropy training mode. You can find the source code [here](https://github.com/ebarsoum/FERPlus).

## Model input and output
### Input
shape `(1x1x64x64)`
### Output
shape`(1x8)`
### Pre-processing steps
### Post-processing steps
Route the model output through a softmax function to map the aggregated activations across the network to probabilities across the 8 classes, where the labels map as follows:  
`emotion_table = {'neutral':0, 'happiness':1, 'surprise':2, 'sadness':3, 'anger':4, 'disgust':5, 'fear':6, 'contempt':7}`
### Sample test data 
Sets of sample input and output files are provided in 
* the .npz format (`test_data_*.npz`). The input is a `(1x1x64x64)` numpy array of a test image, while the output is an array of length 8 corresponding to the output of evaluating the model on the sample input.
* serialized protobuf TensorProtos (`.pb`), which are stored in the folders `test_data_set_*/`.

## License
MIT

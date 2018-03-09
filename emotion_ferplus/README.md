# FER+ Emotion Recognition

Download: https://www.cntk.ai/OnnxModels/emotion_ferplus.tar.gz

## Description
This model is a deep convolutional neural network for emotion recognition in faces. It is trained on the FER+ annotations for the standard Emotion FER [dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), as described in this [paper](https://arxiv.org/abs/1608.01041).

## Source
The model is trained in CNTK. You can find the source code [here](https://github.com/ebarsoum/FERPlus).

## Model input and output
The model expects a grayscale input image of the shape (1x64x64), normalized to pixel values between `[-1, 1]`. To normalize the input image, the following computation is performed: `(image - 127.5)/127.5`. 

Sets of sample input and output files are provided in .npz format (`test_data_*.npz`). The input is a normalized (1x64x64) numpy array of a test image, while the output is an array of length 8 corresponding to the output of evaluating the model on the sample input.

## License
MIT
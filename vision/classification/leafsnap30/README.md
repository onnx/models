# LeafSnap30

## Description
LeafSnap30 is a Neural Network model trained on the [LeafSnap 30 dataset](https://zenodo.org/record/5061353/). It addresses an image classificaiton task- identifying 30 tree species from images of their leaves. This task has been approached with classical computer vision methods a decade ago on more species dataset containing artifacts, while this model is trained on a smaller and cleaner dataset, particularly useful for demonstrating NN classification on simple, yet realistic scientific task.

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Model Name       | Relative link to ONNX Model with size  | tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |
|LeafSnap30|    [1.48 MB](model/leafsnap_model.onnx)    | [1.55 MB](model/leafsnap30.tar.gz) | 1.9.0  |11 | train: 95%, validation: 86, test: 83%     |

### Source
Pytorch LeafSnap30 ==> ONNX LeafSnap30 

## Inference
The steps needed to run the pretrained model with the onnxruntime are implemented within the explainability library [dianna](https://github.com/dianna-ai/dianna) in this [code](https://github.com/dianna-ai/dianna/blob/main/dianna/utils/onnx_runner.py). An example [tutorial notebook](https://github.com/dianna-ai/dianna/blob/main/tutorials/lime_images.ipynb) shows on how to use the model with dianna.

### Input
The input to the model is a ``float32`` tensor of shape ``(-1, 3, 128, 128)``, where -1 is the batch axis. Each image is a ``128x128`` RGB image, with the colour channels as first axis.

### Preprocessing
The input image is loaded to a numpy array. The pixel values are then scaled to the 0-1 range. 

Example:

```
# load and plot the example image
img = np.array(Image.open(f'data/leafsnap_example_{true_species}.jpg'))

plt.imshow(img)
plt.title(f'Species: {true_species}');

# the model expects float32 values in the 0-1 range for each pixel, with the colour channels as first axis
# the .jpg file has 0-255 ints with the channel axis last so it needs to be changed
input_data = img.transpose(2, 0, 1).astype(np.float32) / 255.
```

### Output
Output of this model is the likelihood of each tree species before softmax, a tensor of shape ``` 1 x 30```. 

## Model Creation

### Dataset (Train and validation)
From the original LeafSnap dataset, the 30 most prominent classes were selected. The images taken in a lab were cropped semi-manually to remove any rulers and color calibration image parts. Notebooks describing these steps are available [here](https://github.com/dianna-ai/dianna-exploration/tree/main/example_data/dataset_preparation/LeafSnap). The LeafSnap30 dataset is also available on [Zenodo](https://zenodo.org/record/5061353).

### Training
The model is a CNN with 4 hidden layers, built in PyTorch and converted to ONNX. A notebook for the generation of the model, including the used hyperparameters, is available [here](https://github.com/dianna-ai/dianna-exploration/main/example_data/model_generation/).

### Validation accuracy
The notebook used for training the model also shows how accuracy on the validation and test datasets is calculated. The actual values were taken from the hyperparameter sweep executed with [Weights & Biases](wandb.ai).

### References
[Leafsnap: A Computer Vision System for Automatic Plant Species Identification](https://rdcu.be/c0aBX) (original LeafSnap paper)

[LeafSnap30](https://zenodo.org/record/5061353/) (Zenodo dataset archive)

DIANNA: Deep Insight And Neural Network Analysis [![status](https://joss.theoj.org/papers/f0592c1aecb3711e068b58970588f185/status.svg)](https://joss.theoj.org/papers/f0592c1aecb3711e068b58970588f185)

## Contributors
- [Leon Oostrum](https://github.com/loostrum) (Netherlands eScience Center)
- [Christiaan Meijer](https://github.com/cwmeijer) (Netherlands eScience Center)
- [Yang Liu](https://github.com/geek-yang) (Netherlands eScience Center)
- [Patrick Bos](https://github.com/egpbos) (Netherlands eScience Center)
- [Elena Ranguelova](https://github.com/elboyran) (Netherlands eScience Center)

## License
Apache 2.0
<hr>

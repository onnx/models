# LeafSnap30

## Description
LeafSnap30 is a Neural Network model trained on the [LeafSnap 30 dataset](https://zenodo.org/record/5061353/). It addresses an image classificaiton task- identifying 30 tree species from images of their leaves. This task has been approached with classical computer vision methods a decade ago on more species dataset containing artifacts, while this model is trained on a smaller and cleaner dataset, particularly useful for demonstrating NN classification on simple, yet realistic scientific task.

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Model Name       | Relative link to ONNX Model with size  | tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |
|LeafSnap30|    [1.48 MB](model/leafsnap_model.onnx)    |N/A | 1.9.0  |11 | train: 87%, test: 74%     |

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
This section should discuss datasets and any preparation steps if required.

### Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional).

Also clarify in case the model is not trained from scratch and include the source/process used to obtain the ONNX model.

### Validation accuracy
Validation script/notebook used to obtain accuracy reported above along with details of how to use it and reproduce accuracy. Details of experiments leading to accuracy from the reference paper.

## Test Data Creation

Creating test data for uploaded models can help CI to verify the uploaded models by ONNXRuntime utilties. Please upload the ONNX model with created test data (`test_data_set_0`) in the .tar.gz.

### Requirement
```
pip install onnx onnxruntime numpy
git clone https://github.com/onnx/models.git
````
### Usage
```
def create_test_dir(model_path, root_path, test_name,
                    name_input_map=None, symbolic_dim_values_map=None,
                    name_output_map=None):
    """
    Create a test directory that can be used with onnx_test_runner, onnxruntime_perf_test.
    Generates random input data for any missing inputs.
    Saves output from running the model if name_output_map is not provided.

    :param model_path: Path to the onnx model file to use.
    :param root_path: Root path to create the test directory in.
    :param test_name: Name for test. Will be added to the root_path to create the test directory name.
    :param name_input_map: Map of input names to numpy ndarray data for each input.
    :param symbolic_dim_values_map: Map of symbolic dimension names to values to use for the input data if creating
                                    using random data.
    :param name_output_map: Optional map of output names to numpy ndarray expected output data.
                            If not provided, the model will be run with the input to generate output data to save.
    :return: None
    """
```
### Example
The input/output .pb files will be produced under `temp/examples/test1/test_data_set_0`.
```
import sys
sys.path.append('<onnx/models root dir>/workflow_scripts/')
import ort_test_dir_utils
import numpy as np

# example model with two float32 inputs called 'input' [batch_size, 1, 224, 224])
model_path = '<onnx/models root dir>/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx'

# If there is no symbolic_dim
ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test1')

# when using the default data generation any symbolic dimension values can be provided
# otherwise the default value for missing symbolic_vals would be 1

symbolic_vals = {'batch_size': 1} # provide value for symbolic dim named 'batch_size'

# let create_test_dir create random input in the (arbitrary) default range of -10 to 10. 
# it will create data of the correct type based on the model.
ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test1', symbolic_dim_values_map=symbolic_vals)

# alternatively some or all input can be provided directly. any missing inputs will have random data generated.
batch_size = 64
inputs = {'input': np.random.rand(batch_size, 1, 224, 224).astype(np.float32)}

ort_test_dir_utils.create_test_dir(model_path, 'temp/examples', 'test2', name_input_map=inputs)

```

### More details
https://github.com/microsoft/onnxruntime/blob/master/tools/python/PythonTools.md

<hr>

### Update ONNX_HUB_MANIFEST.json for ONNX Hub
If this PR does update/add .onnx or .tar.gz files, please use `python workflow_scripts/generate_onnx_hub_manifest.py --target diff` to update ONNX_HUB_MANIFEST.json with according model information (especially SHA) for ONNX Hub.

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

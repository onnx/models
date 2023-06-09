# Create test data from onnxruntime test utilities

Creating test data for uploaded models can help CI to verify the uploaded models by ONNXRuntime utilties. Please upload the ONNX model with created test data (`test_data_set_0`).

## Requirement
```
pip install onnx onnxruntime numpy
git clone https://github.com/onnx/models.git
````
## Usage
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
## Example
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

## More details
https://github.com/microsoft/onnxruntime/blob/main/tools/python/PythonTools.md

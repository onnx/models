# ONNX Models

This is a repository for storing ONNX models.

### Models

- [BVLC AlexNet](bvlc_alexnet) (244 MByte)

- [BVLC GoogleNet](bvlc_googlenet) (28 MByte)

- [BVLC CaffeNet](bvlc_reference_caffenet) (244 MByte)

- [BVLC R-CNN ILSVRC13](bvlc_reference_rcnn_ilsvrc13) (231 MByte)

- [DenseNet-121](densenet121) (33 MByte)

- [Inception-v1](inception_v1) (28 MByte)

- [Inception-v2](inception_v2) (45 MByte)

- [ResNet-50](resnet50) (103 MByte)

- [ShuffleNet](shufflenet) (5.3 MByte)

- [SqueezeNet](squeezenet) (5 MByte)

- [VGG-19](vgg19) (575 MByte)

- [ZFNet](zfnet) (349 MByte)

- [MNIST](mnist) (26 kByte)

- [Emotion-FERPlus](emotion_ferplus) (34 MByte)

- [Tiny-YOLOv2](tiny_yolov2) (61 MByte)

### Usage

Every ONNX backend should support running these models out of the box. After downloading and extracting the tarball of each model, there should be

- A protobuf file `model.onnx` which is the serialized ONNX model.
- Test data.


The test date are provided in two different formats:
- Serialized Numpy archives, which are files named like `test_data_*.npz`, each file contains one set of test inputs and outputs.
They can be used like this:

```python
import numpy as np
import onnx
import onnx_backend as backend

# Load the model and sample inputs and outputs
model = onnx.load(model_pb_path)
sample = np.load(npz_path, encoding='bytes')
inputs = list(sample['inputs'])
outputs = list(sample['outputs'])

# Run the model with an onnx backend and verify the results
np.testing.assert_almost_equal(outputs, backend.run_model(model, inputs))
```

- Serialized protobuf TensorProtos, which are stored in folders named like `test_data_set_*`.
They can be used as the following:
```python
import numpy as np
import onnx
import os
import glob
import onnx_backend as backend

from onnx import numpy_helper

model = onnx.load('model.onnx')
test_data_dir = 'test_data_set_0'

# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# Run the model on the backend
outputs = list(backend.run_model(model, inputs))

# Compare the results with reference outputs.
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o)
```

# License

[MIT License](LICENSE)

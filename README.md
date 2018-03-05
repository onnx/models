# ONNX Models

This is a repository for storing ONNX models.

### Models

- [BVLC AlexNet](bvlc_alexnet) (217 MByte)

- [BVLC GoogleNet](bvlc_googlenet) (26 MByte)

- [BVLC CaffeNet](bvlc_reference_caffenet) (217 MByte)

- [BVLC R-CNN ILSVRC13](bvlc_reference_rcnn_ilsvrc13) (206 MByte)

- [DenseNet-121](densenet121) (31 MByte)

- [Inception-v1](inception_v1) (26 MByte)

- [Inception-v2](inception_v2) (41 MByte)

- [ResNet-50](resnet50) (92 MByte)

- [ShuffleNet](shufflenet) (6.7 MByte)

- [SqueezeNet](squeezenet) (6.0 MByte)

- [VGG-16](vgg16) (310 MByte)

- [VGG-19](vgg19) (510 MByte)

### Usage

Every ONNX backend should support running these models out of the box. After dowloading and extracting the tarball of each model, there should be

- A protobuf file `model.onnx` which is the serialized ONNX model.
- Several sets of sample inputs and outputs files (`test_data_*.npz`), they are numpy serialized archive.

e.g. they can be used like this:

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

# License

[MIT License](LICENSE)

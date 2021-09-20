# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import onnxruntime
import numpy as np

from retinanet.model import Model
from PIL import Image
from torchvision import transforms

from onnx import numpy_helper
import urllib

data_dir = 'test_data_set_0'
url, filename = ("https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/retinanet/dependencies/demo.jpg", "demo.jpg")
urllib.request.urlretrieve(url, filename)


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def save_tensor_proto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())


def save_data(test_data_dir, prefix, names, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        save_tensor_proto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), names[i], d)


def save_model(name, model, inputs, outputs, input_names=None, output_names=None, **kwargs):
    if hasattr(model, 'train'):
        model.train(False)
    dir = './'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'test_' + name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])
    if input_names is None:
        input_names = []
        for i, _ in enumerate(inputs_flatten):
            input_names.append('input' + str(i+1))
    else:
        np.testing.assert_equal(len(input_names), len(inputs_flatten),
                                "Number of input names provided is not equal to the number of inputs.")

    if output_names is None:
        output_names = []
        for i, _ in enumerate(outputs_flatten):
            output_names.append('output' + str(i+1))
    else:
        np.testing.assert_equal(len(output_names), len(outputs_flatten),
                                "Number of output names provided is not equal to the number of output.")

    model_dir = os.path.join(dir, 'model.onnx')
    torch.onnx.export(model, inputs, model_dir, verbose=True, input_names=input_names,
                      output_names=output_names, example_outputs=outputs, **kwargs)

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    save_data(test_data_dir, "input", input_names, inputs_flatten)
    save_data(test_data_dir, "output", output_names, outputs_flatten)

    return model_dir, test_data_dir


def inference(file, inputs, outputs):
    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])

    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
    sess = ort.InferenceSession(file)
    ort_inputs = dict((sess.get_inputs()[i].name, to_numpy(input)) for i, input in enumerate(inputs_flatten))
    res = sess.run(None, ort_inputs)

    if outputs is not None:
        print("== Checking model output ==")
        [np.testing.assert_allclose(to_numpy(output), res[i], rtol=1e-03, atol=1e-05) for i, output in enumerate(outputs_flatten)]
        print("== Done ==")


def torch_inference(model, input):
    print("====== Torch Inference ======")
    output=model(input)
    return output


def ort_inference(file, inputs_flatten, outputs_flatten):
    print("====== ORT Inference ======")
    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
    ort_sess = onnxruntime.InferenceSession(file)
    ort_inputs = dict((ort_sess.get_inputs()[i].name, to_numpy(input)) for i, input in enumerate(inputs_flatten))
    ort_outs = ort_sess.run(None, ort_inputs)
    if outputs_flatten is not None:
        print("== Checking model output ==")
        [np.testing.assert_allclose(to_numpy(output), ort_outs[i], rtol=1e-03, atol=1e-05) for i, output in
         enumerate(outputs_flatten)]
    print("== Done ==")


# Download pretrained model from:
# https://github.com/NVIDIA/retinanet-examples/releases/tag/19.04
model, state = Model.load('retinanet_rn101fpn/retinanet_rn101fpn.pth')
model.eval()
model.exporting = True
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)
output = torch_inference(model, input_tensor)

# Test exported model with TensorProto data saved in files
inputs_flatten = flatten(input_tensor.detach().cpu().numpy())
inputs_flatten = update_flatten_list(inputs_flatten, [])
outputs_flatten = flatten(output)
outputs_flatten = update_flatten_list(outputs_flatten, [])

model_dir, data_dir = save_model('retinanet_resnet101', model.cpu(), input_tensor, output, input_names=['input'],
                                 opset_version=9)

ort_inference(model_dir, inputs_flatten, outputs_flatten)

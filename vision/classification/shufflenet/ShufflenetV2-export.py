import torch
import onnxruntime
import onnx
from onnx import numpy_helper
from PIL import Image
from torchvision import transforms

import numpy as np
import os
import urllib

#           GitHub Repo    |    Model
MODELS = [
    ('pytorch/vision:v0.5.0', 'shufflenet_v2_x0_5'),
    ('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0'),
]
data_dir = 'test_data_set_0'

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


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
    if isinstance(model, torch.jit.ScriptModule):
        torch.onnx._export(model, inputs, model_dir, verbose=True, input_names=input_names,
                           output_names=output_names, example_outputs=outputs, **kwargs)
    else:
        torch.onnx.export(model, inputs, model_dir, verbose=True, input_names=input_names,
                          output_names=output_names, example_outputs=outputs, **kwargs)

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    save_data(test_data_dir, "input", input_names, inputs_flatten)
    save_data(test_data_dir, "output", output_names, outputs_flatten)

    return model_dir, test_data_dir


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def inference(file, inputs, outputs):
    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])

    sess = onnxruntime.InferenceSession(file)
    ort_inputs = dict((sess.get_inputs()[i].name, to_numpy(input)) for i, input in enumerate(inputs_flatten))
    res = sess.run(None, ort_inputs)

    if outputs is not None:
        print("== Checking model output ==")
        [np.testing.assert_allclose(to_numpy(output), res[i], rtol=1e-03, atol=1e-05) for i, output in enumerate(outputs_flatten)]
        print("== Done ==")


def shufflenetv2_test():
    for github_repo, model in MODELS:
        # Load pretrained model
        model = torch.hub.load(github_repo, model, pretrained=True)
        model.eval()
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_1 = input_tensor.unsqueeze(0)
        output_1 = model(input_1)

        model_dir, data_dir = save_model('shufflenetv2', model.cpu(), input_1, output_1,
                                         opset_version=10,
                                         input_names=['input'],
                                         output_names=['output'],
                                         dynamic_axes={"input": {0: 'batch_size'}, "output": {0: 'batch_size'}})

        # Test exported model with TensorProto data saved in files
        inputs_flatten = flatten(input_1)
        inputs_flatten = update_flatten_list(inputs_flatten, [])
        outputs_flatten = flatten(output_1)
        outputs_flatten = update_flatten_list(outputs_flatten, [])

        inputs = []
        for i, _ in enumerate(inputs_flatten):
            f_ = os.path.join(data_dir, '{0}_{1}.pb'.format("input", i))
            tensor = onnx.TensorProto()
            with open(f_, 'rb') as file:
                tensor.ParseFromString(file.read())
            inputs.append(numpy_helper.to_array(tensor))

        outputs = []
        for i, _ in enumerate(outputs_flatten):
            f_ = os.path.join(data_dir, '{0}_{1}.pb'.format("output", i))
            tensor = onnx.TensorProto()
            with open(f_, 'rb') as file:
                tensor.ParseFromString(file.read())
            outputs.append(numpy_helper.to_array(tensor))

        inference(model_dir, inputs, outputs)

        # Test model with different input
        input_2 = torch.randn(6, 3, 224, 224)
        output_2 = model(input_2)
        inference(model_dir, input_2, output_2)


shufflenetv2_test()

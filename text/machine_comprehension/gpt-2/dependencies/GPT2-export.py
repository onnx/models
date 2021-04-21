# SPDX-License-Identifier: Apache-2.0

import torch
import onnxruntime
import onnx
from onnx import numpy_helper
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

import numpy as np
import os

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut          | save_name
MODELS = [
    (GPT2Model, GPT2Tokenizer, 'gpt2', 'gpt2'),
    (GPT2LMHeadModel, GPT2Tokenizer, 'gpt2', 'gpt2-lm-head'),
]
data_dir = 'test_data_set_0'


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


def gpt2_test():
    for model_class, tokenizer_class, pretrained_weights, save_name in MODELS:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        model.eval()
        # Encode text
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids_1 = torch.tensor(
            [[tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)]])
        with torch.no_grad():
            output_1 = model(input_ids_1)  # Models outputs are now tuples

        model_dir, data_dir = save_model(save_name, model.cpu(), input_ids_1, output_1,
                                         opset_version=10,
                                         input_names=['input1'],
                                         dynamic_axes={'input1': [0, 1, 2, 3]})

        # Test exported model with TensorProto data saved in files
        inputs_flatten = flatten(input_ids_1)
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

        # Test exported model with a new input
        print("== Feeding model with new input ==")
        input_ids_2 = torch.tensor(
              [[tokenizer.encode("Here is some alternative text to encode I love Seattle", add_special_tokens=True)]])
        with torch.no_grad():
            output_2 = model(input_ids_2)

        inference(model_dir, input_ids_2, output_2)


gpt2_test()

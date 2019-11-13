import torch
import onnxruntime
from onnx import numpy_helper
from transformers import GPT2Model, GPT2Tokenizer

import numpy as np
import os

#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [
    (GPT2Model, GPT2Tokenizer, 'gpt2'),
]
data_dir = 'test_data_set_0'


def save_tensor_proto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())


def save_data(test_data_dir, prefix, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        save_tensor_proto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), prefix + str(i+1), d)


def save(name, model, inputs, outputs, input_names=['input'], output_names=['output'], **kwargs):
    if hasattr(model, 'train'):
        model.train(False)
    dir = './'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'test_' + name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    def f(t):
        return [f(i) for i in t] if isinstance(t, (list, tuple)) else t

    def g(t, res):
        for i in t:
            res.append(i) if not isinstance(i, (list, tuple)) else g(i, res)
        return res

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

    inputs = f(inputs)
    inputs = g(inputs, [])
    outputs = f(outputs)
    outputs = g(outputs, [])

    save_data(test_data_dir, 'input', inputs)
    save_data(test_data_dir, 'output', outputs)

    return model_dir


def inference(f, inputs, outputs):
    sess = onnxruntime.InferenceSession(f)
    ort_inputs = dict((sess.get_inputs()[i].name, np.expand_dims(input, 0)) for i, input in enumerate(inputs))
    res = sess.run(None, ort_inputs)

    if outputs is not None:
        print("== CHEKING OUTPUT ==")
        all_hidden_states, past = outputs
        np.testing.assert_allclose(all_hidden_states.numpy(), res[0], rtol=1e-02, atol=1e-05)
        for i in range(len(past)):
            np.testing.assert_allclose(past[i].detach().numpy(), res[1 + i], rtol=1e-03, atol=1e-05)
        print("== DONE ==")


def gpt2_test():
    for model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        model.eval()
        # Encode text
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids_1 = torch.tensor(
            [tokenizer.encode("Here is some text to encode Hello World", add_special_tokens=True)])
        with torch.no_grad():
            output_1 = model(input_ids_1)  # Models outputs are now tuples

        input_ids_2 = torch.tensor(
            [tokenizer.encode("Here is some alternative text to encode I love Seattle", add_special_tokens=True)])
        with torch.no_grad():
            output_2 = model(input_ids_2)

        f = save('gpt2', model.cpu(), input_ids_1, output_1,
                 opset_version=10,
                 input_names=['input'],
                 dynamic_axes={'input': [0, 1, 2, 3]})

        inference(f, input_ids_1, output_1)
        #Test dynamic input shapes
        inference(f, input_ids_2, output_2)


gpt2_test()

import onnx
import torch
import os
import numpy as np
from onnx import numpy_helper

data_dir = 'test_data_set_0'

def SaveTensorProto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())

def SaveData(test_data_dir, prefix, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        SaveTensorProto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), prefix + str(i+1), d)

def Save(dir, name, model, inputs, outputs, input_names = ['input1'], output_names = ['output1']):
    if hasattr(model, 'train'):
        model.train(False)
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

    if isinstance(model, torch.jit.ScriptModule):
        torch.onnx._export(model, tuple(inputs), os.path.join(dir, 'model.onnx'), verbose=True, input_names=input_names, output_names=output_names, example_outputs=outputs)
    else:
        torch.onnx.export(model, tuple(inputs), os.path.join(dir, 'model.onnx'), verbose=True, input_names=input_names, output_names=output_names)

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    inputs = f(inputs)
    inputs = g(inputs, [])
    outputs = f(outputs)
    outputs = g(outputs, [])

    SaveData(test_data_dir, 'input', inputs)
    SaveData(test_data_dir, 'output', outputs)

# def UpdateInputOutputDims(dir, name, input_dims, output_dims):
#     dir = os.path.join(dir, 'test_' + name, 'model.onnx')
#     model = onnx.load(dir)

#     # inputs
#     for i, input_dim_arr in enumerate(input_dims):
#         for j, dim in enumerate(input_dim_arr):
#             dim_proto = model.graph.input[i].type.tensor_type.shape.dim[j]
#             if dim >= 0:
#                 dim_proto.dim_value = dim
#             else:
#                 dim_proto.dim_param = 'in_' + str(i) + '_' + str(j)

#     for i, output_dim_arr in enumerate(output_dims):
#         for j, dim in enumerate(output_dim_arr):
#             dim_proto = model.graph.output[i].type.tensor_type.shape.dim[j]
#             if dim >= 0:
#                 dim_proto.dim_value = dim
#             else:
#                 dim_proto.dim_param = 'out_' + str(i) + '_' + str(j)

#     model.ir_version = 3
#     onnx.checker.check_model(model)
#     inferred_model = shape_inference.infer_shapes(model)
#     onnx.checker.check_model(inferred_model)
#     onnx.save(model, dir)

# def update_inputs_outputs_dims(model, input_dims, output_dims):
#     """
#         This function updates the sizes of dimensions of the model's inputs and outputs to the values
#         provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
#         will be set for that dimension.
#     """
#     for i, input_dim_arr in enumerate(input_dims):
#         for j, dim in enumerate(input_dim_arr):
#             dim_proto = model.graph.input[i].type.tensor_type.shape.dim[j]
#             if dim >= 0:
#                 dim_proto.dim_value = dim
#             else:
#                 dim_proto.dim_param = 'in_' + str(i) + '_' + str(j)

#     for i, output_dim_arr in enumerate(output_dims):
#         for j, dim in enumerate(output_dim_arr):
#             dim_proto = model.graph.output[i].type.tensor_type.shape.dim[j]
#             if dim >= 0:
#                 dim_proto.dim_value = dim
#             else:
#                 dim_proto.dim_param = 'out_' + str(i) + '_' + str(j)

#     onnx.checker.check_model(model)
#     return model
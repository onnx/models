import sys
import json
import time
import numpy as np
import onnxruntime as ort
import os
from image_utils import (
    load_and_preprocess_image,
    top_n_probabilities,
    load_labels
)

onnx_model_path  = sys.argv[1]
model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]

num_runs = 10

test_img_file = 'YellowLabradorLooking_new.jpg'
labels = load_labels('imagenet-classes-1000.txt')
test_img_normalized_nchw = load_and_preprocess_image(test_img_file)
ifm = test_img_normalized_nchw.numpy()
run_softmax = True

session_options = ort.SessionOptions()
onnx_session = ort.InferenceSession(
                   onnx_model_path,
                   sess_options=session_options,
                   providers=["VitisAIExecutionProvider"],
               )

input_name = onnx_session.get_inputs()[0].name
for i in range(num_runs):
    
    start = time.time()
    ofm_aie = onnx_session.run(None, {input_name: ifm})
    end = time.time()
    running_time = (end-start)
    print(i + 1, 'Running Time:',running_time,'seconds')



print('Top 3 Probabilities')
# ofm is [ndarray[1,1000]]
res = ofm_aie[0][0]
top3 = top_n_probabilities(res, labels, top_n=3, run_softmax=run_softmax)
header_format = '{0:36s}|{1:20s}'
cell_format = '{0:36s}|{1:8.2f}'
print(header_format.format(36*'-', 12*'-'))
print(header_format.format('Classification', 'Percentage'))
print(header_format.format(36*'-', 12*'-'))
for c in top3:
    print(cell_format.format(c[0], c[1]))
    print(header_format.format(36*'-', 12*'-'))

# Check if the result matches expectations
error_code = 0
exp_prob = 67.02
if top3[0][0] != "Labrador retriever":
    print(f"ERROR: Incorrect classifcation result: Test: {top3[0][0]}. Expected: Labrador retriever.")
    error_code = 1
else:
    print("INFO: Test passed")

sys.exit(error_code)

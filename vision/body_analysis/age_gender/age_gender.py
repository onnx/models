import onnxruntime
import cv2
import numpy as np
from datetime import datetime


# fields
start_time = datetime.now()
age_model = "vgg_ilsvrc_16_age_imdb_wiki.onnx"
gender_model = "vgg_ilsvrc_16_gender_imdb_wiki.onnx"
image_name = "images/brad.jpg"
size = (224, 224)


# prepare image
# resize to 224 x 224, BGR, float [0, 255], expand to batch size 1
jpg = cv2.imread(image_name)
img = cv2.resize(jpg, size)
img = img[:, :, ::-1].transpose(2, 0, 1)
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)
print('Input image: {0}'.format(image_name))


# prepare and run gender prediction session
gender_session = onnxruntime.InferenceSession(gender_model)
input_name = gender_session.get_inputs()[0].name
outputs = gender_session.run(None, {input_name: img})[0][0]
label = "Woman" if (np.argmax(outputs) == 0) else "Man"
print('Gender: {}'.format(label))


# prepare age prediction session
age_session = onnxruntime.InferenceSession(age_model)
input_name = age_session.get_inputs()[0].name
outputs = age_session.run(None, {input_name: img})[0][0]
age = round(sum(outputs * list(range(0, 101))), 1)
print('Age: {}'.format(age))


# time check
time_elapsed = datetime.now() - start_time 
print('Time elapsed {} (hh:mm:ss.ms)'.format(time_elapsed))

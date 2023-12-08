# SPDX-License-Identifier: Apache-2.0

import numpy as np
import math
import cv2 as cv
import mxnet as mx

def preprocess(im, rgb_mean):
    '''
    Preprocessing function for DUC
    input : input image and rgb mean
    output : MXNet ndarray
    '''
    # Convert to float32
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0],im.shape[1]]
    cell_shapes = [math.ceil(l / 8)*8 for l in test_shape]
    test_img = cv.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0, max(0, int(cell_shapes[1]) - im.shape[1]), cv.BORDER_CONSTANT, value=rgb_mean)
    test_img = np.transpose(test_img, (2, 0, 1))
    # subtract rbg mean
    for i in range(3):
        test_img[i] -= rgb_mean[i]
    test_img = np.expand_dims(test_img, axis=0)
    # convert to ndarray
    test_img = mx.ndarray.array(test_img)
    return test_img

# SPDX-License-Identifier: Apache-2.0

'''
This script contains helper functions which are used by cityscapes_loader.py in duc-validation.ipynb
'''

import logging
import math
import os
import random
import time
from PIL import Image
from datetime import datetime

import cv2 as cv
import mxnet as mx
import numpy as np

import cityscapes_labels


# save symbol
def save_symbol(net, net_prefix):
    net.save('%s-symbol.json' % net_prefix)


# save parameters
def save_parameter(net, net_prefix, data_shape):
    executor = net.simple_bind(mx.gpu(0), data=data_shape)
    arg_params = executor.arg_dict
    aux_params = executor.aux_dict

    save_dict = {('arg:%s' % k): v for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k): v for k, v in aux_params.items()})
    param_name = '%s.params' % net_prefix
    mx.ndarray.save(param_name, save_dict)


# save log
def save_log(prefix, output_dir):
    fmt = '%(asctime)s %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        format=fmt,
                        datefmt=date_fmt,
                        filename=os.path.join(output_dir,
                                              prefix + '_' + datetime.now().strftime('%Y_%m_%d_%H:%M:%S') + '.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# replace ids with train_ids
def replace_city_labels(label_data):
    labels = cityscapes_labels.labels
    converted = np.ones(label_data.shape, dtype=np.float) * 255
    # id to trainId
    id2trainId = {label.id: label.trainId for label in labels}
    for id in id2trainId:
        trainId = id2trainId[id]
        converted[label_data == id] = trainId
    return converted


# get the data of image and label for networks including a ye layer
def get_single_image_duc(item, input_args):
    # parse options
    data_path = input_args.get('data_path')
    label_path = input_args.get('label_path', '')
    data_shape = input_args.get('data_shape')
    convert_label = input_args.get('convert_label', False)

    crop_sz = (data_shape[0][-1], data_shape[0][-2])
    use_random_crop = input_args.get('use_random_crop', False)
    use_mirror = input_args.get('use_mirror', False)
    scale_factors = input_args.get('scale_factors', [1])
    rgb_mean = input_args.get('rgb_mean', [128, 128, 128])
    ignore_label = input_args.get('ignore_label', 255)
    stride = input_args.get('ds_rate', 8)
    cell_width = input_args.get('cell_width', 1)
    random_bound = input_args.get('random_bound')

    # read data, scale, and random crop
    im = cv.imread(os.path.join(data_path, item[0]))
    # change bgr to rgb
    im = im[:, :, [2, 1, 0]]

    im_size = (im.shape[0], im.shape[1])
    scale_factor = random.choice(scale_factors)
    scaled_shape = (int(im_size[0]*scale_factor), int(im_size[1]*scale_factor))
    random_bound = (int(random_bound[0]*scale_factor), int(random_bound[1]*scale_factor))
    crop_coor = [int(int(c) * scale_factor) for c in item[-1]]

    if use_random_crop:
        x0 = crop_coor[0] + random.randint(-random_bound[0], random_bound[0]) - crop_sz[0] / 2
        y0 = crop_coor[1] + random.randint(-random_bound[1], random_bound[1]) - crop_sz[1] / 2
    else:
        # center crop
        x0 = int(crop_coor[0] - crop_sz[0] / 2)
        y0 = int(crop_coor[1] - crop_sz[1] / 2)
    x1 = int(x0 + crop_sz[0])
    y1 = int(y0 + crop_sz[1])

    # resize
    scaled_img = cv.resize(im, (scaled_shape[1], scaled_shape[0]), interpolation=cv.INTER_LINEAR)

    # crop and make boarder
    pad_w_left = max(0 - y0, 0)
    pad_w_right = max(y1 - scaled_shape[1], 0)
    pad_h_up = max(0 - x0, 0)
    pad_h_bottom = max(x1 - scaled_shape[0], 0)

    x0 += pad_h_up
    x1 += pad_h_up
    y0 += pad_w_left
    y1 += pad_w_left

    img_data = np.array(scaled_img, dtype=np.float)
    img_data = cv.copyMakeBorder(img_data, pad_h_up, pad_h_bottom, pad_w_left, pad_w_right, cv.BORDER_CONSTANT,
                                 value=list(rgb_mean))
    img_data = img_data[x0:x1, y0:y1, :]
    img_data = np.transpose(img_data, (2, 0, 1))

    # subtract rgb mean
    for i in range(3):
        img_data[i] -= rgb_mean[i]

    # read label
    img_label = np.array(Image.open(os.path.join(label_path, item[1])))
    img_label = cv.resize(img_label, (scaled_shape[1], scaled_shape[0]), interpolation=cv.INTER_NEAREST)
    img_label = np.array(img_label, dtype=np.float)
    img_label = cv.copyMakeBorder(img_label, pad_h_up, pad_h_bottom, pad_w_left, pad_w_right, cv.BORDER_CONSTANT,
                                  value=ignore_label)
    img_label = img_label[x0:x1, y0:y1]

    # resize label according to down sample rate
    if cell_width > 1:
        img_label = cv.resize(img_label, (int(crop_sz[1] / cell_width), int(crop_sz[0] / cell_width)),
                              interpolation=cv.INTER_NEAREST)

    # use mirror
    if use_mirror and random.randint(0, 1) == 1:
        img_data = img_data[:, :, ::-1]
        img_label = img_label[:, ::-1]

    # convert label from label id to train id
    if convert_label:
        img_label = replace_city_labels(img_label)

    feat_height = int(math.ceil(float(crop_sz[0]) / stride))
    feat_width = int(math.ceil(float(crop_sz[1]) / stride))

    img_label = img_label.reshape((feat_height, int(stride / cell_width), int(feat_width), int(stride / cell_width)))
    img_label = np.transpose(img_label, (1, 3, 0, 2))
    img_label = img_label.reshape((-1, feat_height, feat_width))
    img_label = img_label.reshape(-1)
    return [img_data], [img_label]


# get palette for coloring
def get_palette():
    # get palette
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette


# check point
def do_checkpoint(prefix, interval):
    def _callback(iter_no, sym, arg, aux):
        if (iter_no + 1) % interval == 0:
            mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback


# speed calculator
class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.tic = time.time()
        self.last_count = 0

    def __call__(self, param):
        if param.nbatch % self.frequent == 0:
            speed = self.frequent * self.batch_size / (time.time() - self.tic)
            logging.info('Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec',
                         param.epoch, param.nbatch, speed)
            param.eval_metric.print_log()
            self.tic = time.time()


# draw network
def draw_network(net, title, data_shape=(8, 3, 224, 224)):
    t = mx.viz.plot_network(net, title=title, shape={'data': data_shape})
    t.render()

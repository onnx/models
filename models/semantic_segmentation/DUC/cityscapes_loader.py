import atexit
import logging
import multiprocessing as mp
import random

import mxnet as mx
import numpy as np

try:
    import Queue
except ImportError:
    import queue

import utils


class CityLoader(mx.io.DataIter):
    """
    Data Loader class for Cityscapes Dataset.
    Performs loading and preparing of images from the dataset for train/val/test.
    Used in duc-validation.ipynb
    """
    def __init__(self, data_list, input_args):
        super(CityLoader, self).__init__()
        self.input_args = input_args
        self.data_list = data_list
        self.data = CityLoader.read_data(self.data_list)
        self.data_path = input_args.get('data_path', '')
        self.data_shape = input_args.get('data_shape')
        self.label_shape = input_args.get('label_shape')
        self.multi_thread = input_args.get('multi_thread', False)
        self.n_thread = input_args.get('n_thread', 7)
        self.data_name = input_args.get('data_name', ['data'])
        self.label_name = input_args.get('label_name', ['seg_loss_label'])
        self.data_loader = input_args.get('data_loader')
        self.stop_word = input_args.get('stop_word', '==STOP--')
        self.batch_size = input_args.pop('batch_size', 4)
        self.current_batch = None
        self.data_num = None
        self.current = None
        self.worker_proc = None

        if self.multi_thread:
            self.stop_flag = mp.Value('b', False)
            self.result_queue = mp.Queue(maxsize=self.batch_size*3)
            self.data_queue = mp.Queue()
            
    @staticmethod        
    def read_data(data_list):
        data = []
        with open(data_list, 'r') as f:
            for line in f:
                frags = line.strip().split('\t')
                item = list()
                item.append(frags[1])       # item[0] is image path
                item.append(frags[2])       # item[1] is label path
                if len(frags) > 3:
                    item.append(frags[3:])  # item[2] is parameters for cropping
                data.append(item)
        return data
    
    def _insert_queue(self):
        for item in self.data:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in range(self.n_thread)]

    def _thread_start(self):
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=CityLoader._worker,
                                       args=[pid,
                                             self.data_queue,
                                             self.result_queue,
                                             self.input_args,
                                             self.stop_word,
                                             self.stop_flag])
                            for pid in range(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()
        atexit.register(cleanup)

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, input_args, stop_word, stop_flag):
        count = 0
        for item in iter(data_queue.get, stop_word):
            if stop_flag == 1:
                break
            image, label = CityLoader._get_single(item, input_args)
            result_queue.put((image, label))
            count += 1

    @property
    def provide_label(self):
        return [(self.label_name[i], self.label_shape[i]) for i in range(len(self.label_name))]

    @property
    def provide_data(self):
        return [(self.data_name[i], self.data_shape[i]) for i in range(len(self.data_name))]
        
    def reset(self): 
        self.data_num = len(self.data)
        self.current = 0
        self.shuffle()
        if self.multi_thread:
            self.shutdown()
            self._insert_queue()
            self._thread_start()

    def get_batch_size(self):
        return self.batch_size

    def shutdown(self):
        if self.multi_thread:
            # clean queue
            while True:
                try:
                    self.result_queue.get(timeout=1)
                except Queue.Empty:
                    break
            while True:
                try:
                    self.data_queue.get(timeout=1)
                except Queue.Empty:
                    break
            # stop worker
            self.stop_flag = True
            if self.worker_proc:
                for i, worker in enumerate(self.worker_proc):
                    worker.join(timeout=1)
                    if worker.is_alive():
                        logging.error('worker {} is join fail'.format(i))
                        worker.terminate()

    def shuffle(self):
        random.shuffle(self.data)

    def next(self):
        if self._get_next():
            return self.current_batch
        else:
            raise StopIteration

    def _get_next(self):
        batch_size = self.batch_size
        if self.current + batch_size > self.data_num:
            return False
        xs = [np.zeros(ds) for ds in self.data_shape]
        ys = [np.zeros(ls) for ls in self.label_shape]
        cnt = 0
        for i in range(self.current, self.current + batch_size):
            if self.multi_thread:
                image, label = self.result_queue.get()
            else:
                image, label = CityLoader._get_single(self.data[i], self.input_args)
            for j in range(len(image)):
                xs[j][cnt, :, :, :] = image[j]
            for j in range(len(label)):
                ys[j][cnt, :] = label[j]
            cnt += 1
        xs = [mx.ndarray.array(x) for x in xs]
        ys = [mx.ndarray.array(y) for y in ys]
        self.current_batch = mx.io.DataBatch(data=xs, label=ys, pad=0, index=None)
        self.current += batch_size
        return True

    @staticmethod
    def _get_single(item, input_args):
        return utils.get_single_image_duc(item, input_args)

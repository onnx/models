# SPDX-License-Identifier: Apache-2.0

"""Prepare the ImageNet dataset"""
import os
import argparse
import tarfile
import pickle
import gzip
from tqdm import tqdm
from mxnet.gluon.utils import check_sha1

_TARGET_DIR = os.path.expanduser('~/.onnx/datasets/imagenet')
_TRAIN_TAR = 'ILSVRC2012_img_train.tar'
_TRAIN_TAR_SHA1 = '43eda4fe35c1705d6606a6a7a633bc965d194284'
_VAL_TAR = 'ILSVRC2012_img_val.tar'
_VAL_TAR_SHA1 = '5f3f73da3395154b60528b2b2a2caf2374f5f178'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup the ImageNet dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', required=True,
                        help="The directory that contains downloaded tar files")
    parser.add_argument('--target-dir', default=_TARGET_DIR,
                        help="The directory to store extracted images")
    parser.add_argument('--checksum', action='store_true',
                        help="If check integrity before extracting.")
    args = parser.parse_args()
    return args

def check_file(filename, checksum, sha1):
    if not os.path.exists(filename):
        raise ValueError('File not found: '+filename)
    if checksum and not check_sha1(filename, sha1):
        raise ValueError('Corrupted file: '+filename)

def extract_train(tar_fname, target_dir):
    os.makedirs(target_dir)
    with tarfile.open(tar_fname) as tar:
        print("Extracting "+tar_fname+"...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description('Extract '+class_tar.name)
            tar.extract(class_tar, target_dir)
            class_fname = os.path.join(target_dir, class_tar.name)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                f.extractall(class_dir)
            os.remove(class_fname)
            pbar.update(1)
        pbar.close()

def extract_val(tar_fname, target_dir):
    os.makedirs(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)
    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pklz')
    with gzip.open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d))
    for m in mappings:
        os.rename(os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]))

def main():
    args = parse_args()

    target_dir = os.path.expanduser(args.target_dir)
    if os.path.exists(target_dir):
        raise ValueError('Target dir ['+target_dir+'] exists. Remove it first')

    tar_dir = os.path.expanduser(args.download_dir)
    train_tar_fname = os.path.join(tar_dir, _TRAIN_TAR)
    check_file(train_tar_fname, args.checksum, _TRAIN_TAR_SHA1)
    val_tar_fname = os.path.join(tar_dir, _VAL_TAR)
    check_file(val_tar_fname, args.checksum, _VAL_TAR_SHA1)

    extract_train(train_tar_fname, os.path.join(target_dir, 'train'))
    extract_val(val_tar_fname, os.path.join(target_dir, 'val'))

if __name__ == '__main__':
    main()

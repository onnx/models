<!--- SPDX-License-Identifier: Apache-2.0 -->

# Guide for preparing ImageNet Dataset

<!-- refer to https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-horovod.html#tutorial-horovod-imagenet

Are you consolidating efforts on this approach? Is the prep similar/different?
What can be reused?

According to the horovod tutorial you should also resize the images before training. Why is that not done here?

-->

## Download
First, go to the [ImageNet download page](http://www.image-net.org/download-images) (you may need to register an account), and find the page for ILSVRC2012. Next, find and download the following two files:

|Filename                 | Size  |
|-------------------------|:------|
|ILSVRC2012_img_train.tar | 138 GB|
|ILSVRC2012_img_val.tar   | 6.3 GB|


## Setup
<!-- Isn't the assumption that they've cloned this repo and they have the files already?
If so, then update the command below to use a relative path or give directions from the root of the repo.
-->
* Download the helper script [extract_imagenet.py](extract_imagenet.py), and the validation labels [imagenet_val_maps.pklz](imagenet_val_maps.pklz).
* Place both files in the same folder.
* Run the following command:

``python extract_imagenet.py --download-dir *path to download folder* --target-dir *path to extract folder*``

Please note that the usage of the dataset must be non-commercial as per the [ImageNet license](http://www.image-net.org/download-faq)

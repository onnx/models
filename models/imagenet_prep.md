## Guidelines for preparing Imagenet dataset
### Download
First, go to the [download page](http://www.image-net.org/download-images) (you may need to register an account), and find the page for ILSVRC2012. Next, find and download the following two files:

|Filename                 | Size  |
|-------------------------|:------|
|ILSVRC2012_img_train.tar | 138 GB|
|ILSVRC2012_img_val.tar   | 6.3 GB|
### Setup
* Download helper script [extract_imagenet.py](../extract_imagenet.py) and validation image info [imagenet_val_maps.pklz](../imagenet_val_maps.pklz) and place in the same folder
* Run `python extract_imagenet.py --download-dir *path to download folder* --target-dir *path to extract folder*`


import cv2
import numpy as np
from skimage import transform as trans

def parse_lst_line(line):
  '''
  Helper function for line parsing
  '''
  vec = line.strip().split("\t")
  assert len(vec)>=3
  aligned = int(vec[0])
  image_path = vec[1]
  label = int(vec[2])
  bbox = None
  landmark = None
  if len(vec)>3:
    bbox = np.zeros( (4,), dtype=np.int32)
    for i in xrange(3,7):
      bbox[i-3] = int(vec[i])
    landmark = None
    if len(vec)>7:
      _l = []
      for i in xrange(7,17):
        _l.append(float(vec[i]))
      landmark = np.array(_l).reshape( (2,5) ).T
  return image_path, label, bbox, landmark, aligned


def read_image(img_path, **kwargs):
  '''
  Read and transpose input image
  '''
  mode = kwargs.get('mode', 'rgb')
  layout = kwargs.get('layout', 'HWC')
  # Read image (transpose if necessary)
  if mode=='gray':
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  else:
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
    if mode=='rgb':
      img = img[...,::-1]
    if layout=='CHW':
      img = np.transpose(img, (2,0,1))
  return img

def preprocess(img, bbox=None, landmark=None, **kwargs):
  '''
  Preprocess input image - returns aligned face images
  '''
  if isinstance(img, str):
    img = read_image(img, **kwargs)
  M = None
  image_size = []
  str_image_size = kwargs.get('image_size', '')
  # Assert input image shape
  if len(str_image_size)>0:
    image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size)==1:
      image_size = [image_size[0], image_size[0]]
    assert len(image_size)==2
    assert image_size[0]==112
    assert image_size[0]==112 or image_size[1]==96
  # Do alignment using landmnark points
  if landmark is not None:
    assert len(image_size)==2
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0
    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

  # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
  if M is None:
    if bbox is None: #use center crop
      det = np.zeros(4, dtype=np.int32)
      det[0] = int(img.shape[1]*0.0625)
      det[1] = int(img.shape[0]*0.0625)
      det[2] = img.shape[1] - det[0]
      det[3] = img.shape[0] - det[1]
    else:
      det = bbox
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
      ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret 
  else: #do align using landmark
    assert len(image_size)==2

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

    return warped

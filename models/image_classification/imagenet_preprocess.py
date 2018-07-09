import mxnet
from mxnet.gluon.data.vision import transforms

# Pre-processing function for ImageNet models
def preprocess(img):   
    '''
    Preprocessing required on the images for inference with mxnet gluon
    The function takes path to an image and returns processed tensor
    '''
    transform_fn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0) # batchify
    
    return img

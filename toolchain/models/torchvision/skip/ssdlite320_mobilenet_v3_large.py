# labels: test_group::turnkey name::ssdlite320_mobilenet_v3_large author::torchvision task::Computer_Vision license::bsd-3-clause
# Skip reason: Fails during the analysis stage of turnkey
"""
https://pytorch.org/vision/stable/models/ssdlite.html
"""

# Skip reason: triggers a bug in analysis where two models are discovered during
# profiling instead of one.
# Reinstate this model once https://github.com/aigdat/onnxmodelzoo/issues/239 is fixed


# Models discovered during profiling:
#
# ssdlite320_mobilenet_v3_large.py:
#         model (executed 1x - 0.06s)
#                 Model Type:     Pytorch (torch.nn.Module)
#                 Class:          SSDLiteFeatureExtractorMobileNet (<class 'torchvision.models.detection.ssdlite.SSDLiteFeatureExtractorMobileNet'>)
#                 Location:       /home/jfowers/miniconda3/envs/tkml/lib/python3.8/site-packages/torchvision/models/detection/_utils.py, line 461
#                 Parameters:     3,546,992 (13.53 MB)
#                 Input Shape:    'Positional Arg 1': (1, 3, 320, 320)
#                 Hash:           7b1ed851
#                 Build dir:      /home/jfowers/.cache/turnkey/ssdlite320_mobilenet_v3_large_torchvision_40d8a795
#
#         model (executed 1x - 0.09s)
#                 Model Type:     Pytorch (torch.nn.Module)
#                 Class:          SSD (<class 'torchvision.models.detection.ssd.SSD'>)
#                 Location:       /home/jfowers/miniconda3/envs/tkml/lib/python3.8/site-packages/torchvision/models/detection/ssdlite.py, line 331
#                 Parameters:     5,198,540 (19.83 MB)
#                 Input Shape:    'images': (1, 3, 224, 224)
#                 Hash:           40d8a795
#                 Build dir:      /home/jfowers/.cache/turnkey/ssdlite320_mobilenet_v3_large_torchvision_40d8a795

from turnkeyml.parser import parse
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)


torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
)
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)

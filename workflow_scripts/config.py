import os.path as osp

vision_dir = "vision"
classification_dir = osp.join(vision_dir, "classification")
object_detection_dir = osp.join(vision_dir, "object_detection_segmentation")

models_info = [
    # (script_path, model_name, model_zoo_path)
    ("torch_hub/alexnet.py", "alexnet_torch_hub_2891f54c", osp .join(classification_dir, "alexnet/alexnet.onnx")),
    ("torchvision/fasterrcnn_resnet50_fpn_v2.py", "fasterrcnn_resnet50_fpn_v2_torchvision_ae446d48", osp.join(object_detection_dir, "faster-rcnn/fasterrcnn_resnet50_fpn_v2.onnx")),
]

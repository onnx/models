from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image  # type: ignore [import-untyped]
from torchvision import transforms


def load_and_preprocess_image(image_file: str) -> torch.Tensor:
    """
    Load and preprocess image_file for inference test
    It works for all imagenet images
    """

    img = Image.open(image_file).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def softmax(vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Calculate softmax of a vector
    """
    e = np.exp(vector)
    res: npt.NDArray[np.float32] = e / e.sum()
    return res


def top_n_probabilities(
    res: npt.NDArray[np.float32],
    labels: List[str],
    top_n: int = 3,
    run_softmax: bool = False,
) -> List[Tuple[Any, Any]]:
    """
    Compute probabilities of top 3 classifications from res
    Inputs:
        data_in: output as 1-D numpy array from full connected layer or softmax
        run_softmax: whether or not to run softmax on data_in
    """
    indices = np.flip(np.argsort(res))
    if run_softmax:
        percentage = softmax(res) * 100
    else:
        percentage = res * 100

    print(indices[:3])
    top_n_result = [(labels[idx], percentage[idx].item()) for idx in indices[:3]]

    return top_n_result


def top3_probabilities(
    data_in: npt.NDArray[np.float32], labels: List[str], run_softmax: bool = False
) -> List[Tuple[Any, Any]]:
    """
    Helper function to get top 3 probabilities for backward compatibility
    """
    top3 = top_n_probabilities(data_in, labels, top_n=3, run_softmax=run_softmax)

    return top3


def load_labels(label_file: str) -> List[str]:
    classes_fh = open(label_file)
    labels = [line.strip() for line in classes_fh]

    classes_fh.close()
    return labels

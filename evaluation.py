from PIL import Image
from typing import List, Tuple
import torch
from tqdm.auto import tqdm
import numpy as np
import config


def get_y_score_from_training_set(
    normal_img_folder_path, transform, memory_bank, backbone
) -> List[np.ndarray]:
    """Calculate anomaly scores for the training set"""
    y_score_training: List[np.ndarray] = []
    for pth in tqdm(normal_img_folder_path.iterdir(), leave=False):
        data = transform(Image.open(pth)).to(config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            features = backbone(data)
        distances = torch.cdist(features, memory_bank, p=2.0)
        s_star = torch.max(torch.min(distances, dim=1)[0])
        y_score_training.append(s_star.cpu().numpy())
    return y_score_training


def get_y_score_and_true_from_testing(
    transform, backbone, memory_bank, testing_folder_path
) -> Tuple[List[float], List[int]]:
    """Calculate y_score and y_true for the testing set

    :Example:
    >>>
    testing_folder_path: "mvtec/carpet/test"

    """
    y_score: List[float] = []
    y_true: List[int] = []

    defect_types = [f.parts[-1] for f in testing_folder_path.iterdir() if f.is_dir()]

    for defect_type in defect_types:
        folder_path = testing_folder_path / defect_type

        for pth in tqdm(folder_path.iterdir(), leave=False):
            class_label = pth.parts[-2]
            with torch.no_grad():
                test_image = transform(Image.open(pth)).to(config.DEVICE).unsqueeze(0)
                features = backbone(test_image)
            distances = torch.cdist(features, memory_bank, p=2.0)
            s_star = torch.max(torch.min(distances, dim=1)[0])
            y_score.append(s_star.cpu().numpy().item())
            y_true.append(0 if class_label == "good" else 1)

    return y_score, y_true

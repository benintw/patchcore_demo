from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import config


def create_memory_bank(
    backbone: nn.Module,
    normal_folder_path: Path,
    transform: torchvision.transforms.Compose,
) -> torch.Tensor:
    """
    Creates memory bank from nominal data
    """

    memory_bank = []

    for pth in tqdm(normal_folder_path.iterdir(), leave=False):

        with torch.no_grad():
            data = transform(Image.open(pth)).to(config.DEVICE).unsqueeze(0)
            features = backbone(data)
            memory_bank.append(features.cpu().detach())

    memory_bank = torch.cat(memory_bank, dim=0).to(config.DEVICE)

    """Only select 10% of total patches to avoid long inference time and computation"""
    selected_indices = np.random.choice(
        len(memory_bank), size=len(memory_bank) // 10, replace=False
    )
    memory_bank = memory_bank[selected_indices]

    return memory_bank

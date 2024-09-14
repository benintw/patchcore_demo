import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import config


## Load a pretrained Resnet Model
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(ResNetFeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, input):

        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        self.avg = torch.nn.AvgPool2d(3, stride=1)

        fmap_size = self.features[0].shape[-2]  # Feature map sizes h, w
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [
            self.resize(self.avg(fmap.to("cpu"))).to(config.DEVICE)
            for fmap in self.features
        ]
        patch = torch.cat(resized_maps, 1)  # Merge the resized feature maps
        patch = patch.reshape(patch.shape[1], -1).T  # Create a column tensor

        return patch

from __future__ import annotations

import timm
import torch.nn as nn


def create_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )

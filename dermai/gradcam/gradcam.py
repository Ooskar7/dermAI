from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last_conv


class GradCAM:
    """Minimal Grad-CAM implementation for CNN classifiers."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or find_last_conv_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_tensor = (weights * self.activations).sum(dim=1, keepdim=True)
        cam_tensor = F.relu(cam_tensor)
        cam_tensor = F.interpolate(
            cam_tensor,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam_tensor.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam


def overlay_heatmap(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.42,
    colormap: Callable | None = None,
) -> Image.Image:
    image = image.convert("RGB")
    colormap = colormap or colormaps["jet"]
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    cam_resized = np.asarray(cam_image).astype(np.float32) / 255.0
    heatmap = colormap(cam_resized)[..., :3]
    base = np.asarray(image).astype(np.float32) / 255.0
    overlay = np.clip((1.0 - alpha) * base + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray(np.uint8(overlay * 255))


def generate_gradcam_overlay(
    model: nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
    target_class: int | None = None,
    alpha: float = 0.42,
) -> Image.Image:
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    gradcam = GradCAM(model)
    try:
        cam = gradcam(input_tensor, target_class=target_class)
    finally:
        gradcam.close()
    return overlay_heatmap(image, cam, alpha=alpha)

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image

from dermai.data.ham10000 import get_inference_transform
from dermai.labels import HAM10000_LABELS, HAM10000_LABEL_DESCRIPTIONS
from dermai.models.efficientnet import create_efficientnet_b0


def choose_device(requested_device: str | torch.device | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(image: str | Path | BinaryIO | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as pil_image:
        return pil_image.convert("RGB")


def _extract_state_dict(checkpoint: dict) -> dict:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


class DermAIInference:
    """Loads a trained EfficientNet checkpoint and predicts one uploaded image."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = choose_device(device)
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names = self.checkpoint.get("class_names", HAM10000_LABELS)
        self.image_size = int(self.checkpoint.get("image_size", 224))
        self.transform = get_inference_transform(self.image_size)

        self.model = create_efficientnet_b0(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(_extract_state_dict(self.checkpoint))
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: str | Path | BinaryIO | Image.Image, top_k: int = 7) -> list[dict]:
        pil_image = load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        top_k = min(top_k, len(self.class_names))
        values, indices = torch.topk(probabilities, k=top_k)

        results = []
        for probability, index in zip(values.cpu().tolist(), indices.cpu().tolist(), strict=False):
            class_name = self.class_names[index]
            results.append(
                {
                    "class_name": class_name,
                    "description": HAM10000_LABEL_DESCRIPTIONS.get(class_name, class_name),
                    "probability": float(probability),
                    "class_index": int(index),
                }
            )
        return results

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dermai.labels import HAM10000_LABELS, LABEL_TO_INDEX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ISIC2018_ONE_HOT_COLUMNS = {
    "AKIEC": "akiec",
    "BCC": "bcc",
    "BKL": "bkl",
    "DF": "df",
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
}


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    return get_val_transform(image_size=image_size)


def normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Normalize HAM10000 or ISIC 2018 task 3 metadata to image_id/dx columns."""

    metadata = metadata.copy()
    if {"image_id", "dx"}.issubset(metadata.columns):
        metadata["image_id"] = metadata["image_id"].astype(str)
        metadata["dx"] = metadata["dx"].astype(str).str.lower()
        return metadata

    one_hot_columns = list(ISIC2018_ONE_HOT_COLUMNS)
    if "image" in metadata.columns and set(one_hot_columns).issubset(metadata.columns):
        label_matrix = metadata[one_hot_columns].astype(float)
        row_sums = label_matrix.sum(axis=1)
        invalid_rows = row_sums[row_sums != 1.0]
        if not invalid_rows.empty:
            raise ValueError(
                "ISIC 2018 metadata must have exactly one positive diagnosis column per row. "
                f"Invalid row count: {len(invalid_rows)}"
            )

        metadata["image_id"] = metadata["image"].astype(str)
        metadata["dx"] = label_matrix.idxmax(axis=1).map(ISIC2018_ONE_HOT_COLUMNS)
        return metadata

    raise ValueError(
        "Unsupported metadata format. Expected HAM10000 columns image_id/dx or ISIC 2018 "
        "columns image plus MEL,NV,BCC,AKIEC,BKL,DF,VASC."
    )


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for HAM10000/ISIC 2018 task 3 metadata and image folders."""

    def __init__(
        self,
        metadata_csv: str | Path | None = None,
        image_dirs: str | Path | Iterable[str | Path] | None = None,
        transform: transforms.Compose | None = None,
        label_to_index: dict[str, int] | None = None,
        metadata_df: pd.DataFrame | None = None,
    ) -> None:
        if metadata_csv is None and metadata_df is None:
            raise ValueError("Provide either metadata_csv or metadata_df.")
        if image_dirs is None:
            raise ValueError("Provide at least one image directory.")

        raw_metadata = pd.read_csv(metadata_csv) if metadata_df is None else metadata_df.copy()
        self.metadata = normalize_metadata(raw_metadata)
        required_columns = {"image_id", "dx"}
        missing_columns = required_columns.difference(self.metadata.columns)
        if missing_columns:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

        self.label_to_index = label_to_index or LABEL_TO_INDEX
        self.class_names = [
            label for label, _ in sorted(self.label_to_index.items(), key=lambda item: item[1])
        ]
        unknown_labels = sorted(set(self.metadata["dx"]) - set(self.label_to_index))
        if unknown_labels:
            raise ValueError(f"Unknown diagnosis labels in metadata: {unknown_labels}")

        self.image_dirs = self._normalize_image_dirs(image_dirs)
        self.image_paths = self._index_image_paths(self.image_dirs)
        self.transform = transform or get_val_transform()
        self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_id = str(row["image_id"])
        image_path = self._resolve_image_path(image_id)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        label = self.label_to_index[row["dx"]]
        return image_tensor, label, image_id

    @staticmethod
    def _normalize_image_dirs(
        image_dirs: str | Path | Iterable[str | Path],
    ) -> list[Path]:
        if isinstance(image_dirs, (str, Path)):
            paths = [Path(image_dirs)]
        else:
            paths = [Path(path) for path in image_dirs]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Image directories do not exist: {missing}")
        return paths

    @staticmethod
    def _index_image_paths(image_dirs: list[Path]) -> dict[str, Path]:
        image_paths: dict[str, Path] = {}
        extensions = {".jpg", ".jpeg", ".png"}
        for image_dir in image_dirs:
            for path in image_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    image_paths[path.stem] = path
        return image_paths

    def _resolve_image_path(self, image_id: str) -> Path:
        if image_id in self.image_paths:
            return self.image_paths[image_id]
        tried = []
        for image_dir in self.image_dirs:
            for extension in (".jpg", ".jpeg", ".png"):
                candidate = image_dir / f"{image_id}{extension}"
                tried.append(str(candidate))
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"Image for image_id={image_id!r} was not found. Tried: {tried[:6]}")

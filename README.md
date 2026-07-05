# DermAI MVP

DermAI is a Streamlit-based educational prototype for skin lesion image recognition and
explainability. It uses an EfficientNet-B0 classifier trained on HAM10000/ISIC-style
dermoscopic lesion images and generates Grad-CAM overlays to visualize model attention.

This project is not a medical device and must not be used to diagnose disease, rule out
cancer, choose treatment, or replace a dermatologist. The app is intended for educational
experiments with image classification only.

## Project layout

```text
dermai/
  app/              Streamlit UI
  data/             HAM10000/ISIC metadata loading and transforms
  gradcam/          Grad-CAM overlay generation
  inference/        Checkpoint loading and single-image prediction
  models/           EfficientNet model factory
  rag/              Local guidance/reporting experiments, not used by the current app
  training/         Training entrypoint and metrics
scripts/            Utility scripts
outputs/            Training outputs such as checkpoints and history.csv
```

## Setup

```bash
python -m venv .dermai
source .dermai/bin/activate
pip install -r requirements.txt
```

If you already installed dependencies before the NumPy pin was added, run:

```bash
pip install --force-reinstall "numpy>=1.26,<2"
pip install -r requirements.txt
```

Check that the active Python environment is coherent:

```bash
python scripts/check_environment.py
```

## Dataset expectations

The training script accepts either HAM10000 metadata or ISIC 2018 task 3 ground truth.

Supported metadata formats:

- HAM10000 metadata with `image_id` and `dx` columns
- ISIC 2018 task 3 ground truth with `image` plus one-hot columns
  `MEL,NV,BCC,AKIEC,BKL,DF,VASC`

Images are expected as `.jpg`, `.jpeg`, or `.png` files in one or more image folders.
Image filenames should match the metadata image identifier, such as:

```text
dataset/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
    ISIC_0024306.jpg
  HAM10000_images_part_2/
    ISIC_0034310.jpg
```

or:

```text
ISIC2018_Task3_Training_GroundTruth/
  ISIC2018_Task3_Training_GroundTruth.csv
ISIC2018_Task3_Training_Input/
  ISIC_0024306.jpg
  ISIC_0024307.jpg
```

## Train

For ISIC 2018 files:

```bash
python -m dermai.training.train \
  --metadata-csv /path/to/ISIC2018_Task3_Training_GroundTruth.csv \
  --image-dir /path/to/ISIC2018_Task3_Training_Input \
  --output-dir outputs/efficientnet_b0 \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 2
```

For HAM10000 metadata with separate image folders:

```bash
python -m dermai.training.train \
  --metadata-csv /path/to/HAM10000_metadata.csv \
  --image-dir /path/to/HAM10000_images_part_1 \
  --image-dir /path/to/HAM10000_images_part_2 \
  --output-dir outputs/efficientnet_b0 \
  --epochs 10 \
  --batch-size 32
```

The script saves `best.pt`, `last.pt`, and `history.csv` in the output directory.
Training uses ImageNet transfer learning by default, class-weighted cross entropy, and
validation after each epoch. If pretrained weight download fails, retry with
`--no-pretrained`; training from scratch usually needs more epochs.

## Run the app

```bash
streamlit run app.py
```

The default checkpoint path is `outputs/efficientnet_b0/best.pt`. In the Streamlit
sidebar, choose a page, confirm the checkpoint status, select how many prediction
classes to display, upload a lesion image, and click `Analyze image`.

The app includes:

- Introduction: purpose, supported classes, and safety limits
- Model Training: data pipeline, model details, training process, and `history.csv` charts
- DermAI Prediction: image upload, class probabilities, prediction table, and Grad-CAM overlay

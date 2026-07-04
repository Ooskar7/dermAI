# DermAI MVP

Streamlit-based MVP for educational dermatology image triage experiments using HAM10000, EfficientNet-B0, Grad-CAM, and local ChromaDB retrieval over curated safety guidance.

This project is not a medical device and must not be used to diagnose disease. Generated reports are intentionally cautious and educational.

## Project layout

```text
dermai/
  app/              Streamlit UI
  data/             HAM10000 dataset loading and transforms
  gradcam/          Grad-CAM overlay generation
  inference/        Checkpoint loading and single-image prediction
  models/           EfficientNet model factory
  rag/              ChromaDB retrieval and safe report generation
  training/         Training entrypoint and metrics
data/guidance/      Curated dermatology safety guidance text files
scripts/            Utility scripts
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

## HAM10000 data

Supported metadata formats:

- HAM10000 metadata with `image_id` and `dx`
- ISIC 2018 task 3 ground truth with `image` and one-hot columns `MEL,NV,BCC,AKIEC,BKL,DF,VASC`

Images are expected as `.jpg`, `.jpeg`, or `.png` files in one or more image folders.

## Train

For the ISIC 2018 files in `/Users/oscarsegura/dermAI_data`:

```bash
python -m dermai.training.train \
  --metadata-csv /Users/oscarsegura/dermAI_data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv \
  --image-dir /Users/oscarsegura/dermAI_data/ISIC2018_Task3_Training_Input \
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

The script saves `best.pt`, `last.pt`, and `history.csv` in the output directory. The loss uses class weights derived from the training split.

If pretrained weight download fails, retry with `--no-pretrained`. This will train from scratch and usually needs more epochs.

## Build the guidance index

```bash
python scripts/build_rag_index.py \
  --guidance-dir data/guidance \
  --persist-dir data/chroma
```

The Streamlit app can also build or refresh the index from the sidebar.

## Run the app

```bash
python -m streamlit run app.py
```

Provide a trained checkpoint path in the sidebar, upload a skin lesion image, enter symptoms/context, and click Analyze.

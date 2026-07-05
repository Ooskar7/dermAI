from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from dermai.gradcam.gradcam import generate_gradcam_overlay
from dermai.inference.inference import DermAIInference, load_image
from dermai.labels import HAM10000_LABEL_DESCRIPTIONS, HAM10000_LABELS

DEFAULT_CHECKPOINT_PATH = "outputs/efficientnet_b0/best.pt"
DEFAULT_HISTORY_PATH = Path("outputs/efficientnet_b0/history.csv")
PAGE_INTRODUCTION = "Introduction"
PAGE_MODEL_TRAINING = "Model Training"
PAGE_DERMAI_PREDICTION = "DermAI Prediction"


@st.cache_resource(show_spinner=False)
def load_inference(checkpoint_path: str) -> DermAIInference:
    return DermAIInference(checkpoint_path)


def configure_page() -> None:
    st.set_page_config(page_title="DermAI MVP", page_icon=":mag:", layout="wide")


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("DermAI")
        page = st.radio(
            "Navigation",
            [PAGE_INTRODUCTION, PAGE_MODEL_TRAINING, PAGE_DERMAI_PREDICTION],
        )

        st.divider()
        st.header("Settings")
        checkpoint_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT_PATH)
        top_k = st.slider("Prediction classes", min_value=3, max_value=7, value=7)

        checkpoint_exists = Path(checkpoint_path).exists()
        if checkpoint_exists:
            st.success("Checkpoint found")
        else:
            st.error("Checkpoint not found")
        st.caption(checkpoint_path)

    return {
        "page": page,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
        "top_k": top_k,
    }


def render_introduction_page() -> None:
    st.title("DermAI MVP")
    st.caption("Educational skin lesion image recognition and explainability prototype.")

    st.markdown(
        """
        DermAI is an educational prototype for skin lesion image recognition. It classifies
        dermoscopic lesion images into the seven HAM10000/ISIC classes and uses Grad-CAM to
        visualize which image regions most influenced the model's top prediction.

        DermAI is not a diagnostic medical device. It does not replace a dermatologist,
        a clinical examination, biopsy, or professional medical judgment.
        """
    )

    left, right = st.columns(2)
    with left:
        st.subheader("What DermAI does")
        st.markdown(
            """
            - Accepts one dermoscopic skin lesion image.
            - Runs an EfficientNet-B0 image classifier trained for seven lesion categories.
            - Shows class probabilities for the selected number of predictions.
            - Generates a Grad-CAM overlay to make model attention easier to inspect.
            """
        )
    with right:
        st.subheader("What DermAI does not do")
        st.markdown(
            """
            - Diagnose cancer or any other disease.
            - Assess symptoms, lesion history, or patient risk factors.
            - Replace clinical judgment, dermoscopy expertise, biopsy, or follow-up care.
            - Provide treatment recommendations or urgent triage decisions.
            """
        )

    st.subheader("HAM10000/ISIC Classes")
    class_rows = [
        {"Class": class_name, "Description": HAM10000_LABEL_DESCRIPTIONS[class_name]}
        for class_name in HAM10000_LABELS
    ]
    st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")

    render_static_safety_notice()


def render_training_page() -> None:
    st.title("Model Training")
    st.caption("Training workflow for the EfficientNet-B0 lesion image classifier.")

    st.subheader("Data Pipeline")
    st.markdown(
        """
        - Metadata is loaded from HAM10000 or ISIC 2018 task 3 CSV files.
        - Metadata is normalized into `image_id` and `dx` columns with seven supported labels.
        - Images are resolved from one or more image directories.
        - The dataset is split into train and validation subsets, with stratification when possible.
        - Training transforms resize images and apply flips, rotation, color jitter, tensor conversion,
          and ImageNet normalization. Validation and inference use deterministic resize and normalization.
        """
    )

    st.subheader("Model")
    st.markdown(
        """
        - EfficientNet-B0 is created with `timm`.
        - ImageNet transfer learning is used by default during training.
        - The classifier head is replaced with a seven-class HAM10000/ISIC output layer.
        """
    )

    st.subheader("Training Loop")
    st.markdown(
        """
        - PyTorch `DataLoader` objects feed train and validation batches.
        - Class-weighted `CrossEntropyLoss` helps compensate for class imbalance.
        - Validation runs after every epoch.
        - `best.pt` stores the best validation macro F1 checkpoint and `last.pt` stores the latest epoch.
        - `history.csv` records loss, accuracy, macro F1, and per-class validation recall metrics.
        """
    )

    render_training_history(DEFAULT_HISTORY_PATH)

    st.subheader("Limitations")
    st.markdown(
        """
        HAM10000/ISIC class distributions are imbalanced, so rare lesion categories can be harder to
        learn and evaluate reliably. The model is trained for dermoscopic image inputs, not general
        phone photos or clinical context. Dataset collection patterns can introduce bias across skin
        type, acquisition device, site, and labeling practice. This prototype has not undergone
        clinical validation and must not be used for diagnosis or treatment decisions.
        """
    )


def render_training_history(history_path: Path) -> None:
    st.subheader("Training History")
    if not history_path.exists():
        st.warning(
            "Training history will appear here after training creates "
            f"`{history_path.as_posix()}`."
        )
        return

    history = pd.read_csv(history_path)
    if history.empty:
        st.warning(f"`{history_path.as_posix()}` exists but does not contain any training rows.")
        return

    if {"epoch", "train_loss", "val_loss"}.issubset(history.columns):
        loss_data = history.melt(
            id_vars=["epoch"],
            value_vars=["train_loss", "val_loss"],
            var_name="metric",
            value_name="loss",
        )
        loss_chart = (
            alt.Chart(loss_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:O", title="Epoch"),
                y=alt.Y("loss:Q", title="Loss"),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["epoch:O", "metric:N", alt.Tooltip("loss:Q", format=".4f")],
            )
            .properties(height=300)
        )
        st.altair_chart(loss_chart, width="stretch")

    metric_columns = [
        column for column in ["val_accuracy", "val_macro_f1"] if column in history.columns
    ]
    if "epoch" in history.columns and metric_columns:
        metric_data = history.melt(
            id_vars=["epoch"],
            value_vars=metric_columns,
            var_name="metric",
            value_name="score",
        )
        metric_chart = (
            alt.Chart(metric_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:O", title="Epoch"),
                y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["epoch:O", "metric:N", alt.Tooltip("score:Q", format=".4f")],
            )
            .properties(height=300)
        )
        st.altair_chart(metric_chart, width="stretch")

    final_metrics = history.tail(1).T.reset_index()
    final_metrics.columns = ["Metric", "Value"]
    st.markdown("Final recorded metrics")
    st.dataframe(final_metrics, hide_index=True, width="stretch")


def render_prediction_page(settings: dict[str, Any]) -> None:
    st.title("DermAI Prediction")
    st.caption("Upload one lesion image to view model predictions and a Grad-CAM explanation.")

    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    analyze = st.button("Analyze image", type="primary", disabled=uploaded_file is None)

    if not analyze:
        render_static_safety_notice()
        return

    checkpoint_path = settings["checkpoint_path"]
    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.info("Train a model first, then set a valid checkpoint path in the sidebar.")
        return

    try:
        image = load_image(uploaded_file)
        inference = load_inference(checkpoint_path)
        probabilities = inference.predict(image, top_k=settings["top_k"])
        target_class = int(probabilities[0]["class_index"]) if probabilities else None
        overlay = generate_gradcam_overlay(
            inference.model,
            image,
            inference.transform,
            inference.device,
            target_class=target_class,
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded Image")
        st.image(image, width="stretch")
    with right:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay, width="stretch")

    st.subheader("Prediction Probabilities")
    probability_df = pd.DataFrame(probabilities)
    render_probability_chart(probability_df)
    render_prediction_table(probability_df)

    st.subheader("How to read this result")
    st.markdown(
        """
        The highest probability is the model's top image-only classification for this upload.
        Probabilities describe the model output across the seven training classes, not clinical
        certainty. The Grad-CAM overlay highlights regions that contributed to the selected class,
        but it does not prove that the model focused on medically meaningful features.
        """
    )
    render_static_safety_notice()


def render_probability_chart(probability_df: pd.DataFrame) -> None:
    if probability_df.empty:
        st.info("No probabilities were returned.")
        return

    chart_df = probability_df.copy()
    chart_df["probability_percent"] = chart_df["probability"] * 100
    chart_df["label"] = chart_df["description"].fillna(chart_df["class_name"])

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort="-x", title="Class"),
            x=alt.X(
                "probability_percent:Q",
                title="Probability (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("class_name:N", title="Class"),
                alt.Tooltip("description:N", title="Description"),
                alt.Tooltip("probability_percent:Q", title="Probability (%)", format=".2f"),
            ],
        )
        .properties(height=max(220, 34 * len(chart_df)))
    )
    st.altair_chart(chart, width="stretch")


def render_prediction_table(probability_df: pd.DataFrame) -> None:
    if probability_df.empty:
        return

    table_df = probability_df.copy()
    table_df["Probability (%)"] = (table_df["probability"] * 100).round(2)
    table_df = table_df.rename(
        columns={
            "class_name": "Class",
            "description": "Description",
        }
    )
    st.dataframe(
        table_df[["Class", "Description", "Probability (%)"]],
        hide_index=True,
        width="stretch",
    )


def render_static_safety_notice() -> None:
    st.warning(
        "Safety notice: DermAI is an educational prototype only. It is not a diagnostic medical "
        "device and must not be used to decide whether a lesion is benign or malignant. Seek care "
        "from a qualified clinician for changing, bleeding, painful, rapidly growing, or otherwise "
        "concerning skin lesions."
    )


def main() -> None:
    configure_page()
    settings = render_sidebar()

    if settings["page"] == PAGE_INTRODUCTION:
        render_introduction_page()
    elif settings["page"] == PAGE_MODEL_TRAINING:
        render_training_page()
    else:
        render_prediction_page(settings)


if __name__ == "__main__":
    main()

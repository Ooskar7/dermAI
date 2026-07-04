from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from dermai.gradcam.gradcam import generate_gradcam_overlay
from dermai.inference.inference import DermAIInference, load_image
from dermai.rag.report import generate_triage_report
from dermai.rag.retriever import DermGuidanceRetriever


@st.cache_resource(show_spinner=False)
def load_inference(checkpoint_path: str) -> DermAIInference:
    return DermAIInference(checkpoint_path)


@st.cache_resource(show_spinner=False)
def load_retriever(persist_dir: str) -> DermGuidanceRetriever:
    return DermGuidanceRetriever(persist_dir=persist_dir)


def main() -> None:
    st.set_page_config(page_title="DermAI MVP", page_icon=":mag:", layout="wide")
    st.title("DermAI MVP")
    st.caption("Educational image triage prototype. Not for diagnosis or treatment decisions.")

    with st.sidebar:
        st.header("Settings")
        checkpoint_path = st.text_input("Checkpoint path", value="outputs/efficientnet_b0/best.pt")
        guidance_dir = st.text_input("Guidance directory", value="data/guidance")
        chroma_dir = st.text_input("ChromaDB directory", value="data/chroma")
        top_k = st.slider("Prediction classes", min_value=3, max_value=7, value=7)

        if st.button("Build or refresh guidance index"):
            retriever = load_retriever(chroma_dir)
            try:
                count = retriever.build_index(guidance_dir)
                st.success(f"Indexed {count} guidance chunks.")
            except Exception as exc:
                st.error(f"Could not build index: {exc}")

    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    symptoms = st.text_area(
        "Symptoms and context",
        placeholder="Example: changing mole on shoulder, itchy for two weeks, no bleeding...",
        height=120,
    )

    analyze = st.button("Analyze", type="primary", disabled=uploaded_file is None)
    if not analyze:
        return

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.info("Train a model first, then set the checkpoint path in the sidebar.")
        return

    try:
        image = load_image(uploaded_file)
        inference = load_inference(checkpoint_path)
        probabilities = inference.predict(image, top_k=top_k)
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

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("Prediction Probabilities")
    probability_df = pd.DataFrame(probabilities)
    probability_df["probability_percent"] = probability_df["probability"] * 100
    st.bar_chart(probability_df.set_index("class_name")["probability"])
    st.dataframe(
        probability_df[["class_name", "description", "probability_percent"]].rename(
            columns={"probability_percent": "probability (%)"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    retriever = load_retriever(chroma_dir)
    if retriever.collection.count() == 0:
        try:
            retriever.build_index(guidance_dir)
        except Exception:
            pass

    query = " ".join(
        [
            symptoms,
            " ".join(item["description"] for item in probabilities[:3]),
            "skin lesion warning signs dermatology triage",
        ]
    )
    retrieved_chunks = retriever.retrieve(query, top_k=3)
    report = generate_triage_report(probabilities, symptoms, retrieved_chunks)

    st.subheader("Educational Triage Report")
    st.markdown(report)


if __name__ == "__main__":
    main()

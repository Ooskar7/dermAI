from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import HashingVectorizer


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    distance: float | None = None


class HashingEmbedding:
    """Stateless local text embedding for ChromaDB examples."""

    def __init__(self, n_features: int = 384) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            stop_words="english",
        )

    def embed(self, documents: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(documents)
        return matrix.astype("float32").toarray().tolist()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= chunk_size:
        return [normalized] if normalized else []

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


class DermGuidanceRetriever:
    def __init__(
        self,
        persist_dir: str | Path = "data/chroma",
        collection_name: str = "dermatology_guidance",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding = HashingEmbedding()

    def build_index(self, guidance_dir: str | Path) -> int:
        guidance_path = Path(guidance_dir)
        if not guidance_path.exists():
            raise FileNotFoundError(f"Guidance directory does not exist: {guidance_path}")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for file_path in sorted(guidance_path.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            for index, chunk in enumerate(chunk_text(text)):
                ids.append(f"{file_path.stem}-{index}")
                documents.append(chunk)
                metadatas.append({"source": file_path.name, "chunk": index})

        if not documents:
            return 0

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=self.embedding.embed(documents),
        )
        return len(documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=self.embedding.embed([query]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            chunks.append(
                RetrievedChunk(
                    text=document,
                    source=str(metadata.get("source", "unknown")),
                    distance=float(distance) if distance is not None else None,
                )
            )
        return chunks

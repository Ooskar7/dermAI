from __future__ import annotations

import argparse
from pathlib import Path

from dermai.rag.retriever import DermGuidanceRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local ChromaDB dermatology guidance index.")
    parser.add_argument("--guidance-dir", default=Path("data/guidance"), type=Path)
    parser.add_argument("--persist-dir", default=Path("data/chroma"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = DermGuidanceRetriever(persist_dir=args.persist_dir)
    count = retriever.build_index(args.guidance_dir)
    print(f"Indexed {count} guidance chunks into {args.persist_dir}")


if __name__ == "__main__":
    main()

"""RAG and report-generation utilities."""

__all__ = ["DermGuidanceRetriever", "RetrievedChunk", "generate_triage_report"]


def __getattr__(name: str):
    if name == "generate_triage_report":
        from dermai.rag.report import generate_triage_report

        return generate_triage_report
    if name in {"DermGuidanceRetriever", "RetrievedChunk"}:
        from dermai.rag.retriever import DermGuidanceRetriever, RetrievedChunk

        return {"DermGuidanceRetriever": DermGuidanceRetriever, "RetrievedChunk": RetrievedChunk}[name]
    raise AttributeError(name)

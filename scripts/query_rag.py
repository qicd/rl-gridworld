from __future__ import annotations

try:
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.embeddings import resolve_embed_model
    def _get_embed_model():
        return resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    from llama_index import StorageContext, load_index_from_storage  # type: ignore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
    def _get_embed_model():
        return HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

from pathlib import Path
import sys

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)


def main(query: str) -> None:
    storage_context = StorageContext.from_defaults(persist_dir="data/index")
    index = load_index_from_storage(storage_context, embed_model=_get_embed_model())
    # Disable default LLM to avoid OpenAI API key requirement
    try:
        from llama_index.core import Settings  # type: ignore
        Settings.llm = None
    except Exception:
        pass
    query_engine = index.as_query_engine(similarity_top_k=5)
    print(query_engine.query(query))


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "如何在本项目里开启人类反馈并可视化训练曲线？"
    main(q)



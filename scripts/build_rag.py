from __future__ import annotations

from pathlib import Path

# Ensure project root on sys.path
import sys
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

# LlamaIndex imports with version-compatible fallback
try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.embeddings import resolve_embed_model
    def _get_embed_model():
        return resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    from llama_index import SimpleDirectoryReader, VectorStoreIndex  # type: ignore
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Cannot import LlamaIndex embeddings. Please upgrade llama-index.") from e
    def _get_embed_model():
        return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def main() -> None:
    docs_dir = Path("docs")
    persist_dir = Path("data/index")
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = SimpleDirectoryReader(str(docs_dir)).load_data()
    embed_model = _get_embed_model()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"Index saved to {persist_dir}")


if __name__ == "__main__":
    main()



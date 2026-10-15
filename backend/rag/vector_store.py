from pathlib import Path

import chromadb

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from backend.rag.document_loader import load_context_notes
from functools import lru_cache
PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chroma_context_notes"
)

COLLECTION_NAME = "freight_context_notes"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    return FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_vector_store():
    documents = load_context_notes()

    VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_PATH)
    )

    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(
            name=COLLECTION_NAME
        )

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )

    if documents:
        vector_store.add_documents(
            documents=documents,
            ids=[
                document.metadata["note_id"]
                for document in documents
            ],
        )

    return vector_store

@lru_cache(maxsize=1)
def get_vector_store():
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            "Vector store directory not found. "
            "Run the ingestion command first."
        )

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_PATH)
    )

    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME not in existing_collections:
        raise FileNotFoundError(
            "Context-note collection not found. "
            "Run the ingestion command first."
        )

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


if __name__ == "__main__":
    store = build_vector_store()

    print("Context-note ingestion completed.")
    print(f"Stored documents: {store._collection.count()}")

    results = store.similarity_search(
        "What could explain increased shipping costs?",
        k=3,
    )

    print("\nSample retrieval results:")

    for document in results:
        print(
            document.metadata["note_id"],
            document.metadata["applies_to"],
        )
        print(document.page_content)
        print("-" * 50)
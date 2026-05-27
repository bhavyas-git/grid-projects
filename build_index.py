import argparse

from src.config import PDF_PATH, PHASE6_FAISS_INDEX_PATH, PHASE6_QDRANT_COLLECTION
from src.embeddings import get_embeddings
from src.ingestion import load_and_chunk
from src.multimodal_ingestion import load_all_documents
from src.phase6_ingestion import build_phase6_patch_documents
from src.vectorstore import create_vectorstore, save_vectorstore


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS and/or Qdrant indexes for the IFC RAG project.")
    parser.add_argument(
        "--backend",
        choices=["faiss", "qdrant", "both"],
        default="both",
        help="Vector store backend to build.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip extracting and captioning PDF images.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip extracting PDF tables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    text_chunks = load_and_chunk(PDF_PATH)
    documents = load_all_documents(
        text_chunks,
        include_images=not args.skip_images,
        include_tables=not args.skip_tables,
        pdf_path=PDF_PATH,
    )
    embeddings = get_embeddings()
    backends = ["faiss", "qdrant"] if args.backend == "both" else [args.backend]

    for backend in backends:
        vectorstore = create_vectorstore(documents, embeddings, backend=backend)
        if backend == "faiss":
            save_vectorstore(vectorstore)
        print(f"Built {backend} index with {len(documents)} documents.")

    patch_documents = build_phase6_patch_documents(PDF_PATH)
    for backend in backends:
        patch_vectorstore = create_vectorstore(
            patch_documents,
            embeddings,
            backend=backend,
            collection_name=PHASE6_QDRANT_COLLECTION,
        )
        if backend == "faiss":
            save_vectorstore(patch_vectorstore, path=PHASE6_FAISS_INDEX_PATH)
        print(f"Built Phase 6 {backend} patch index with {len(patch_documents)} patches.")

    print("✅ Index build complete.")


if __name__ == "__main__":
    main()

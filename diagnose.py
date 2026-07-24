"""
Diagnostic script - run this directly to see what's actually happening at each
stage of the pipeline, without going through Streamlit.

Usage:
    python diagnose.py "path/to/your/file.pdf" "your question here"

Works with any supported file type (.pdf, .docx, .xlsx, .html, .xml).
"""
import sys
import shutil

from utils.document_loader import load_document
from utils.chunking import chunk_documents
from utils.vector_store import VectorStore


def main():
    if len(sys.argv) < 3:
        print('Usage: python diagnose.py "path/to/your/file" "your question"')
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    print("=" * 70)
    print("STEP 1: Text extraction")
    print("=" * 70)
    sections = load_document(file_path)
    print(f"Extracted {len(sections)} section(s) with text.\n")
    for s in sections[:2]:
        print(f"--- {s['location']} (first 300 chars) ---")
        print(s["text"][:300])
        print()

    if not sections:
        print("!! No text extracted at all. If this is a PDF, it's likely scanned/image-only.")
        print("!! That's the root cause - you'd need OCR to handle this file.")
        return

    print("=" * 70)
    print("STEP 2: Chunking")
    print("=" * 70)
    chunks = chunk_documents(sections, source_name="diagnostic", chunk_size=1000, chunk_overlap=200)
    print(f"Created {len(chunks)} chunks.\n")
    for c in chunks[:3]:
        print(f"--- Chunk {c['id']} ({c['location']}) ---")
        print(c["text"][:300])
        print()

    print("=" * 70)
    print("STEP 3: Embedding + retrieval (fresh, temporary collection)")
    print("=" * 70)
    # use a throwaway dir so this never collides with your real chroma_db
    test_dir = "chroma_db_diagnostic"
    vs = VectorStore(persist_dir=test_dir)
    vs.get_or_create_collection(name="diagnostic_test")
    vs.clear_collection()
    vs.add_chunks(chunks)

    results = vs.query(question, top_k=4)
    print(f"Question: {question}\n")
    for i, (doc, meta, dist) in enumerate(results, 1):
        print(f"--- Match {i} | {meta.get('location')} | distance {dist:.4f} ---")
        print(doc[:300])
        print()

    print("=" * 70)
    print("How to read this:")
    print("- If STEP 1 text looks garbled/empty -> extraction problem")
    print("- If STEP 2 chunks look fine but STEP 3 matches are unrelated to")
    print("  the question -> embedding model problem (check torch/sentence-transformers)")
    print("- Lower distance = more similar. If ALL distances are high (>1.0-1.5")
    print("  depending on scale) even for an obviously relevant question, the")
    print("  embedding model likely isn't loading/working correctly.")
    print("=" * 70)

    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

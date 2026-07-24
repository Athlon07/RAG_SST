"""Chunking strategy.

Uses recursive character splitting: it tries to split on paragraph breaks
first, then sentences, then words, only falling back to hard character
cuts as a last resort. This keeps chunks semantically coherent instead of
cutting mid-sentence. Overlap between consecutive chunks preserves context
that would otherwise be lost at chunk boundaries.

Chunking happens per-section (a PDF page, a Word heading block, an Excel
sheet, etc.) so every chunk keeps an accurate `location` label for citations,
rather than blending text across unrelated sections.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(sections, source_name, chunk_size=1000, chunk_overlap=200):
    """
    Args:
        sections: list of {"location": str, "text": str} from document_loader.load_document
        source_name: identifier for this document (used in chunk IDs)
        chunk_size: target characters per chunk
        chunk_overlap: characters shared between consecutive chunks

    Returns:
        list[dict]: [{"id", "text", "source", "location"}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    counter = 0
    for section in sections:
        for chunk_text in splitter.split_text(section["text"]):
            counter += 1
            chunks.append({
                "id": f"{source_name}_c{counter}",
                "text": chunk_text,
                "source": source_name,
                "location": section["location"],
            })
    return chunks

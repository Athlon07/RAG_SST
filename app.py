import os
import hashlib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.document_loader import load_documents_parallel
from utils.google_drive import download_from_drive_link
from utils.chunking import chunk_documents
from utils.vector_store import VectorStore
from utils.rag_pipeline import answer_question, condense_question
from utils import cache_manager

load_dotenv()

st.set_page_config(page_title="SmartAssistant", page_icon="static/smartshift_logo.png", layout="wide")

# Placeholder hook for custom frontend later: drop a static/style.css file in
# and it will be picked up automatically, no code changes needed.
css_path = os.path.join("static", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))
TOP_K = int(os.getenv("TOP_K", 4))
MAX_PARALLEL_LOADS = int(os.getenv("MAX_PARALLEL_LOADS", 4))

PERSIST_DIR = "chroma_db"
CACHE_DIR = "file_cache"
DOWNLOAD_DIR = "downloads"
UPLOAD_DIR = "uploads"
UPLOAD_COLLECTION_KEY = "uploaded_files"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "current_collection_key" not in st.session_state:
    st.session_state.current_collection_key = None
if "current_file_paths" not in st.session_state:
    st.session_state.current_file_paths = None

header_col1, header_col2 = st.columns([1, 12])
with header_col1:
    st.image("static/smartshift_logo.png", width=60)
with header_col2:
    st.title("SmartAssistant")
st.caption("Ask questions about your documents. Answers are grounded strictly in their content.")


def process_files(file_paths, collection_key, force_full_resync=False):
    """(Re)builds a collection from a batch of files.

    Speed optimizations applied here:
    - Files are loaded (parsed) in parallel via a thread pool.
    - Files whose content hash matches the cache are skipped entirely - no
      re-parsing, re-chunking, or re-embedding.
    - Every chunk that DOES need embedding, across every changed/new file in
      the batch, is embedded in a single batched call at the end, instead of
      one call per file.

    force_full_resync=True ignores the cache and the existing collection
    entirely, rebuilding everything from scratch.
    """
    vs = VectorStore(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL)
    vs.get_or_create_collection(name=collection_key)

    if force_full_resync:
        vs.clear_collection()
        cache = {}
    else:
        cache = cache_manager.load_cache(CACHE_DIR, collection_key)

    unchanged, changed_or_new, removed, file_hashes = cache_manager.diff_against_cache(file_paths, cache)

    # files that disappeared from this batch (deleted/renamed) - drop their chunks
    for name in removed:
        old_record = cache.pop(name, None)
        if old_record:
            vs.delete_by_ids(old_record.get("chunk_ids", []))

    # parse only what actually changed, in parallel
    paths_to_load = [fp for fp, _name in changed_or_new]
    loaded_sections, load_errors = load_documents_parallel(paths_to_load, max_workers=MAX_PARALLEL_LOADS)

    all_new_chunks = []
    added_names, updated_names = [], []
    error_names = [f"{os.path.basename(fp)} ({err})" for fp, err in load_errors.items()]

    for fp, name in changed_or_new:
        if fp not in loaded_sections:
            continue  # failed to load - already captured in error_names
        sections = loaded_sections[fp]
        if not sections:
            error_names.append(f"{name} (no extractable text)")
            continue

        was_present = name in cache
        if was_present:
            # content changed since last time - drop its old chunks before adding new ones
            vs.delete_by_ids(cache[name].get("chunk_ids", []))

        source_name = os.path.splitext(name)[0]
        chunks = chunk_documents(
            sections, source_name=source_name,
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )
        all_new_chunks.extend(chunks)

        cache[name] = {"hash": file_hashes[name], "chunk_ids": [c["id"] for c in chunks]}
        (updated_names if was_present else added_names).append(name)

    # single batched embedding call for everything new/changed in this run
    if all_new_chunks:
        vs.add_chunks(all_new_chunks)

    cache_manager.save_cache(CACHE_DIR, collection_key, cache)

    summary = {
        "added": added_names,
        "updated": updated_names,
        "removed": removed,
        "unchanged": unchanged,
        "errors": error_names,
        "synced_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary, vs


def summary_message(summary):
    msg = (
        f"Added: {len(summary['added'])} · Updated: {len(summary['updated'])} · "
        f"Removed: {len(summary['removed'])} · Unchanged (skipped): {len(summary['unchanged'])}"
    )
    if summary.get("synced_at"):
        msg = f"Last synced: {summary['synced_at']}\n\n{msg}"
    if summary["errors"]:
        msg += f"\n\n⚠️ Errors: {', '.join(summary['errors'])}"
    return msg


# Simple keyword match for questions ABOUT the document set itself (how many
# files, what files are loaded, list the documents, etc.) rather than about
# their CONTENT. Vector search has no reliable way to answer these - there's
# no chunk of text that says "here is the full file list," so retrieval ends
# up pulling back essentially arbitrary chunks depending on subtle wording
# differences. These questions are answered directly and deterministically
# from the vector store's own metadata instead of going through retrieval + LLM.
META_QUESTION_PATTERNS = [
    "how many files", "how many documents", "how many docs",
    "what files", "what documents", "which files", "which documents",
    "list the files", "list the documents", "list files", "list documents",
    "what are the files", "what are the documents",
    "name the files", "name the documents",
]


def is_meta_question(question: str) -> bool:
    q = question.lower().strip()
    return any(pattern in q for pattern in META_QUESTION_PATTERNS)


def answer_meta_question(vector_store):
    sources = vector_store.list_sources()
    if not sources:
        return "No files are currently loaded.", []
    lines = "\n".join(f"{i}. {name}" for i, name in enumerate(sources, start=1))
    answer = f"There are {len(sources)} file(s) currently loaded:\n\n{lines}"
    return answer, []


with st.sidebar:
    st.header("1. Load document(s)")
    input_method = st.radio("Choose input method", ["Upload file(s)", "Google Drive link"])

    file_paths = None
    collection_key = None

    if input_method == "Upload file(s)":
        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=["pdf", "docx", "xlsx", "xlsm", "html", "htm", "xml"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            file_paths = []
            for uf in uploaded_files:
                fp = os.path.join(UPLOAD_DIR, uf.name)
                with open(fp, "wb") as f:
                    f.write(uf.getbuffer())
                file_paths.append(fp)
        collection_key = UPLOAD_COLLECTION_KEY
    else:
        drive_url = st.text_input(
            "Paste a Google Drive shareable link",
            help="Works for a single file link OR a folder link — a folder link will "
                 "pull in every supported file inside it (PDF, Word, Excel, HTML, XML).",
        )
        if drive_url:
            collection_key = hashlib.md5(drive_url.encode()).hexdigest()[:10]

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process", type="primary", use_container_width=True)
    with col2:
        resync_btn = st.button("🔁 Full Re-sync", use_container_width=True)

    st.divider()
    st.header("2. Settings")
    st.write(f"Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    top_k = st.slider("Chunks to retrieve per question (top-k)", 2, 10, TOP_K)

    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not set. Add it to a .env file to enable answering. Get a free key at console.groq.com.")

    if st.session_state.last_summary:
        st.divider()
        st.caption(summary_message(st.session_state.last_summary))

if process_btn or resync_btn:
    if input_method == "Upload file(s)" and not file_paths:
        st.sidebar.error("Please upload at least one file first.")
    elif input_method == "Google Drive link" and not collection_key:
        st.sidebar.error("Please paste a Google Drive link.")
    else:
        try:
            with st.spinner("Processing document(s)... this can take a while for large batches."):
                if input_method == "Google Drive link":
                    file_paths = download_from_drive_link(drive_url, output_dir=DOWNLOAD_DIR)

                summary, vs = process_files(
                    file_paths, collection_key, force_full_resync=resync_btn
                )

            st.session_state.vector_store = vs
            st.session_state.doc_ready = True
            st.session_state.current_collection_key = collection_key
            st.session_state.current_file_paths = file_paths
            st.session_state.last_summary = summary
            total_active = len(summary["added"]) + len(summary["updated"]) + len(summary["unchanged"])
            st.session_state.doc_name = f"{total_active} file(s)"
            st.session_state.chat_history = []

            action = "Rebuilt" if resync_btn else "Processed"
            st.sidebar.success(f"{action}. {summary_message(summary)}")
        except Exception as e:
            st.sidebar.error(f"Failed to process document(s): {e}")

if st.session_state.doc_ready:
    st.success(f"Ready to answer questions about: **{st.session_state.doc_name}**")
else:
    st.info(
        "Upload one or more documents (PDF, Word, Excel, HTML, or XML), or paste a "
        "Google Drive file/folder link, then click 'Process' to begin. Unchanged files "
        "are automatically skipped on repeat runs — use 'Full Re-sync' to force a "
        "complete rebuild instead."
    )

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} — {s['location']}:** {s['excerpt']}...")

question = st.chat_input("Ask a question about the document...")

if question:
    if not st.session_state.doc_ready:
        st.error("Please process a document first.")
    elif not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file.")
    else:
        # capture history BEFORE adding the current question, so it's used
        # as "prior context" for both retrieval and the answer itself
        prior_history = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
        ]

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if is_meta_question(question):
                # answered directly from vector store metadata - no retrieval,
                # no LLM call, so it's fast, free, and always correct
                answer, sources = answer_meta_question(st.session_state.vector_store)
                st.markdown(answer)
            else:
                with st.spinner("Searching document and generating answer..."):
                    # resolve follow-ups like "elaborate on that" into a standalone
                    # query before doing vector search, which can't resolve pronouns on its own
                    search_query = condense_question(GROQ_API_KEY, GROQ_MODEL, question, prior_history)
                    retrieved = st.session_state.vector_store.query(search_query, top_k=top_k)
                    answer, sources = answer_question(
                        GROQ_API_KEY, GROQ_MODEL, question, retrieved, history=prior_history
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("View sources"):
                            for s in sources:
                                st.markdown(f"**{s['source']} — {s['location']}:** {s['excerpt']}...")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
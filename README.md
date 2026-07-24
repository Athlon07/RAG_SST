# Document RAG Assistant (Prototype)

A minimal Retrieval-Augmented Generation app: upload one or more documents
(PDF, Word, Excel, HTML, or XML), or point to a Google Drive file/folder
link, then ask questions and get answers grounded strictly in that content.

## Architecture

```
Files / Google Drive  --->  document_loader.py  --->  chunking.py  --->  vector_store.py (Chroma)
                                                                              |
                                                                              v
                                                               rag_pipeline.py (Groq API - free)
                                                                              |
                                                                              v
                                                                         app.py (Streamlit UI)
```

- **Loading**: `utils/document_loader.py` supports `.pdf` (via **PyMuPDF**,
  chosen over `pypdf` for its faster text extraction), `.docx`, `.xlsx`/
  `.xlsm`, `.html`/`.htm`, and `.xml`. Each loader returns a common structure -
  a list of `{"location": str, "text": str}` sections. `load_documents_parallel()`
  loads multiple files concurrently with a thread pool, so processing a batch
  of files doesn't happen one-at-a-time.
- **Google Drive**: `utils/google_drive.py` downloads via `gdown` - no API
  key or OAuth needed, just "Anyone with the link" sharing. The download
  folder is fully cleared before every fetch, so a file deleted from Drive
  won't linger locally and keep getting picked up.
- **Caching**: `utils/cache_manager.py` hashes each file's raw bytes (SHA-256)
  and records, per collection, which chunk IDs each file produced. On the
  next "Process" click:
    - unchanged files (same hash) are **skipped entirely** - no re-parsing,
      re-chunking, or re-embedding
    - changed files have their old chunks deleted and are re-processed
    - files no longer in the batch (deleted/renamed) have their chunks removed
  This is what makes repeat runs fast, and also means deleted files actually
  disappear from answers instead of lingering.
- **Chunking**: `utils/chunking.py` uses `RecursiveCharacterTextSplitter`
  (paragraph → sentence → word → character fallback). Default chunk size was
  increased from 1000/200 to **1500/300 characters** - larger chunks mean
  fewer total chunks per document, which means fewer (and more efficient)
  embedding calls. Tune `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env` if your
  documents need finer-grained retrieval instead.
- **Vector DB**: `utils/vector_store.py` uses **Chroma** (persistent, on disk
  at `./chroma_db`) instead of FAISS, with a `delete_by_ids()` method the
  cache system uses to surgically remove just one file's chunks. Embeddings
  are computed locally with `sentence-transformers` (`all-MiniLM-L6-v2` by
  default) - no embedding API key needed. Every chunk that needs (re)embedding
  across an entire batch of files is sent to `add_chunks()` in a **single
  batched call**, rather than one call per file - Chroma/sentence-transformers
  embed a batch far more efficiently than many small calls.
- **Answering**: `utils/rag_pipeline.py` retrieves the top-k most relevant
  chunks and sends them to Groq's free LLM API (Llama 3.3 70B by default)
  with a system prompt that forbids outside knowledge and requires a fixed
  fallback sentence when the answer isn't in the retrieved context. Sources
  (document name + location + excerpt) are shown under every answer.
- **Conversation memory**: follow-up questions like "elaborate on that" are
  first rewritten into a standalone question (`condense_question`) using the
  last few turns, since vector search can't resolve pronouns on its own.
- **UI**: `app.py`, a Streamlit chat interface with two processing buttons:
  - **Process** - incremental, cache-aware (the fast path for everyday use)
  - **🔁 Full Re-sync** - ignores the cache, clears the collection, and
    rebuilds everything from scratch (use this if you ever suspect the
    cache/index has drifted from reality)

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY
```

Get a **free** Groq API key (no credit card required):
1. Go to https://console.groq.com and sign in
2. **API Keys** → **Create API Key**
3. Paste it into `.env` as `GROQ_API_KEY`

Free tier: roughly 30 requests/min and ~1,000 requests/day on
`llama-3.3-70b-versatile` (check https://console.groq.com/docs/rate-limits
for current limits).

No other API keys are needed - Google Drive access works via `gdown` with
just a shareable link, no Google account/API key required.

## Run

```bash
streamlit run app.py
```

Then in the sidebar:
1. Choose "Upload file(s)" (now accepts multiple files at once) or "Google
   Drive link" (a single file's link, or a folder's link to pull in every
   supported file inside it).
2. Click **Process** to build/update the index - unchanged files from a
   previous run are skipped automatically. Click **🔁 Full Re-sync** instead
   if you want to force a complete rebuild (e.g. after changing chunk size
   settings, or if something seems off).
3. Ask questions in the chat box.

## Notes & limitations

- **Scanned/image-only PDFs**: PyMuPDF extracts embedded text only. Scanned
  pages with no text layer won't have content to index - you'd need OCR
  (e.g. `pytesseract`) added to `document_loader.py` for that case.
- **Excel formulas**: `openpyxl` is loaded with `data_only=True`, which reads
  the last *calculated* value stored in the file. If a workbook has never
  been opened in Excel/LibreOffice since a formula changed, that cached value
  may be stale or missing.
- **XML**: the loader is generic (one section per element containing text,
  labeled by tag path) so it works reasonably for document-like XML. Deeply
  nested data-oriented XML may produce a lot of very small sections.
- **Google Drive access**: only works for files/folders shared as "Anyone
  with the link". Restricted/private files would need the Drive API with
  OAuth credentials, which isn't included in this prototype.
- **Google Drive folder links**: `gdown.download_folder` scrapes Drive's
  folder listing page rather than using the official API - can be slow for
  large folders and isn't guaranteed to handle every edge case (Shared
  Drives, very large folders) as reliably as the official API would.
- **Caching is per-collection, keyed by filename + content hash**: if you
  rename a file without changing its content, it'll be treated as a new file
  (old chunks under the old name stick around until you run Full Re-sync or
  the old name naturally drops out of a future batch).
- **Parallel loading** uses a thread pool (`MAX_PARALLEL_LOADS`, default 4)
  rather than a process pool - simpler (no pickling issues) and effective
  since PyMuPDF/python-docx/openpyxl/BeautifulSoup release the GIL during
  their C-extension work. For very large batches of very large files, a
  process pool could parallelize CPU-bound parsing further, at the cost of
  more complexity.
- **No hallucination guardrail**: the model is instructed to answer only from
  retrieved context and to output a fixed "not found" sentence otherwise. This
  is a strong prompt-level guardrail but not a hard guarantee.

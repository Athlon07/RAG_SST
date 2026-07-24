"""Content-hash based caching so unchanged files skip re-extraction,
re-chunking, and re-embedding entirely - the expensive parts of processing.

Much simpler than tracking Drive-specific metadata: we just hash each file's
raw bytes. If a file's hash matches what's recorded from last time, its
chunks are already sitting in the vector store and there's nothing to do.
If the hash differs (or the file is new), we reprocess just that file. If a
previously-seen file is no longer present in the current batch, its old
chunks get removed so deleted/renamed files don't linger.

Cache is a small JSON manifest on disk, keyed by whatever "collection_key"
the caller uses for that vector store collection.
"""
import os
import json
import hashlib


def _manifest_path(cache_dir: str, collection_key: str) -> str:
    return os.path.join(cache_dir, f"cache_{collection_key}.json")


def load_cache(cache_dir: str, collection_key: str) -> dict:
    """Returns {filename: {"hash": str, "chunk_ids": [str, ...]}}"""
    path = _manifest_path(cache_dir, collection_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_dir: str, collection_key: str, cache: dict):
    os.makedirs(cache_dir, exist_ok=True)
    with open(_manifest_path(cache_dir, collection_key), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def compute_file_hash(file_path: str) -> str:
    """SHA-256 of the file's raw bytes, computed in chunks to handle large files."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def diff_against_cache(file_paths: list, cache: dict):
    """Compare the current batch of files against the cache.

    Returns:
        unchanged: list[str] filenames whose hash matches the cache (skip entirely)
        changed_or_new: list[(file_path, filename)] that need (re)processing
        removed: list[str] filenames that were cached but aren't in this batch anymore
        file_hashes: dict[filename -> hash] for the current batch (to save back to cache)
    """
    current_names = {}
    file_hashes = {}
    for fp in file_paths:
        name = os.path.basename(fp)
        current_names[name] = fp
        file_hashes[name] = compute_file_hash(fp)

    unchanged = []
    changed_or_new = []
    for name, fp in current_names.items():
        prior = cache.get(name)
        if prior and prior.get("hash") == file_hashes[name]:
            unchanged.append(name)
        else:
            changed_or_new.append((fp, name))

    removed = [name for name in cache.keys() if name not in current_names]

    return unchanged, changed_or_new, removed, file_hashes

"""Download file(s) from a Google Drive link using gdown - no API key or
OAuth needed, just "Anyone with the link" sharing.

Important: the local download folder is fully cleared before every download.
Earlier versions of this app reused the same folder across runs without
clearing it, so a file deleted from Drive would linger locally forever and
keep getting picked up. Clearing first means what's on disk after a
download always matches Drive's *current* contents - deletions are then
naturally reflected. Content-hash caching (see cache_manager.py) is what
keeps this fast despite the re-download, since re-parsing/re-chunking/
re-embedding unchanged files is skipped even though their bytes get
re-fetched.
"""
import os
import re
import glob
import shutil

import gdown

SUPPORTED_EXTS = (".pdf", ".docx", ".xlsx", ".xlsm", ".html", ".htm", ".xml")


def is_drive_folder(url_or_id: str) -> bool:
    return "/folders/" in url_or_id


def extract_file_id(url_or_id: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()


def extract_folder_id(url_or_id: str) -> str:
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()


def _download_single_file(url_or_id: str, output_dir: str) -> str:
    file_id = extract_file_id(url_or_id)
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Ending `output` in a path separator tells gdown to infer the filename
    # (and its extension) from the server's Content-Disposition header, so
    # this works for PDFs, Word docs, Excel files, HTML, XML - not just PDFs.
    output_path = gdown.download(
        url=download_url, output=os.path.join(output_dir, ""), fuzzy=True, quiet=False
    )

    if not output_path or not os.path.exists(output_path):
        raise FileNotFoundError(
            "Could not download the file from Google Drive. Make sure sharing is set "
            "to 'Anyone with the link'."
        )
    return output_path


def _download_folder(url_or_id: str, output_dir: str) -> list:
    folder_id = extract_folder_id(url_or_id)
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)

    matched_paths = []
    for ext in SUPPORTED_EXTS:
        matched_paths.extend(glob.glob(os.path.join(output_dir, "**", f"*{ext}"), recursive=True))

    if not matched_paths:
        supported = ", ".join(SUPPORTED_EXTS)
        raise FileNotFoundError(
            "Could not find any supported files in that Google Drive folder. Make sure "
            f"the folder is shared as 'Anyone with the link' and contains at least one "
            f"file of type: {supported}."
        )
    return sorted(matched_paths)


def download_from_drive_link(url_or_id: str, output_dir: str = "downloads") -> list:
    """Download from a Google Drive link, whether it points to a single file
    or an entire folder. Clears output_dir first so deleted-from-Drive files
    don't linger locally.

    Returns:
        list[str]: paths to every downloaded file (length 1 for a single-file link).
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    if is_drive_folder(url_or_id):
        return _download_folder(url_or_id, output_dir)
    else:
        return [_download_single_file(url_or_id, output_dir)]

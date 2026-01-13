#!/usr/bin/env python3
"""
Google Drive downloader for GitHub Actions pipelines.

Supports:
- Google Drive FILE links (including Google Sheets) via:
  1) Service Account (recommended) using Drive API (exports Google Sheets -> XLSX)
  2) Public "anyone-with-link" fallback:
     - For Google Sheets: uses the public export URL -> downloads XLSX reliably
     - For regular files: uses gdown

- Google Drive FOLDER links:
  1) Service Account mode: downloads all eligible files in folder
  2) Public mode: uses gdown (NOTE: cannot reliably export Google Sheets in folders without API)

Env secrets (optional but recommended):
- GDRIVE_SERVICE_ACCOUNT_JSON (raw JSON string)
- GDRIVE_SERVICE_ACCOUNT_B64  (base64-encoded JSON)
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import requests


@dataclass(frozen=True)
class DriveLink:
    kind: str  # "file" or "folder"
    id: str
    is_google_sheet: bool = False


_FILE_ID_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)"),
    re.compile(r"drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
]

_FOLDER_ID_PATTERNS = [
    re.compile(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)"),
]

_SHEETS_ID_PATTERNS = [
    re.compile(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)"),
]


def parse_drive_link(link: str) -> DriveLink:
    link = link.strip()

    for pat in _FOLDER_ID_PATTERNS:
        m = pat.search(link)
        if m:
            return DriveLink(kind="folder", id=m.group(1), is_google_sheet=False)

    for pat in _SHEETS_ID_PATTERNS:
        m = pat.search(link)
        if m:
            return DriveLink(kind="file", id=m.group(1), is_google_sheet=True)

    for pat in _FILE_ID_PATTERNS:
        m = pat.search(link)
        if m:
            return DriveLink(kind="file", id=m.group(1), is_google_sheet=False)

    raise ValueError(f"Unrecognized Google Drive link format: {link}")


def _load_service_account_json() -> Optional[dict]:
    raw = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    b64 = os.environ.get("GDRIVE_SERVICE_ACCOUNT_B64", "").strip()

    if raw:
        return json.loads(raw)

    if b64:
        decoded = base64.b64decode(b64).decode("utf-8")
        return json.loads(decoded)

    return None


def _has_service_account() -> bool:
    try:
        return _load_service_account_json() is not None
    except Exception:
        return False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download_public_sheet_as_xlsx(sheet_id: str, dest_dir: Path) -> Path:
    """Download XLSX via public export endpoint (works if sheet is shared)."""
    _ensure_dir(dest_dir)
    out_path = dest_dir / f"{sheet_id}.xlsx"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(
                f"Could not export Google Sheet via public link (HTTP {r.status_code}). "
                "Share the sheet as 'Anyone with the link' OR provide a service account secret."
            )
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if out_path.stat().st_size < 5000:
        raise RuntimeError("Downloaded XLSX looks too small; export likely failed.")
    return out_path


def _ensure_gdown() -> None:
    if shutil.which("gdown"):
        return
    raise RuntimeError("gdown is not installed. Ensure requirements.txt includes 'gdown'.")


def download_public_with_gdown(link: str, dest_dir: Path, allowed_exts: Sequence[str]) -> List[Path]:
    """Public fallback for regular Drive file/folder links."""
    _ensure_dir(dest_dir)
    _ensure_gdown()

    import subprocess

    cmd = ["gdown", "--no-cookies", "--quiet"]

    parsed = None
    try:
        parsed = parse_drive_link(link)
    except Exception:
        pass

    if parsed and parsed.kind == "folder":
        cmd += ["--folder", link, "-O", str(dest_dir)]
    else:
        cmd += [link, "-O", str(dest_dir)]

    subprocess.check_call(cmd)

    files: List[Path] = []
    allowed = {e.lower() for e in allowed_exts}
    for p in dest_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p)

    if not files:
        raise RuntimeError(
            f"Downloaded from Drive but found no files with extensions {allowed_exts}. "
            f"Downloaded contents are in: {dest_dir}"
        )
    return sorted(files)


def _drive_api_client():
    sa = _load_service_account_json()
    if not sa:
        raise RuntimeError("Service account JSON not found in env.")

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(sa, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _download_drive_file_via_api(file_id: str, dest_path: Path) -> Path:
    from googleapiclient.http import MediaIoBaseDownload
    import io

    service = _drive_api_client()
    meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    name = meta.get("name", file_id)
    mime = meta.get("mimeType", "")

    # If caller passed a bare file_id, use the Drive filename.
    if dest_path.name == file_id and name:
        dest_path = dest_path.with_name(name)

    if mime == "application/vnd.google-apps.spreadsheet":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        dest_path = dest_path.with_suffix(".xlsx")
    else:
        request = service.files().get_media(fileId=file_id)

    _ensure_dir(dest_path.parent)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()
    return dest_path


def _list_folder_files_via_api(folder_id: str) -> List[Tuple[str, str, str]]:
    service = _drive_api_client()
    q = f"'{folder_id}' in parents and trashed=false"
    page_token = None
    out: List[Tuple[str, str, str]] = []
    while True:
        resp = (
            service.files()
            .list(q=q, fields="nextPageToken, files(id,name,mimeType)", pageToken=page_token, pageSize=200)
            .execute()
        )
        for f in resp.get("files", []):
            out.append((f["id"], f.get("name", f["id"]), f.get("mimeType", "")))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def download_with_service_account(link: str, dest_dir: Path, allowed_exts: Sequence[str]) -> List[Path]:
    _ensure_dir(dest_dir)
    parsed = parse_drive_link(link)

    downloaded: List[Path] = []
    allowed = {e.lower() for e in allowed_exts}

    if parsed.kind == "file":
        out = _download_drive_file_via_api(parsed.id, dest_dir / parsed.id)
        downloaded.append(out)
    else:
        items = _list_folder_files_via_api(parsed.id)
        for fid, name, mime in items:
            if mime == "application/vnd.google-apps.spreadsheet":
                out = _download_drive_file_via_api(fid, dest_dir / f"{name}.xlsx")
                downloaded.append(out)
                continue

            suffix = Path(name).suffix.lower()
            if suffix in allowed:
                out = _download_drive_file_via_api(fid, dest_dir / name)
                downloaded.append(out)

    downloaded = [p for p in downloaded if p.exists()]
    if not downloaded:
        raise RuntimeError(
            f"Service account download succeeded but no eligible files were found for extensions {allowed_exts}."
        )
    return sorted(downloaded)


def download_drive_inputs(link: str, dest_dir: str | Path, allowed_exts: Sequence[str] = (".xlsx", ".csv")) -> List[Path]:
    dest_dir = Path(dest_dir)
    _ensure_dir(dest_dir)

    parsed = parse_drive_link(link)

    if _has_service_account():
        return download_with_service_account(link, dest_dir=dest_dir, allowed_exts=allowed_exts)

    # Public fallback: Google Sheet single-file export (reliable)
    if parsed.kind == "file" and parsed.is_google_sheet:
        return [_download_public_sheet_as_xlsx(parsed.id, dest_dir)]

    # Public fallback: gdown
    # NOTE: public folder export of Google Sheets is not reliable without API.
    return download_public_with_gdown(link, dest_dir=dest_dir, allowed_exts=allowed_exts)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-link", required=True)
    ap.add_argument("--dest-dir", required=True)
    args = ap.parse_args()

    files = download_drive_inputs(args.drive_link, dest_dir=args.dest_dir)
    for f in files:
        print(f)

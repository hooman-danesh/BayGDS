#!/usr/bin/env python3
"""Download the archived datasets and cached results from Zenodo.

This script fetches the data archive that accompanies the manuscript and
extracts it into the directory layout the analysis scripts expect:

    data/structures/structures.npz          microstructure images
    data/pca/pc_scores.npz                  PC score descriptors
    data/oracle/oracle_0deg.npz             oracle responses, 0 deg loading
    data/oracle/oracle_45deg.npz            oracle responses, 45 deg loading
    output/surrogate/active_learning/       selected surrogate checkpoint
    output/inverse_design/                  cached inverse design batches
    output/studies/                         cached sensitivity and baseline runs

Typical use, from the repository root::

    python download_data.py                 # fetch and extract everything
    python download_data.py --check         # verify what is already present
    python download_data.py --list          # show the archive and its contents

After a successful run, ``python reproduction/reproduce.py`` regenerates every
code-produced manuscript figure and table without any further computation.

The archive name, size, and MD5 checksum are read from ``zenodo_manifest.json``.
Every downloaded file is checksum-verified before it is extracted, and a
partially downloaded file is never left in place.

Only the Python standard library is used, so this script runs before the
project dependencies in ``requirements.txt`` are installed.

Author:  Hooman Danesh <hooman.danesh@tu-braunschweig.de>, TU Braunschweig
License: MIT (see LICENSE). Released September 2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "zenodo_manifest.json"
CACHE_DIR = ROOT / ".zenodo_cache"

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
ZENODO_FILE = "https://zenodo.org/records/{record_id}/files/{name}?download=1"

#: Stand-in used before the deposition is published.  A real record id is a
#: plain integer such as ``12345678``.
PLACEHOLDER_RECORD = "REPLACE_WITH_ZENODO_RECORD_ID"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    """Read ``zenodo_manifest.json`` and fail loudly if it is missing."""
    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "The manifest ships with the repository; re-clone to restore it."
        )
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_record_id(manifest: dict, override: str | None) -> str:
    """Resolve the Zenodo record id from the CLI, the environment, or the manifest.

    Precedence is ``--record`` > ``BAYGDS_ZENODO_RECORD`` > the manifest value.
    This lets a reviewer point the script at a sandbox or draft deposition
    without editing any file.
    """
    record = override or os.environ.get("BAYGDS_ZENODO_RECORD") or manifest.get("record_id", "")
    record = str(record).strip()
    if not record or record == PLACEHOLDER_RECORD:
        raise SystemExit(
            "No Zenodo record id is configured.\n"
            "Set it in one of the following ways:\n"
            f"  1. edit \"record_id\" in {MANIFEST_PATH.name}\n"
            "  2. export BAYGDS_ZENODO_RECORD=<record id>\n"
            "  3. pass --record <record id>\n"
            "The record id is the number in the Zenodo URL, for example\n"
            "https://zenodo.org/records/12345678 -> 12345678"
        )
    return record


def select_archives(manifest: dict) -> list[dict]:
    """Return the archive entries declared by the manifest."""
    archives = manifest.get("archives", [])
    if not archives:
        raise SystemExit(f"No archives are declared in {MANIFEST_PATH.name}.")
    return list(archives)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def human_bytes(num_bytes: float) -> str:
    """Format a byte count for terminal output."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024.0 or unit == "GiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GiB"


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute the MD5 digest of a file, reading it in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Presence checks
# ---------------------------------------------------------------------------
def missing_targets(entry: dict) -> list[str]:
    """Return the archive's sentinel paths that are absent from the repository.

    The sentinels are the files the analysis scripts actually open, so an empty
    list means this archive has already been extracted successfully.
    """
    return [target for target in entry.get("targets", []) if not (ROOT / target).exists()]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def zenodo_file_urls(record_id: str, name: str) -> list[str]:
    """Return candidate download URLs for one archive, best first.

    The record API is queried so that the official per-file link and checksum
    are used when Zenodo is reachable; the direct content URL is kept as a
    fallback for mirrors and for offline proxies.
    """
    urls: list[str] = []
    api_url = ZENODO_API.format(record_id=record_id)
    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            record = json.load(response)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        print(f"  note: could not query the Zenodo API ({error}); using the direct file URL")
    else:
        for file_entry in record.get("files", []):
            if file_entry.get("key") == name:
                link = file_entry.get("links", {}).get("self")
                if link:
                    urls.append(link)
                break
        else:
            available = ", ".join(str(f.get("key")) for f in record.get("files", []))
            print(f"  note: '{name}' is not listed on record {record_id}. Files there: {available}")
    urls.append(ZENODO_FILE.format(record_id=record_id, name=name))
    return urls


#: HTTP statuses worth retrying. Zenodo returns these while a gateway is busy.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

#: Attempts per URL before giving up on it.
MAX_ATTEMPTS = 5


def _is_transient(error: Exception) -> bool:
    """Whether an error is worth retrying rather than reporting."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_STATUS
    # URLError wraps connection resets and DNS failures; TimeoutError covers a
    # stalled read. Both are routine on a large transfer.
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))


def download(url: str, destination: Path, expected_size: int | None) -> None:
    """Stream one URL to ``destination``, resuming and retrying as needed.

    A partial transfer is kept in a ``.part`` file beside the destination and
    continued with a Range request, so a dropped connection costs only the
    bytes still missing rather than the whole archive.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resuming = response.status == 206
                if have and not resuming:
                    # The server ignored the Range header; start over.
                    have = 0
                total = expected_size or (
                    int(response.headers.get("Content-Length") or 0) + have
                )
                downloaded = have
                with open(partial, "ab" if resuming else "wb") as handle:
                    if resuming:
                        print(f"  resuming at {human_bytes(have)}")
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = 100.0 * downloaded / total
                            bar = "#" * int(percent // 2.5)
                            sys.stdout.write(
                                f"\r  [{bar:<40}] {percent:5.1f}%  "
                                f"{human_bytes(downloaded)} / {human_bytes(total)}"
                            )
                        else:
                            sys.stdout.write(f"\r  {human_bytes(downloaded)} downloaded")
                        sys.stdout.flush()
            sys.stdout.write("\n")
            partial.replace(destination)
            return
        except Exception as error:  # noqa: BLE001 - re-raised below when fatal
            sys.stdout.write("\n")
            if not _is_transient(error) or attempt == MAX_ATTEMPTS:
                raise
            delay = min(2 ** attempt, 30)
            reason = getattr(error, "code", None) or type(error).__name__
            print(f"  {reason} from the server; retrying in {delay}s "
                  f"(attempt {attempt + 1} of {MAX_ATTEMPTS})")
            time.sleep(delay)


def fetch_archive(entry: dict, record_id: str, local_dir: Path | None) -> Path:
    """Obtain one archive, either from ``--from-dir`` or from Zenodo.

    A cached copy whose checksum already matches is reused, which makes a
    re-run after an interruption cheap.
    """
    name = entry["name"]
    expected_md5 = entry.get("md5")

    if local_dir is not None:
        candidate = local_dir / name
        if not candidate.is_file():
            raise SystemExit(f"Archive not found in --from-dir: {candidate}")
        return candidate

    cached = CACHE_DIR / name
    if cached.is_file():
        print(f"  cached archive found, verifying: {cached.relative_to(ROOT)}")
        if not expected_md5 or md5sum(cached) == expected_md5:
            return cached
        print("  cached archive is corrupt, downloading again")
        cached.unlink()

    last_error: Exception | None = None
    for url in zenodo_file_urls(record_id, name):
        print(f"  downloading from {url}")
        try:
            download(url, cached, entry.get("size"))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            print(f"  failed: {error}")
            continue
        return cached
    raise SystemExit(f"Could not download '{name}'. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract(archive: Path, destination: Path) -> None:
    """Extract a tar.gz archive into ``destination`` with path-traversal guards."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination.resolve())):
                raise SystemExit(f"Refusing to extract outside the repository: {member.name}")
        # ``filter='data'`` strips ownership and special files (Python >= 3.12).
        try:
            tar.extractall(destination, filter="data")
        except TypeError:  # pragma: no cover - Python < 3.12
            tar.extractall(destination)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def command_list(manifest: dict, archives: list[dict]) -> int:
    """Print the archives, their sizes, and where they extract to."""
    doi = manifest.get("doi") or "(DOI not yet assigned)"
    print(f"Zenodo deposition: {doi}")
    print(f"Record id in manifest: {manifest.get('record_id')}\n")
    total = 0
    for entry in archives:
        size = int(entry.get("size", 0))
        total += size
        state = "present" if not missing_targets(entry) else "missing"
        print(f"{entry['name']}  ({human_bytes(size)})  [{state}]")
        print(f"  {entry.get('description', '')}")
        for target in entry.get("targets", []):
            print(f"    -> {target}")
        print()
    print(f"Total download size: {human_bytes(total)}")
    return 0


def command_check(archives: list[dict]) -> int:
    """Report which expected files are present and which are missing."""
    incomplete = False
    for entry in archives:
        missing = missing_targets(entry)
        if missing:
            incomplete = True
            print(f"[missing] {entry['name']}")
            for target in missing:
                print(f"    absent: {target}")
        else:
            print(f"[ok]      {entry['name']}")
    if incomplete:
        print("\nSome files are missing. Run: python download_data.py")
        return 1
    print("\nAll expected data files are present.")
    return 0


def command_download(
    manifest: dict,
    archives: list[dict],
    record_override: str | None,
    local_dir: Path | None,
    force: bool,
    keep_archives: bool,
) -> int:
    """Fetch, verify, and extract the selected archives."""
    pending = archives if force else [entry for entry in archives if missing_targets(entry)]
    if not pending:
        print("Everything is already in place. Use --force to download again.")
        return 0

    record_id = "" if local_dir is not None else resolve_record_id(manifest, record_override)
    total = sum(int(entry.get("size", 0)) for entry in pending)
    print(f"Archives to fetch: {len(pending)} ({human_bytes(total)})\n")

    for index, entry in enumerate(pending, start=1):
        name = entry["name"]
        print(f"[{index}/{len(pending)}] {name} - {entry.get('description', '')}")
        archive = fetch_archive(entry, record_id, local_dir)

        expected_md5 = entry.get("md5")
        if expected_md5:
            print("  verifying checksum ...")
            actual = md5sum(archive)
            if actual != expected_md5:
                raise SystemExit(
                    f"  checksum mismatch for {name}\n"
                    f"    expected {expected_md5}\n"
                    f"    got      {actual}\n"
                    "  Delete .zenodo_cache and try again; if it persists the "
                    "deposition and this repository are out of sync."
                )
            print("  checksum ok")
        else:
            print("  no checksum in the manifest, skipping verification")

        print(f"  extracting into {ROOT}")
        extract(archive, ROOT)

        still_missing = missing_targets(entry)
        if still_missing:
            raise SystemExit(
                "  extraction did not produce the expected files: " + ", ".join(still_missing)
            )
        if not keep_archives and local_dir is None:
            archive.unlink(missing_ok=True)
        print("  done\n")

    if CACHE_DIR.is_dir() and not keep_archives and not any(CACHE_DIR.iterdir()):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    print("All requested data is in place.")
    print("Next step:  python reproduction/reproduce.py")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and extract the Zenodo data for this repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python download_data.py                    fetch everything that is missing\n"
            "  python download_data.py --check            verify what is already present\n"
            "  python download_data.py --force            re-download and overwrite\n"
            "  python download_data.py --from-dir ~/Downloads  use an archive you already have\n"
        ),
    )
    parser.add_argument("--record", help="Zenodo record id, overriding the manifest.")
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="Extract the archive from a local directory instead of downloading it, "
        "for example after fetching it by hand from the Zenodo page.",
    )
    parser.add_argument("--force", action="store_true", help="Download again even if files exist.")
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded .tar.gz files in .zenodo_cache after extraction.",
    )
    parser.add_argument("--list", action="store_true", help="List the archives and exit.")
    parser.add_argument(
        "--check", action="store_true", help="Report which expected files are present and exit."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest()
    archives = select_archives(manifest)

    if args.list:
        return command_list(manifest, archives)
    if args.check:
        return command_check(archives)
    return command_download(
        manifest,
        archives,
        record_override=args.record,
        local_dir=args.from_dir,
        force=args.force,
        keep_archives=args.keep_archives,
    )


if __name__ == "__main__":
    raise SystemExit(main())

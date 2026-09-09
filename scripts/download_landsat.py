#!/usr/bin/env python3
"""Download the Landsat assets used by the HSI/UHI notebook.

The Planetary Computer STAC item contains unsigned Azure URLs.  This script
requests short-lived public SAS URLs, downloads only the assets needed by the
experiment, and resumes partial files when the server supports byte ranges.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANDSAT_DIR = ROOT / "data" / "landsat"
ITEM_IDS = (
    "LC08_L2SP_125052_20250409_02_T1",
    "LC08_L2SP_125053_20250409_02_T1",
)
ASSETS = (
    "blue",
    "green",
    "red",
    "nir08",
    "swir16",
    "swir22",
    "lwir11",
    "qa_pixel",
    "qa_radsat",
    "emis",
    "emsd",
    "mtl.json",
)
SIGN_ENDPOINT = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def sign(href: str) -> str:
    url = SIGN_ENDPOINT + "?" + urllib.parse.urlencode({"href": href})
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["href"]


def download(asset: tuple[str, str, str]) -> Path:
    item_id, key, href = asset
    target_dir = LANDSAT_DIR / item_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(urllib.parse.urlparse(href).path).name

    for attempt in range(5):
        existing = target.stat().st_size if target.exists() else 0
        signed = sign(href)
        request = urllib.request.Request(signed)
        if existing:
            request.add_header("Range", f"bytes={existing}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if existing and status == 206 else "wb"
                with target.open(mode) as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
            print(f"DONE {item_id} {key}: {target.name} ({target.stat().st_size:,} bytes)", flush=True)
            return target
        except Exception as exc:
            print(f"RETRY {item_id} {key} ({attempt + 1}/5): {exc}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {item_id}:{key}")


def main() -> None:
    LANDSAT_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, str]] = []
    for item_id in ITEM_IDS:
        item_path = LANDSAT_DIR / f"{item_id}.json"
        with item_path.open() as stream:
            item = json.load(stream)
        for key in ASSETS:
            jobs.append((item_id, key, item["assets"][key]["href"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs))


if __name__ == "__main__":
    main()

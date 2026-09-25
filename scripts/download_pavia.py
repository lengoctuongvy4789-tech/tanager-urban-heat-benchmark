#!/usr/bin/env python3
"""Download the Pavia University HSI cube and pixel labels."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark" / "pavia_university"

FILES = {
    "PaviaU.mat": [
        "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/PaviaU.mat",
    ],
    "PaviaU_gt.mat": [
        "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data/PaviaU_gt.mat",
    ],
}


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 HSI-Lab/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(destination)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, urls in FILES.items():
        destination = OUT / filename
        if destination.exists() and destination.stat().st_size > 1_000:
            print(f"Có sẵn: {destination.relative_to(ROOT)}")
            continue
        for url in urls:
            try:
                print(f"Đang tải {filename} từ {url}")
                download(url, destination)
                print(f"Đã lưu: {destination.relative_to(ROOT)} ({destination.stat().st_size / 2**20:.1f} MB)")
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                print(f"Không tải được từ nguồn này: {exc}")
        else:
            raise RuntimeError(f"Không tải được {filename} từ các nguồn công khai đã cấu hình")


if __name__ == "__main__":
    main()

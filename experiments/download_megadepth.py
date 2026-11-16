"""Download only official MegaDepth-1500 images/cameras using HTTP byte ranges."""

import argparse
import io
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"


def fetch_range(start, end):
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                URL, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                expected = f"bytes {start}-{end}/"
                if response.status != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(expected):
                    raise ValueError("Server did not honor the requested byte range")
                data = response.read()
            if len(data) != end - start + 1:
                raise ValueError("Truncated HTTP range")
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def download(destination, workers=16, verbose=False):
    destination = Path(destination).resolve()
    with urllib.request.urlopen(
        urllib.request.Request(URL, method="HEAD"), timeout=30
    ) as response:
        size = int(response.headers["Content-Length"])
    tail = fetch_range(size - 65536, size - 1)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError("Missing ZIP central directory")
    fields = struct.unpack_from("<4s4H2LH", tail, end)
    central_offset = fields[6]
    # zipfile adjusts offsets relative to this shortened central-directory buffer.
    directory = zipfile.ZipFile(io.BytesIO(fetch_range(central_offset, size - 1)))
    members = [
        m
        for m in directory.infolist()
        if not m.is_dir()
        and (
            "/images/" in m.filename or m.filename.endswith(("views.txt", "pairs.txt"))
        )
    ]

    def extract(member):
        path = (destination / member.filename).resolve()
        if not path.is_relative_to(destination):
            raise ValueError("Unsafe archive path")
        if path.exists():
            data = path.read_bytes()
            if len(data) == member.file_size and zlib.crc32(data) == member.CRC:
                return
        offset = member.header_offset + central_offset
        header = fetch_range(offset, offset + 29)
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise ValueError("Invalid or encrypted ZIP member")
        start = offset + 30 + fields[-2] + fields[-1]
        compressed = fetch_range(start, start + member.compress_size - 1)
        if member.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(compressed, -15)
        elif member.compress_type == zipfile.ZIP_STORED:
            data = compressed
        else:
            raise ValueError("Unsupported ZIP compression")
        if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
            raise ValueError(f"ZIP checksum mismatch: {member.filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)

    megabytes = sum(m.compress_size for m in members) / 1e6
    print(
        f"Downloading/checking {len(members)} files ({megabytes:.0f} MB); "
        "depth maps omitted",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract, member) for member in members]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if verbose or index % 50 == 0 or index == len(futures):
                print(f"Verified {index}/{len(futures)} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("Use 1–32 download workers")
    download(args.destination, args.workers, args.verbose)

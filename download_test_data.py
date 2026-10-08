#!/usr/bin/env python3
"""Fetch the image files referenced by this repository's GT annotations.

The checked-in annotations define the retained evaluation subset. ImageNet and
HICO archives are temporary extraction inputs and are removed after a
successful download unless ``--keep-archives`` is supplied.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
OPEN_IMAGES = {
    "openimages_common_214": (
        "openimages_common_214_ram_annots.txt",
        "openimages_common_214_tag2text_idannots.txt",
    ),
    "openimages_rare_200": ("openimages_rare_200_ram_annots.txt",),
}
IMAGENET_URL = "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
IMAGENET_BYTES = 6744924160
IMAGENET_MD5 = "29b22e2961454d5413ddabcf34fc5622"
HICO_DRIVE_ID = "1SSWJZaczRNYeUSEkIzJb019YaVUagt3J"


def checked_ids(dataset, annotation):
    ids = []
    for line in (DATA / dataset / annotation).read_text().splitlines():
        ident = line.split(",", 1)[0]
        if not re.fullmatch(r"test/[0-9a-f]{16}", ident):
            raise ValueError(f"invalid OpenImages ID: {ident}")
        ids.append(ident.split("/", 1)[1])
    return ids


def fetch_image(ident, destinations):
    existing = next((p for p in destinations if p.is_file() and p.stat().st_size > 100), None)
    if existing is None:
        primary = destinations[0]
        primary.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://open-images-dataset.s3.amazonaws.com/test/{ident}.jpg"
        part = primary.with_suffix(".jpg.part")
        for attempt in range(4):
            try:
                if shutil.disk_usage(primary.parent).free < 8 * 1024**3:
                    raise RuntimeError("less than 8 GiB free; stopping before disk exhaustion")
                with urllib.request.urlopen(url, timeout=45) as response, part.open("wb") as target:
                    if response.headers.get_content_type() != "image/jpeg":
                        raise RuntimeError(f"unexpected content type for {url}")
                    shutil.copyfileobj(response, target, 1024 * 1024)
                if part.stat().st_size < 100 or part.open("rb").read(2) != b"\xff\xd8":
                    raise RuntimeError(f"invalid JPEG for {ident}")
                part.replace(primary)
                existing = primary
                break
            except Exception:
                part.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    for path in destinations:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(existing, path)


def openimages(workers, limit):
    destinations = {}
    for dataset, annotations in OPEN_IMAGES.items():
        for annotation in annotations:
            for ident in checked_ids(dataset, annotation):
                destinations.setdefault(ident, []).append(
                    DATA / dataset / "imgs" / "test" / f"{ident}.jpg"
                )
    items = sorted(destinations.items())[:limit or None]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_image, ident, paths): ident for ident, paths in items}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                failures.append((futures[future], str(error)))
            if completed % 1000 == 0 or completed == len(items):
                print(f"OpenImages: {completed}/{len(items)}, failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"OpenImages incomplete; first failures: {failures[:5]}")


def download_archive(url, path, workers):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == IMAGENET_BYTES:
        return
    part = path.with_suffix(path.suffix + ".part")
    state = path.with_suffix(path.suffix + ".ranges.json")
    if path.is_file() and not part.exists():
        path.replace(part)
    chunk_size = 8 * 1024 * 1024
    count = (IMAGENET_BYTES + chunk_size - 1) // chunk_size
    completed = set(json.loads(state.read_text())) if state.is_file() else set()
    descriptor = os.open(part, os.O_RDWR | os.O_CREAT)
    os.ftruncate(descriptor, IMAGENET_BYTES)

    def fetch(index):
        begin = index * chunk_size
        end = min(begin + chunk_size, IMAGENET_BYTES) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != \
                            f"bytes {begin}-{end}/{IMAGENET_BYTES}":
                        raise RuntimeError(f"bad Range response for {index}")
                    data = response.read()
                if len(data) != end - begin + 1:
                    raise RuntimeError(f"short Range response for {index}")
                os.pwrite(descriptor, data, begin)
                return index
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, index): index for index in range(count) if index not in completed}
            for future in concurrent.futures.as_completed(futures):
                completed.add(future.result())
                if len(completed) % 20 == 0 or len(completed) == count:
                    state.write_text(json.dumps(sorted(completed)))
                    print(f"ImageNet: {len(completed)}/{count} ranges", flush=True)
    finally:
        os.close(descriptor)
    digest = hashlib.md5()
    with part.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != IMAGENET_MD5:
        raise RuntimeError(f"ImageNet MD5 mismatch: {digest.hexdigest()}")
    part.replace(path)
    state.unlink(missing_ok=True)


def extract_selected(archive, destination, allowed):
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            name = Path(member.name).name
            if name not in allowed or not member.isfile():
                continue
            target = destination / name
            if target.is_file() and target.stat().st_size == member.size:
                extracted += 1
                continue
            content = source.extractfile(member)
            if content is None:
                continue
            part = target.with_suffix(target.suffix + ".part")
            with part.open("wb") as output:
                shutil.copyfileobj(content, output, 1024 * 1024)
            part.replace(target)
            extracted += 1
    if extracted != len(allowed):
        raise RuntimeError(f"{archive}: extracted {extracted}/{len(allowed)} annotated images")


def imagenet(workers, keep_archive=False):
    archive = DATA / "ILSVRC2012_img_val.tar"
    allowed = {line.split(",", 1)[0] for line in
               (DATA / "imagenet_multi/imagenet_multi_1000_annots.txt").read_text().splitlines()}
    destination = DATA / "imagenet_multi/imgs"
    if all((destination / name).is_file() for name in allowed):
        return
    download_archive(IMAGENET_URL, archive, workers)
    extract_selected(archive, destination, allowed)
    if not keep_archive:
        archive.unlink(missing_ok=True)


def hico(keep_archive=False):
    archive = DATA / "hico_20150920.tar.gz"
    allowed = {line.split(",", 1)[0] for line in
               (DATA / "hico/hico_600_annots.txt").read_text().splitlines()}
    destination = DATA / "hico/imgs"
    if all((destination / name).is_file() for name in allowed):
        return
    if not archive.is_file():
        subprocess.run(["gdown", "--continue", HICO_DRIVE_ID, "-O", str(archive)], check=True)
    extract_selected(archive, destination, allowed)
    if not keep_archive:
        archive.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("all", "openimages", "imagenet", "hico"), default="all", nargs="?")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, help="OpenImages smoke-test count")
    parser.add_argument("--keep-archives", action="store_true",
                        help="retain ImageNet/HICO source archives after extraction")
    args = parser.parse_args()
    if args.dataset in ("all", "openimages"):
        openimages(args.workers, args.limit)
    if args.dataset in ("all", "imagenet"):
        imagenet(args.workers, args.keep_archives)
    if args.dataset in ("all", "hico"):
        hico(args.keep_archives)


if __name__ == "__main__":
    main()

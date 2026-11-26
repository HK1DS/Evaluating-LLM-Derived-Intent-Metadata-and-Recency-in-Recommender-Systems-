"""Download the inputs and launch the pinned dataset-specific experiment."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
import zipfile

ROOT = Path(__file__).resolve().parent
UCSD = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
GOODREADS = {
    "byGenre/goodreads_interactions_children.json.gz":
        (511288540, "d8ce59259e45262b3322bd87f5c2a13228cd8bb659332414a05660aed7cd7c71"),
    "byGenre/goodreads_books_children.json.gz":
        (102502706, "c0b61eb817c1c124967beb5f3e42236220f58f7e207212ce217990ebd3d7d57d"),
    "goodreads_book_authors.json.gz":
        (17988861, "85e6628dee5102c70a4a296f6d68c10ff60b0cf72714444b1e721abeffe9f22d"),
}
MOVIELENS = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_SHA256 = "8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa"
ML_FILES = {
    "ratings.csv": (678260987, "9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385"),
    "movies.csv": (3038099, "c50dea6973e619f9bc1decd4afa3e35358b83a5fb663dac0ec82252a011d023d"),
}
STAGES = {"prepare": "0ATPS", "run": "0ATPSBCDER", "evaluate": "ER"}
GR_SPECS = ("IKGR_kgoff,BPR,LightGCN,IKGR_dyn_nointent,IKGR_dyn,"
            "IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn")


def verify(path, size, digest):
    if path.stat().st_size != size:
        raise ValueError(f"{path}: expected {size:,} bytes, found {path.stat().st_size:,}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != digest:
        raise ValueError(f"{path}: SHA-256 mismatch")


def fetch(url, dest, size, digest):
    if dest.exists():
        verify(dest, size, digest)
        print(f"Verified {dest.name}", flush=True)
        return
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ({size:,} bytes)", flush=True)
    with NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with requests.get(url, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with temp.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
        verify(temp, size, digest)
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)


def download(dataset, folder):
    if dataset == "goodreads":
        for source, (size, digest) in GOODREADS.items():
            fetch(UCSD + source, folder / Path(source).name, size, digest)
        return
    raw = folder / "raw"
    missing = []
    for name, expected in ML_FILES.items():
        try:
            verify(raw / "ml-25m" / name, *expected)
        except (FileNotFoundError, ValueError):
            missing.append(name)
    if not missing:
        print("Verified ratings.csv and movies.csv", flush=True)
        return
    archive = raw / "ml-25m.zip"
    fetch(MOVIELENS, archive, 261978986, ML_SHA256)
    with zipfile.ZipFile(archive) as z:
        for name in missing:
            dest = raw / "ml-25m" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as tmp:
                temp = Path(tmp.name)
            try:
                with z.open("ml-25m/" + name) as src, temp.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                verify(temp, *ML_FILES[name])
                temp.replace(dest)
            finally:
                temp.unlink(missing_ok=True)
    print(f"Extracted ratings.csv and movies.csv in {raw / 'ml-25m'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("goodreads", "movielens"))
    parser.add_argument("action", choices=("download", *STAGES))
    parser.add_argument("--split", choices=("TO", "TO_GLOBAL"), default="TO")
    parser.add_argument("--seeds", default="2020,2021,2022")
    parser.add_argument("--max-usd", type=float, help="Reject a run above the upstream cost estimate")
    args = parser.parse_args()
    if args.max_usd is not None and (not math.isfinite(args.max_usd) or args.max_usd < 0):
        parser.error("--max-usd must be a finite, nonnegative amount")
    if not all(s.isascii() and s.isdecimal() and int(s) < 2**32 for s in args.seeds.split(",")):
        parser.error("--seeds must be comma-separated integers between 0 and 4294967295")
    folder = ROOT / "experiments" / args.dataset
    if args.action != "evaluate":
        download(args.dataset, folder)
    if args.action == "download":
        return 0
    cmd = [sys.executable, "-u", "run_pipeline.py", "--k", "30", "--split", args.split,
           "--steps", STAGES[args.action], "--seeds", args.seeds]
    if args.dataset == "goodreads":
        cmd.extend(["--specs", GR_SPECS])
    if args.max_usd is not None:
        cmd.extend(["--max_usd", str(args.max_usd)])
    env = dict(os.environ, PYTHONUTF8="1")
    code = subprocess.run(cmd, cwd=folder, env=env).returncode
    if code == 0 and args.dataset == "movielens" and args.split == "TO_GLOBAL" and args.action in ("run", "evaluate"):
        report = [sys.executable, "paired_compare.py", "--run_dir", "run_k30_TO_GLOBAL",
                  "--split", "TO_GLOBAL", "--report", "--min_activity", "1"]
        return subprocess.run(report, cwd=folder, env=env).returncode
    return code


if __name__ == "__main__":
    raise SystemExit(main())

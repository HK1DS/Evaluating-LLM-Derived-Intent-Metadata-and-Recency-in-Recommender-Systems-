import hashlib
import io
import sys
import zipfile

import pytest
import requests

import reproduce


def checksum(data):
    return len(data), hashlib.sha256(data).hexdigest()


def test_fetch_discards_corrupt_download(tmp_path, monkeypatch):
    response = requests.Response()
    response.status_code = 200
    response.raw = io.BytesIO(b"bad")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="SHA-256"):
        reproduce.fetch("https://example.invalid/input", tmp_path / "input", *checksum(b"abc"))
    assert list(tmp_path.iterdir()) == []


def test_extract_movies_and_reuse_offline(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    archive = raw / "ml-25m.zip"
    data = {"ratings.csv": b"userId,movieId,rating,timestamp\n", "movies.csv": b"movieId,title,genres\n"}
    with zipfile.ZipFile(archive, "w") as z:
        for name, content in data.items():
            z.writestr("ml-25m/" + name, content)
    monkeypatch.setattr(reproduce, "ML_FILES", {name: checksum(content) for name, content in data.items()})
    monkeypatch.setattr(reproduce, "fetch", lambda *args: None)
    reproduce.download("movielens", tmp_path)
    for name, content in data.items():
        assert (raw / "ml-25m" / name).read_bytes() == content
    monkeypatch.setattr(reproduce, "fetch", lambda *args: pytest.fail("unexpected download"))
    reproduce.download("movielens", tmp_path)


def test_prepare_is_free_and_evaluate_builds_warm_report(monkeypatch):
    downloads, calls = [], []
    monkeypatch.setattr(reproduce, "download", lambda dataset, folder: downloads.append(dataset))
    def run(cmd, cwd, env):
        calls.append((cmd, cwd.name))
        return reproduce.subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(reproduce.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "goodreads", "prepare"])
    assert reproduce.main() == 0
    [(cmd, folder)] = calls
    assert folder == "goodreads" and cmd[cmd.index("--steps") + 1] == "0ATPS"
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "movielens", "evaluate", "--split", "TO_GLOBAL"])
    assert reproduce.main() == 0
    assert downloads == ["goodreads"]
    assert [cmd[1:3] for cmd, _ in calls] == [["-u", "run_pipeline.py"], ["paired_compare.py", "--run_dir"]]
    assert calls[-1][0][-2:] == ["--min_activity", "1"]

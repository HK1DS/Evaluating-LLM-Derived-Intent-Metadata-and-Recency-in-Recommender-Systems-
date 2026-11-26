"""CPU-only checks of scripts/run_ablation_nov22.py (no GPU, data or eval_slices import needed)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("run_ablation_nov22", ROOT / "scripts" / "run_ablation_nov22.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_specs_flip_exactly_one_switch():
    mod = _load()
    base = {"use_kg": False, "kg_cap": 32, "use_meta_kg": True, "meta_kg_path": "m.pt", "use_dynamic": True}
    out = mod.derive_specs({"IKGR_dyn_nointent": ("IKGR", dict(base)), "BPR": ("BPR", {})})
    meta, recency = mod.SPEC_NAMES
    assert out[meta] == ("IKGR", dict(base, use_dynamic=False))
    assert out[recency] == ("IKGR", dict(base, use_meta_kg=False))
    assert out["IKGR_dyn_nointent"] == ("IKGR", base)          # the source spec is untouched


def test_jobs_cover_two_specs_three_seeds_seed2020_first():
    mod = _load()
    assert len(mod.JOBS) == 6 and len(set(mod.JOBS)) == 6
    assert [seed for _, seed in mod.JOBS[:2]] == [2020, 2020]


def test_code_hash_ignores_line_endings(tmp_path):
    mod = _load()
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (tmp_path / "b.py").write_bytes(b"x = 1\ny = 2\n")
    assert mod.normalized_sha256(tmp_path / "a.py") == mod.normalized_sha256(tmp_path / "b.py")


def test_this_checkout_matches_published_code():
    mod = _load()
    assert mod.code_mismatches() == {}


def test_environment_pins_hash_seed_and_specs():
    mod = _load()
    env = mod.environment("IKGR_recency_only_nov22", 2021)
    assert env["PYTHONHASHSEED"] == "2021" and env["IKGR_SEEDS"] == "2021"
    assert env["IKGR_SPECS"] == "IKGR_recency_only_nov22" and env["IKGR_SPLIT"] == "TO"
    assert env["IKGR_DUMP_VALID"] == "1" and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

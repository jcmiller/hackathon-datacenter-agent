"""Canonical Kalos/Xid source resolution + fail-loud validation (bead b5u).

These cover gpusitter.telemetry.sources — the single home for resolving a metric
name to a validated, readable CSV path (worktree file or LFS-cache object), and
for rejecting anything that is not a wide Time+GPU telemetry frame.
"""

import subprocess

import pytest

from gpusitter.telemetry import sources

# A wide time-series CSV the validator must accept.
GOOD_CSV = "Time,172.31.0.5-3,172.31.0.6-1\n2023-08-15 15:30:00+08:00,0.0,43.0\n"

OID = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
POINTER = f"version https://git-lfs.github.com/spec/v1\noid sha256:{OID}\nsize 11\n"


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


def _kalos_dir(repo):
    d = repo / "data" / "utilization" / "kalos"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- DCGM vocabulary -------------------------------------------------------------


def test_dcgm_field_to_metric_map():
    assert sources.DCGM_FIELD_TO_METRIC["DCGM_FI_DEV_POWER_USAGE"] == "POWER_USAGE"
    assert sources.DCGM_FIELD_TO_METRIC["DCGM_FI_DEV_GPU_TEMP"] == "GPU_TEMP"
    assert "XID_ERRORS" in sources.CANONICAL_METRICS


def test_metric_csv_relpath():
    assert sources.metric_csv_relpath("XID_ERRORS") == "data/utilization/kalos/XID_ERRORS.csv"


# --- resolve_metric_csv ----------------------------------------------------------


def test_resolve_metric_csv_materialized_worktree(tmp_path):
    repo = tmp_path / "repo"
    (_kalos_dir(repo) / "POWER_USAGE.csv").write_text(GOOD_CSV)
    resolved = sources.resolve_metric_csv("POWER_USAGE", repo_dir=repo)
    assert resolved == str(repo / "data/utilization/kalos/POWER_USAGE.csv")


def test_resolve_metric_csv_from_lfs_cache(tmp_path):
    # Worktree holds only a pointer; real CSV lives in the LFS cache object.
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "t")
    (_kalos_dir(repo) / "GPU_TEMP.csv").write_text(POINTER)
    cache = repo / ".git" / "lfs" / "objects" / OID[:2] / OID[2:4] / OID
    cache.parent.mkdir(parents=True)
    cache.write_text(GOOD_CSV)  # the materialized object IS the CSV content
    _git(repo, "add", "data/utilization/kalos/GPU_TEMP.csv")
    _git(repo, "commit", "-m", "pointer")
    assert sources.resolve_metric_csv("GPU_TEMP", repo_dir=repo) == str(cache)


def test_resolve_metric_csv_missing_raises_file_not_found(tmp_path):
    # Neither worktree file nor pointer/cache -> operational "not materialized".
    repo = tmp_path / "repo"
    _kalos_dir(repo)  # dir exists, file does not
    with pytest.raises(FileNotFoundError):
        sources.resolve_metric_csv("POWER_USAGE", repo_dir=repo)


def test_resolve_metric_csv_uses_module_repo_dir(tmp_path, monkeypatch):
    # repo_dir omitted -> reads the module REPO_DIR seam at call time.
    repo = tmp_path / "repo"
    (_kalos_dir(repo) / "GPU_UTIL.csv").write_text(GOOD_CSV)
    monkeypatch.setattr(sources, "REPO_DIR", repo)
    assert sources.resolve_metric_csv("GPU_UTIL") == str(
        repo / "data/utilization/kalos/GPU_UTIL.csv"
    )


# --- validate_timeseries_csv (fail-loud) ----------------------------------------


def test_validate_accepts_wide_timeseries(tmp_path):
    p = tmp_path / "POWER_USAGE.csv"
    p.write_text(GOOD_CSV)
    assert sources.validate_timeseries_csv(str(p)) == str(p)


def test_validate_rejects_pkl(tmp_path):
    # A .pkl is a CDF/distribution artifact, not a time-series frame.
    p = tmp_path / "gpu_temp_kalos.pkl"
    p.write_bytes(b"\x80\x04whatever")
    with pytest.raises(ValueError):
        sources.validate_timeseries_csv(str(p))


def test_validate_rejects_lfs_pointer(tmp_path):
    # An un-materialized pointer is text, not the real CSV content.
    p = tmp_path / "XID_ERRORS.csv"
    p.write_text(POINTER)
    with pytest.raises(ValueError):
        sources.validate_timeseries_csv(str(p))


def test_validate_rejects_header_without_time(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_text("Stamp,n1\n0,5\n")
    with pytest.raises(ValueError):
        sources.validate_timeseries_csv(str(p))


def test_validate_rejects_time_only_header(tmp_path):
    # Time column but no GPU columns is not a usable telemetry frame.
    p = tmp_path / "time_only.csv"
    p.write_text("Time\n2023-08-15 15:30:00+08:00\n")
    with pytest.raises(ValueError):
        sources.validate_timeseries_csv(str(p))

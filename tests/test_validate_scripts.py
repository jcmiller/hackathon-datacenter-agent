"""Cache-safe resolution for scripts/validate_kalos.py + validate_rca.py (epp).

The droplet is cache-only: the working tree holds Git LFS pointers and the real
frames live in the LFS object cache. These cover the resolution seams that wire
both validate scripts through gpusitter.telemetry.sources, replacing the old
byte-size pointer heuristic and the "re-hydrate first" bail.
"""

import subprocess

import pytest
from scripts import validate_kalos, validate_rca

# A wide Time+GPU telemetry CSV the validator accepts (the materialized content).
GOOD_CSV = "Time,172.31.0.5-3,172.31.0.6-1\n2023-08-15 15:30:00+08:00,0.0,43.0\n"

OID = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
POINTER = f"version https://git-lfs.github.com/spec/v1\noid sha256:{OID}\nsize 11\n"


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


def _cache_repo(tmp_path, rel, *, content=GOOD_CSV):
    """A git repo whose worktree holds only an LFS pointer for ``rel`` while the
    real ``content`` lives in the LFS object cache."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "t")
    pointer = repo / rel
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(POINTER)
    cache = repo / ".git" / "lfs" / "objects" / OID[:2] / OID[2:4] / OID
    cache.parent.mkdir(parents=True)
    cache.write_text(content)
    _git(repo, "add", rel)
    _git(repo, "commit", "-m", "pointer")
    return repo, cache


# --- validate_kalos.resolve_sources ---------------------------------------------


def test_kalos_repo_dir_resolves_lfs_cache(tmp_path):
    repo, cache = _cache_repo(tmp_path, "data/utilization/kalos/GPU_TEMP.csv")
    sources = validate_kalos.resolve_sources(repo_dir=str(repo))
    assert sources == {"GPU_TEMP": str(cache)}


def test_kalos_repo_dir_skips_unfetched_metric(tmp_path):
    # GPU_TEMP is fetched; the other metrics have neither worktree file nor cache.
    repo, cache = _cache_repo(tmp_path, "data/utilization/kalos/GPU_TEMP.csv")
    sources = validate_kalos.resolve_sources(repo_dir=str(repo))
    assert set(sources) == {"GPU_TEMP"}  # unfetched metrics silently skipped


def test_kalos_data_dir_materialized_and_skips_non_telemetry(tmp_path):
    d = tmp_path / "kalos"
    d.mkdir()
    (d / "GPU_TEMP.csv").write_text(GOOD_CSV)  # materialized -> kept
    (d / "POWER_USAGE.csv").write_text(POINTER)  # pointer -> skipped (not byte size)
    (d / "GPU_UTIL.csv").write_text("Stamp,x\n0,1\n")  # non-telemetry header -> skipped
    sources = validate_kalos.resolve_sources(data_dir=str(d))
    assert sources == {"GPU_TEMP": str(d / "GPU_TEMP.csv")}


def test_kalos_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError):
        validate_kalos.resolve_sources()
    with pytest.raises(ValueError):
        validate_kalos.resolve_sources(repo_dir="a", data_dir="b")


# --- validate_rca.resolve_rca_paths ---------------------------------------------


def test_rca_repo_dir_resolves_cache_and_trace(tmp_path):
    repo, cache = _cache_repo(tmp_path, "data/utilization/kalos/XID_ERRORS.csv")
    trace = repo / validate_rca.TRACE_RELPATH
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("job_id,state\n1,FAILED\n")  # materialized trace
    xid_path, trace_path = validate_rca.resolve_rca_paths(repo_dir=str(repo))
    assert xid_path == str(cache)
    assert trace_path == str(trace)


def test_rca_explicit_pointer_fails_loud_with_repo_dir_hint(tmp_path):
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(POINTER)  # still an un-materialized pointer
    with pytest.raises(ValueError, match="--repo-dir"):
        validate_rca.resolve_rca_paths(xid=str(xid), trace=str(tmp_path / "trace.csv"))


def test_rca_explicit_materialized_paths(tmp_path):
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(GOOD_CSV)
    trace = tmp_path / "trace.csv"
    trace.write_text("job_id,state\n1,FAILED\n")
    assert validate_rca.resolve_rca_paths(xid=str(xid), trace=str(trace)) == (
        str(xid),
        str(trace),
    )


def test_rca_requires_repo_dir_or_both_explicit(tmp_path):
    with pytest.raises(ValueError, match="--repo-dir"):
        validate_rca.resolve_rca_paths(xid=str(tmp_path / "x.csv"))

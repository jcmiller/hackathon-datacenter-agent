import json
import subprocess
import sys

from scripts.lfs_helper import (
    get_lfs_cache_path,
    raw_data_status,
    resolve_data_path,
)


OID = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    f"oid sha256:{OID}\n"
    "size 11\n"
)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _write_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    data = repo / "data" / "utilization" / "kalos"
    data.mkdir(parents=True)
    pointer_path = data / "GPU_TEMP.csv"
    pointer_path.write_text(POINTER)
    cache_path = repo / ".git" / "lfs" / "objects" / OID[:2] / OID[2:4] / OID
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("hello data\n")
    _git(repo, "add", "data/utilization/kalos/GPU_TEMP.csv")
    _git(repo, "commit", "-m", "add lfs pointer")
    return repo, "data/utilization/kalos/GPU_TEMP.csv", cache_path, pointer_path


def test_resolves_cache_from_pointer_file(tmp_path):
    repo, rel, cache_path, _ = _write_repo(tmp_path)

    assert get_lfs_cache_path(str(repo), rel) == str(cache_path)

    status = raw_data_status(str(repo), rel)
    assert status.working_state == "pointer"
    assert status.cache_state == "present"
    assert status.size == 11


def test_resolves_cache_when_worktree_path_was_deleted(tmp_path):
    repo, rel, cache_path, pointer_path = _write_repo(tmp_path)
    pointer_path.unlink()

    assert get_lfs_cache_path(str(repo), rel) == str(cache_path)

    status = raw_data_status(str(repo), rel)
    assert status.working_state == "missing"
    assert status.cache_state == "present"


def test_resolve_data_path_prefers_materialized_file(tmp_path):
    repo, rel, _, pointer_path = _write_repo(tmp_path)
    pointer_path.write_text("materialized csv\n")

    assert resolve_data_path(str(repo), rel) == str(pointer_path)

    status = raw_data_status(str(repo), rel)
    assert status.working_state == "materialized"
    assert status.resolved_path == str(pointer_path)


def test_cli_kalos_status_json(tmp_path):
    repo, _, _, _ = _write_repo(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/lfs_helper.py",
            "kalos-status",
            str(repo),
            "--metrics",
            "GPU_TEMP",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload[0]["file_path"] == "data/utilization/kalos/GPU_TEMP.csv"
    assert payload[0]["cache_state"] == "present"

"""Fixtures: tiny wide DCGM-style CSVs mirroring the kalos schema.

Real kalos files are 800 MB+ wide frames on the droplet (gitignored, too big to
commit). These fixtures reproduce the structural wrinkles that matter for ingest:
empty cells (idle GPUs), the IP-based id scheme used by 6/7 metrics, the
pod-based scheme used by SM_ACTIVE, and a per-metric time window/column-set
mismatch.
"""

import textwrap

import pytest


# GPU_TEMP — IP-named, 3 GPUs, 2 timestamps, 2 empty cells (idle GPUs).
#   wide: 3 cols x 2 rows = 6 cells; 4 non-empty -> 4 long records.
GPU_TEMP_CSV = textwrap.dedent(
    """\
    Time,172.31.0.5-3,172.31.0.5-4,172.31.0.9-0
    2023-08-15 00:00:00+08:00,50.0,51.0,
    2023-08-15 00:00:15+08:00,52.0,,60.0
    """
)

# POWER_USAGE — IP-named, overlaps GPU_TEMP on 172.31.0.5-3, plus a later sample.
POWER_USAGE_CSV = textwrap.dedent(
    """\
    Time,172.31.0.5-3,172.31.0.9-0
    2023-08-15 00:00:00+08:00,300.0,
    2023-08-15 00:00:15+08:00,310.0,250.0
    2023-08-15 00:00:30+08:00,305.0,255.0
    """
)

# SM_ACTIVE — POD-named. lingjun-pod9-0001-3 is the SAME physical GPU as the
# IP-named 172.31.0.5-3, but only an external alias map can prove that.
SM_ACTIVE_CSV = textwrap.dedent(
    """\
    Time,lingjun-pod9-0001-3,lingjun-pod9-0001-4
    2023-08-15 00:00:00+08:00,0.80,0.10
    2023-08-15 00:00:15+08:00,0.82,
    """
)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


@pytest.fixture
def gpu_temp_csv(tmp_path):
    return _write(tmp_path, "GPU_TEMP.csv", GPU_TEMP_CSV)


@pytest.fixture
def power_usage_csv(tmp_path):
    return _write(tmp_path, "POWER_USAGE.csv", POWER_USAGE_CSV)


@pytest.fixture
def sm_active_csv(tmp_path):
    return _write(tmp_path, "SM_ACTIVE.csv", SM_ACTIVE_CSV)


@pytest.fixture
def pod_to_ip_alias():
    """Maps the pod node name to its physical IP node (the join key)."""
    return {"lingjun-pod9-0001": "172.31.0.5"}

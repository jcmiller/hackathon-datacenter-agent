"""GPU-id normalization: parse + canonicalize + cross-namespace alias."""

from gpusitter.telemetry.normalize import GpuId, parse_gpu_id


def test_parse_ip_id():
    g = parse_gpu_id("172.31.0.5-3")
    assert g == GpuId(node="172.31.0.5", index=3)
    assert g.canonical == "172.31.0.5#3"


def test_parse_pod_id_splits_on_last_dash():
    # pod names contain interior dashes; only the trailing -N is the GPU index.
    g = parse_gpu_id("lingjun-pod9-0016-3")
    assert g == GpuId(node="lingjun-pod9-0016", index=3)
    assert g.canonical == "lingjun-pod9-0016#3"


def test_alias_folds_pod_node_into_ip_node():
    alias = {"lingjun-pod9-0001": "172.31.0.5"}
    pod = parse_gpu_id("lingjun-pod9-0001-3", alias=alias)
    ip = parse_gpu_id("172.31.0.5-3")
    # The whole point of normalization: differently-named -> one canonical GPU.
    assert pod == ip
    assert pod.canonical == "172.31.0.5#3"


def test_without_alias_namespaces_stay_distinct():
    # Non-vacuity guard for the join test: absent an alias they must NOT merge.
    pod = parse_gpu_id("lingjun-pod9-0001-3")
    ip = parse_gpu_id("172.31.0.5-3")
    assert pod != ip


def test_gpu_id_is_hashable_and_str_is_canonical():
    g = parse_gpu_id("172.31.0.5-3")
    assert str(g) == "172.31.0.5#3"
    assert len({g, parse_gpu_id("172.31.0.5-3")}) == 1

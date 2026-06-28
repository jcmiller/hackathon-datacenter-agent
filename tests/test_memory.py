from gpusitter.agent.memory import append_incident, search_incidents


def test_search_empty_when_no_file(tmp_path):
    assert search_incidents("train", str(tmp_path / "sop.json")) == []


def test_append_then_search_roundtrip(tmp_path):
    p = str(tmp_path / "sop.json")
    append_incident(
        {
            "type": "train",
            "summary": "NODE_FAIL on 4 nodes",
            "disposition": "page_technician",
            "resolution": "replaced GPU",
        },
        p,
    )
    append_incident(
        {"type": "eval", "summary": "other", "disposition": "restart", "resolution": "ok"}, p
    )
    hits = search_incidents("train", p)
    assert len(hits) == 1
    assert hits[0]["resolution"] == "replaced GPU"


def test_append_creates_parent_dir(tmp_path):
    p = str(tmp_path / "nested" / "sop.json")
    append_incident({"type": "train", "summary": "s", "disposition": "d", "resolution": "r"}, p)
    assert search_incidents("train", p)[0]["resolution"] == "r"

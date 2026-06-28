import backend.tools as tools

def test_get_telemetry(tmp_path, monkeypatch):
    import pandas as pd
    pw = tmp_path/"power.csv"; tp = tmp_path/"temp.csv"
    pd.DataFrame({"Time":[100,110,200],"n1":[10,90,10]}).to_csv(pw, index=False)
    pd.DataFrame({"Time":[100,110,200],"n1":[40,42,41]}).to_csv(tp, index=False)
    monkeypatch.setattr(tools, "POWER_CSV", str(pw))
    monkeypatch.setattr(tools, "TEMP_CSV", str(tp))
    out = tools.get_telemetry(fail_time=110, window=20)
    assert out["DCGM_FI_DEV_POWER_USAGE"]["max"] == 90
    assert out["DCGM_FI_DEV_GPU_TEMP"]["samples"] == 2

def test_find_correlated_failures(tmp_path, monkeypatch):
    import pandas as pd
    p = tmp_path/"trace.csv"
    pd.DataFrame([
        {"job_id":"a","type":"train","node_num":1,"gpu_num":8,"state":"NODE_FAIL",
         "fail_time":1000,"duration":1,"user":"u"},
        {"job_id":"b","type":"train","node_num":1,"gpu_num":8,"state":"NODE_FAIL",
         "fail_time":1050,"duration":1,"user":"u"},
    ]).to_csv(p, index=False)
    monkeypatch.setattr(tools, "TRACE_CSV", str(p))
    out = tools.find_correlated_failures(fail_time=1000, window=120)
    assert out["count"] == 1 and out["jobs"] == ["b"] and out["shared_type"] == "train"

def test_page_and_record(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SOP_PATH", str(tmp_path/"sop.json"))
    assert tools.page_technician("node 4", "drain+replace")["paged"] is True
    assert tools.record_resolution("train","s","page_technician","replaced")["recorded"] is True
    assert tools.search_past_incidents("train")["count"] == 1

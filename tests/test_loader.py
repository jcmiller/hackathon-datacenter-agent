import pandas as pd
from src.loader import load_incidents, telemetry_window, correlated_failures

def _write_trace(tmp_path):
    df = pd.DataFrame([
        {"job_id":"j1","user":"u1","node_num":4,"gpu_num":32,"cpu_num":64,"type":"train",
         "state":"NODE_FAIL","submit_time":100,"start_time":110,"end_time":900,
         "duration":790,"queue":10,"gpu_time":1,"mem_per_pod_GB":80,"shared_mem_per_pod":1,
         "fail_time":880,"stop_time":900},
        {"job_id":"j2","user":"u2","node_num":1,"gpu_num":8,"cpu_num":16,"type":"eval",
         "state":"COMPLETED","submit_time":100,"start_time":110,"end_time":500,
         "duration":390,"queue":10,"gpu_time":1,"mem_per_pod_GB":80,"shared_mem_per_pod":1,
         "fail_time":None,"stop_time":500},
        {"job_id":"j3","user":"u3","node_num":2,"gpu_num":16,"cpu_num":32,"type":"train",
         "state":"FAILED","submit_time":100,"start_time":110,"end_time":920,
         "duration":810,"queue":10,"gpu_time":1,"mem_per_pod_GB":80,"shared_mem_per_pod":1,
         "fail_time":900,"stop_time":920},
    ])
    p = tmp_path/"trace.csv"; df.to_csv(p, index=False); return str(p)

def _write_util(tmp_path):
    df = pd.DataFrame({"Time":[870,885,895,910],
                       "10.0.0.1":[50,90,95,40],"10.0.0.2":[60,70,80,30]})
    p = tmp_path/"util.csv"; df.to_csv(p, index=False); return str(p)

def test_load_incidents_filters_and_sorts(tmp_path):
    inc = load_incidents(_write_trace(tmp_path))
    assert [i["job_id"] for i in inc] == ["j1","j3"]      # COMPLETED dropped, sorted by fail_time
    assert inc[0]["fail_time"] == 880

def test_telemetry_window_aggregates(tmp_path):
    w = telemetry_window(_write_util(tmp_path), 880, 900)
    assert w["samples"] == 2                               # rows at 885 and 895
    assert w["max"] == 95
    assert w["min"] == 70

def test_correlated_failures_within_window(tmp_path):
    inc = load_incidents(_write_trace(tmp_path))
    corr = correlated_failures(inc, fail_time=880, window=30)
    assert [c["job_id"] for c in corr] == ["j3"]           # j3 at 900 within ±30, j1 excluded (self)

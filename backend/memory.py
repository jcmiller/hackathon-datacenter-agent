import json
import os

def search_incidents(incident_type: str, path: str = "data/sop.json") -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        records = json.load(f)
    return [r for r in records if r.get("type") == incident_type]

def append_incident(record: dict, path: str = "data/sop.json") -> None:
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
    records.append(record)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

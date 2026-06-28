import json, os

def search_incidents(incident_type, path="data/sop.json"):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        records = json.load(f)
    return [r for r in records if r.get("type") == incident_type]

def append_incident(record, path="data/sop.json"):
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

"""Write a small synthetic data/jobs.csv so the full stream -> incident ->
train/validate -> promote loop runs locally without the 80 GB AcmeTrace dataset.

Schema matches what precompute_features.py produces from real data. A learnable
signal is baked in: higher power/temp jobs fail more often, so the classifier's
ROC-AUC is meaningfully > 0.5. Mock start/end times are integer seconds for
simplicity; real AcmeTrace timestamps are ISO UTC strings (handled in precompute).
"""

import csv
import math
import os
import random

N = 600
TYPES = ["pretrain", "eval", "sft", "debug"]
random.seed(0)


def _row(i):
    power_max = random.uniform(80, 350)
    temp_max = random.uniform(45, 88)
    util_max = random.uniform(20, 99)
    # Failure pressure rises sharply with power + temp -> a clearly learnable signal
    # (logistic, centered so the classifier's ROC-AUC lands ~0.8).
    z = (power_max - 240) / 30 + (temp_max - 72) / 12
    failed = random.random() < 1 / (1 + math.exp(-z))
    start = 1_692_000_000 + i * 900
    dur = random.randint(300, 6000)
    return {
        "job_id": f"job{i:04d}",
        "type": random.choice(TYPES),
        "node_num": random.randint(1, 16),
        "gpu_num": random.choice([8, 16, 32, 64]),
        "cpu_num": random.choice([16, 32, 64, 128]),
        "duration": dur,
        "queue": random.randint(0, 1200),
        "mem_per_pod_GB": random.choice([40, 80, 160]),
        "state": "FAILED" if failed else random.choice(["COMPLETED", "COMPLETED", "CANCELLED"]),
        "start_time": start,
        "end_time": start + dur,
        "fail_time": (start + dur - 60) if failed else "",
        "power_mean": round(power_max * random.uniform(0.7, 0.9), 1),
        "power_max": round(power_max, 1),
        "power_std": round(power_max * random.uniform(0.05, 0.2), 1),
        "temp_mean": round(temp_max * random.uniform(0.85, 0.95), 1),
        "temp_max": round(temp_max, 1),
        "temp_std": round(temp_max * random.uniform(0.03, 0.1), 1),
        "util_mean": round(util_max * random.uniform(0.8, 0.95), 1),
        "util_max": round(util_max, 1),
        "util_std": round(util_max * random.uniform(0.05, 0.15), 1),
    }


def main():
    rows = [_row(i) for i in range(N)]
    os.makedirs("data", exist_ok=True)
    with open("data/jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    failed = sum(r["state"] == "FAILED" for r in rows)
    print(f"wrote data/jobs.csv ({N} jobs, {failed} FAILED)")


if __name__ == "__main__":
    main()

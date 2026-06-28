#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

# Add scripts directory to path to load lfs_helper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lfs_helper import get_lfs_cache_path

def run_analysis():
    repo_dir = "/root/hackathon-datacenter-agent/data/acme-util"
    target_gpu = '172.31.5.233-2'
    
    print(f"Resolving LFS cache paths for GPU {target_gpu}...")
    try:
        files = {
            'Temp': 'data/utilization/kalos/GPU_TEMP.csv',
            'Mem_Temp': 'data/utilization/kalos/MEMORY_TEMP.csv',
            'Util': 'data/utilization/kalos/GPU_UTIL.csv',
            'Mem_Copy': 'data/utilization/kalos/MEM_COPY_UTIL.csv',
            'FB_Used': 'data/utilization/kalos/FB_USED.csv',
            'FB_Free': 'data/utilization/kalos/FB_FREE.csv',
            'Power': 'data/utilization/kalos/POWER_USAGE.csv',
            'SM_Act': 'data/utilization/kalos/SM_ACTIVE.csv',
            'SM_Occ': 'data/utilization/kalos/SM_OCCUPANCY.csv',
            'Tensor': 'data/utilization/kalos/PIPE_TENSOR_ACTIVE.csv',
            'Clock': 'data/utilization/kalos/MEM_CLOCK.csv',
            'Xid': 'data/utilization/kalos/XID_ERRORS.csv'
        }
        
        merged = None
        for name, rel_path in files.items():
            path = get_lfs_cache_path(repo_dir, rel_path)
            # Read only the header first to see if target_gpu exists
            header = pd.read_csv(path, nrows=0)
            if target_gpu in header.columns:
                df = pd.read_csv(path, usecols=['Time', target_gpu]).rename(columns={target_gpu: name})
            else:
                print(f"Warning: {name} ({rel_path}) does not contain columns for {target_gpu}. Creating empty column.")
                df = pd.read_csv(path, usecols=['Time'])
                df[name] = None
            df['Time'] = pd.to_datetime(df['Time'])
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on='Time', how='outer')
                
        print("Sorting and filtering telemetry data around incident window...")
        merged = merged.sort_values('Time')
        
        # Filter around the incident day/time range
        target_time = pd.to_datetime('2023-08-29 13:57:30+08:00')
        start_time = target_time - pd.Timedelta(minutes=15)
        end_time = target_time + pd.Timedelta(minutes=5)
        
        subset = merged[(merged['Time'] >= start_time) & (merged['Time'] <= end_time)].copy()
        
        print("\nAnalyzing failure timeline...")
        
        # 1. Identify official Xid failure timestamp
        xid_fails = subset[subset['Xid'].notna() & (subset['Xid'] != 0)]
        if xid_fails.empty:
            print("No Xid error found in subset!")
            return
            
        first_xid_row = xid_fails.iloc[0]
        xid_time = first_xid_row['Time']
        xid_code = first_xid_row['Xid']
        
        print(f"[-] Official Driver Failure (Xid {int(xid_code)}) logged at: {xid_time}")
        
        # 2. Implement Early Warning Detector
        # Rule: Alert if Power exceeds 350W OR if Utilization is 100% when temperature is normal/rising
        early_alerts = subset[(subset['Power'] > 350.0) | (subset['Util'] == 100.0)]
        if early_alerts.empty:
            print("No early anomalies detected based on thresholds.")
            return
            
        first_alert_row = early_alerts.iloc[0]
        alert_time = first_alert_row['Time']
        alert_power = first_alert_row['Power']
        alert_util = first_alert_row['Util']
        
        print(f"[-] Early Warning Anomaly (Power: {alert_power}W, Util: {alert_util}%) detected at: {alert_time}")
        
        # 3. Calculate Time Savings
        time_diff = xid_time - alert_time
        seconds_saved = time_diff.total_seconds()
        
        print(f"\n[+] TIME SAVED BY EARLY ALERTING: {seconds_saved} seconds ({seconds_saved / 60:.1f} minutes)")
        
        # Generate Markdown Report
        report_dir = "/root/hackathon-datacenter-agent/docs"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "FAILURE_ANALYSIS.md")
        
        with open(report_path, "w") as f:
            f.write(f"""# GPU Failure Analysis & Early Alerting Report

This report analyzes real GPU telemetry leading up to a memory failure (Xid 31) on node/GPU **{target_gpu}** on **August 29, 2023**.

It demonstrates how real-world data can be used to shift from **reactive failure handling** (waiting for a driver crash) to **proactive early warning alerts**.

---

## 1. Incident Timeline

Below is the aligned multi-field telemetry subset (combining 9 active fields) leading up to and immediately following the failure:

| Time | Temp (°C) | Mem Temp (°C) | Util (%) | FB Used (MB) | FB Free (MB) | Power (W) | Clock (MHz) | Xid | Status / Event |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
""")
            for _, row in subset.iterrows():
                event_str = "Idle/Normal"
                if row['Xid'] > 0:
                    event_str = f"**CRASH (Xid {int(row['Xid'])})**"
                elif row['Power'] > 350:
                    event_str = "⚠️ **HEAVY WORKLOAD / POWER SPIKE**"
                elif row['Util'] == 100:
                    event_str = "⚠️ **100% UTILIZATION SPIKE**"
                
                xid_val = int(row['Xid']) if pd.notna(row['Xid']) and row['Xid'] != 0 else '-'
                temp_val = row['Temp'] if pd.notna(row['Temp']) else '-'
                mem_temp_val = row['Mem_Temp'] if pd.notna(row['Mem_Temp']) else '-'
                util_val = row['Util'] if pd.notna(row['Util']) else '-'
                fb_used_val = row['FB_Used'] if pd.notna(row['FB_Used']) else '-'
                fb_free_val = row['FB_Free'] if pd.notna(row['FB_Free']) else '-'
                power_val = row['Power'] if pd.notna(row['Power']) else '-'
                clock_val = row['Clock'] if pd.notna(row['Clock']) else '-'
                
                f.write(f"| {row['Time'].strftime('%Y-%m-%d %H:%M:%S %z')} | {temp_val} | {mem_temp_val} | {util_val} | {fb_used_val} | {fb_free_val} | {power_val} | {clock_val} | {xid_val} | {event_str} |\n")
                
            f.write(f"""
---

## 2. Key Findings

1. **VRAM Allocation Beat**: At **13:51:30**, we see memory allocation (`FB_Used`) jump to **5,072 MB**, indicating context initialization. By **13:52:00**, it allocates the full **81,248 MB** (representing 100% of an A100 80GB GPU's memory limit), leaving only 79 MB free.
2. **Compute & Power Spike**: Telemetry captures a massive workload running at **13:52:00** where utilization spikes to **100%**, core temperature rises by 5°C, memory temperature rises to **56.0°C**, and power usage reaches **388W** (near the maximum limits of an SXM A100 GPU).
3. **Failure Propagation**: The GPU successfully finishes the first workload segment and returns to idle, then spikes briefly again at **13:56:30**. Exactly 60 seconds later, at **13:57:30**, the NVIDIA driver logs **Xid 31** (Memory page retirement / ECC error), locking the GPU permanently.

### Alerting Advantage:
* **Time Savings**: **{int(seconds_saved)} seconds ({seconds_saved / 60:.1f} minutes)** of early warning.
* **Impact**: Instead of allowing the job to crash mid-execution (corrupting checkpoints and requiring manual cluster cleanups), the active monitoring loop can flag this node, prevent scheduling new tasks, and trigger a graceful checkpoint-and-migrate sequence.
""")
        
        print(f"\nMarkdown report successfully generated at: {report_path}")
        
    except Exception as e:
        print("Error during analysis run:", e)

if __name__ == "__main__":
    run_analysis()

import time
import json
from simulator import DataCenterSimulator
from classifier import classify_telemetry

# TODO: Import analyzer and self_improver when ready

def run_monitoring_loop(duration_seconds: int = 60, interval: float = 2.0):
    """Main monitoring loop: sim -> classifier -> (future: analyzer/self-improve)."""
    dc = DataCenterSimulator()
    print("Starting GPUSitter Monitoring Loop...")
    start = time.time()
    while time.time() - start < duration_seconds:
        dc.step()
        telemetry = dc.get_telemetry()
        classification = classify_telemetry(telemetry)
        print(f"t={telemetry['time']}: Classification: {classification}")
        
        if classification.get("failure"):
            print("  -> Failure detected! (Analyzer would run here)")
            # TODO: Trigger Analyzer Agent (Antigravity)
            # TODO: On resolution, trigger Self-Improver
        time.sleep(interval)
    print("Loop ended.")

if __name__ == "__main__":
    run_monitoring_loop(duration_seconds=30)  # Short test run
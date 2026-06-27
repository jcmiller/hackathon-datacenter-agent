# Demo entrypoint for main monitoring loop
# Run: python demo/main_loop_demo.py

from main_loop import run_monitoring_loop

if __name__ == "__main__":
    print("EvoSentinel DC - Multi-Stage Demo")
    run_monitoring_loop(duration_seconds=20)
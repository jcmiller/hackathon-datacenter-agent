# Enhanced Data Center Simulator with DCGM-inspired GPU metrics

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class GPUComponent:
    name: str = "GPU"
    temp: float = 40.0
    util: float = 0.0
    power_draw: float = 100.0
    ecc_errors: int = 0
    clock: float = 1000.0  # MHz
    status: str = "healthy"

class Server:
    def __init__(self, id: int):
        self.id = id
        self.gpu = GPUComponent()
        self.load: float = 0.5
        self.voltage: float = 220.0

    def update(self, dt: float = 1.0, env_factor: float = 1.0):
        # Realistic GPU dynamics (DCGM-like)
        self.gpu.util = min(100, self.load * 100 * env_factor)
        heat_gen = self.gpu.util / 10 * dt * env_factor
        cooling = 3 * dt
        self.gpu.temp += heat_gen - cooling
        self.gpu.temp = max(30, min(95, self.gpu.temp))
        
        self.gpu.power_draw = 100 + self.gpu.util * 3.5
        if self.gpu.temp > 80:
            self.gpu.status = "overheating"
            self.gpu.clock = max(800, self.gpu.clock * 0.95)  # Throttling
        else:
            self.gpu.status = "healthy"
            self.gpu.clock = min(1500, self.gpu.clock + 5)
        
        # ECC simulation
        if random.random() < 0.02 * (self.gpu.temp - 60) / 20:
            self.gpu.ecc_errors += 1

class DataCenterSimulator:
    def __init__(self):
        self.servers = [Server(i) for i in range(4)]
        self.time = 0
        self.incidents = []
        self.env_mode = "normal"  # normal, arid (dust), coastal (humidity)

    def set_env_mode(self, mode: str):
        self.env_mode = mode

    def step(self):
        self.time += 1
        env_factor = 1.2 if self.env_mode == "arid" else 1.0
        for s in self.servers:
            s.update(env_factor=env_factor)
        # Inject failures
        if random.random() < 0.15:
            self.inject_failure()

    def inject_failure(self):
        server = random.choice(self.servers)
        failure_type = random.choice(["overheat", "voltage", "ecc", "power"])
        if failure_type == "overheat":
            server.gpu.temp += 25
        elif failure_type == "voltage":
            server.voltage -= 30
        elif failure_type == "ecc":
            server.gpu.ecc_errors += 10
        else:
            server.gpu.power_draw += 150
        self.incidents.append(f"{failure_type} at t={self.time} (env={self.env_mode})")

    def get_telemetry(self) -> Dict:
        return {
            "time": self.time,
            "env_mode": self.env_mode,
            "servers": [{
                "id": s.id,
                "load": s.load,
                "voltage": s.voltage,
                "gpu_temp": round(s.gpu.temp, 1),
                "gpu_util": round(s.gpu.util, 1),
                "power_draw": round(s.gpu.power_draw, 1),
                "ecc_errors": s.gpu.ecc_errors,
                "gpu_clock": round(s.gpu.clock, 0),
                "status": s.gpu.status
            } for s in self.servers],
            "recent_incidents": self.incidents[-3:]
        }

if __name__ == "__main__":
    dc = DataCenterSimulator()
    dc.set_env_mode("arid")
    for _ in range(10):
        dc.step()
        print(dc.get_telemetry())
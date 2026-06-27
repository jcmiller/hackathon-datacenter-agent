import time
import random
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Component:
    name: str
    temp: float = 40.0
    status: str = 'healthy'

class Server:
    def __init__(self, id: int):
        self.id = id
        self.components = {
            'cpu': Component('CPU'),
            'psu': Component('PSU'),
            'fans': Component('Fans')
        }
        self.load: float = 0.5  # 0-1
        self.voltage: float = 220.0

    def update(self, dt: float = 1.0):
        # Simple physics: heat from load, cooling
        self.components['cpu'].temp += self.load * 5 * dt - 2 * dt
        if self.components['cpu'].temp > 80:
            self.components['cpu'].status = 'overheating'

class DataCenterSimulator:
    def __init__(self):
        self.servers = [Server(i) for i in range(5)]
        self.time = 0
        self.incidents = []

    def step(self):
        self.time += 1
        for s in self.servers:
            s.update()
        # Inject failures occasionally
        if random.random() < 0.1:
            self.inject_failure()

    def inject_failure(self):
        # Example: overheating or voltage
        server = random.choice(self.servers)
        if random.random() < 0.5:
            server.components['cpu'].temp += 30
        else:
            server.voltage -= 20
        self.incidents.append(f'Failure injected at t={self.time}')

    def get_telemetry(self) -> Dict:
        return {
            'time': self.time,
            'servers': [{
                'id': s.id,
                'load': s.load,
                'voltage': s.voltage,
                'cpu_temp': s.components['cpu'].temp,
                'status': s.components['cpu'].status
            } for s in self.servers]
        }

# For Antigravity code exec testing
if __name__ == '__main__':
    dc = DataCenterSimulator()
    for _ in range(5):
        dc.step()
        print(dc.get_telemetry())
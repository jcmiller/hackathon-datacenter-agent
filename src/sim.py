import os
import json
import asyncio
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpusitter-sim")

app = FastAPI(title="GPUSitter Telemetry Simulator & Processor")

# In-memory storage for GPU telemetry states
# 4 nodes, each with 8 GPUs
NODES = ["172.31.5.201", "172.31.5.202", "172.31.5.233", "172.31.0.32"]
GPU_COUNT = 8

class GPUState:
    def __init__(self, node: str, index: int):
        self.node = node
        self.index = index
        self.gpu_id = f"{node}-{index}"
        
        # Telemetry metrics
        self.temp = 32.0
        self.mem_temp = 47.0
        self.util = 0.0
        self.mem_copy = 0.0
        self.fb_used = 3.0
        self.fb_free = 81248.0
        self.power = 64.0
        self.clock = 1593.0
        self.xid = 0.0
        self.status = "healthy"
        
        # Error sequence state machine
        self.active_sequence: Optional[str] = None
        self.sequence_step = 0
        self.workload_cycle = 0

    def update(self):
        # Workload cycle for healthy nodes (sine-wave-like active training phases)
        self.workload_cycle = (self.workload_cycle + 1) % 60
        
        if self.status == "faulty":
            # Persist in error state
            self.util = 0.0
            self.power = 70.0 + (5.0 if self.active_sequence == "ecc_crash" else 0.0)
            self.fb_used = 3.0 if self.active_sequence != "ecc_crash" else 81248.0
            self.fb_free = 81248.0 - self.fb_used
            return

        if self.active_sequence == "ecc_crash":
            self.run_ecc_crash_step()
        elif self.active_sequence == "overheat":
            self.run_overheat_step()
        elif self.active_sequence == "init_fail":
            self.run_init_fail_step()
        else:
            # Healthy normal telemetry (periodic pretraining workload signature)
            if self.workload_cycle < 40:
                # Active training workload
                self.util = 100.0 if self.workload_cycle % 2 == 0 else 98.0
                self.power = 380.0 + (10.0 * (self.workload_cycle % 3 - 1))
                self.fb_used = 81248.0 - 79.0
                self.fb_free = 79.0
                self.temp = min(68.0, self.temp + 0.5)
                self.mem_temp = min(78.0, self.mem_temp + 0.8)
            else:
                # Idle between batches / pipeline bubbles
                self.util = 0.0
                self.power = 65.0 + (5.0 * (self.workload_cycle % 2))
                self.fb_used = 3.0
                self.fb_free = 81248.0
                self.temp = max(31.0, self.temp - 0.8)
                self.mem_temp = max(47.0, self.mem_temp - 1.2)
                
            self.xid = 0.0
            self.status = "healthy"

    def run_ecc_crash_step(self):
        step = self.sequence_step
        self.sequence_step += 1
        
        # Matches the exact timeline of 172.31.5.233-2 leading to Xid 31
        if step < 4:
            # Active heavy training workload
            self.temp = 40.0
            self.mem_temp = 56.0
            self.util = 100.0
            self.fb_used = 81248.0
            self.fb_free = 79.0
            self.power = 388.0
        elif step < 6:
            # Drops back to idle
            self.temp = 38.0
            self.mem_temp = 50.0
            self.util = 0.0
            self.fb_used = 80770.0
            self.fb_free = 481.0
            self.power = 76.2
        elif step < 10:
            # Cooling down
            self.temp = 33.0
            self.mem_temp = 48.0
            self.util = 0.0
            self.fb_used = 80762.0
            self.fb_free = 481.0
            self.power = 74.4
        elif step < 12:
            # Brief utilization spike
            self.temp = 32.0
            self.mem_temp = 48.0
            self.util = 100.0
            self.fb_used = 3.0
            self.fb_free = 81248.0
            self.power = 90.4
        elif step == 12:
            # Xid 31 ECC failure occurs
            self.temp = 32.0
            self.mem_temp = 47.0
            self.util = 0.0
            self.fb_used = 81248.0
            self.fb_free = 0.0
            self.power = 73.5
            self.xid = 31.0
            self.status = "faulty"
            logger.warning(f"GPU {self.gpu_id} registered uncorrectable ECC error (Xid 31)")
            
    def run_overheat_step(self):
        # Fan/cooling failure under workload
        self.sequence_step += 1
        self.util = 100.0
        self.power = 350.0
        self.fb_used = 81248.0
        self.fb_free = 79.0
        
        # Temp increases rapidly
        self.temp += 4.0
        self.mem_temp += 5.5
        
        if self.temp > 80.0:
            # Throttling clock
            self.clock = max(800.0, self.clock - 100.0)
            
        if self.temp >= 95.0:
            # Thermal limit shutdown reset
            self.xid = 43.0
            self.status = "faulty"
            self.util = 0.0
            self.power = 70.0
            logger.warning(f"GPU {self.gpu_id} shut down due to thermal limit (Xid 43)")

    def run_init_fail_step(self):
        step = self.sequence_step
        self.sequence_step += 1
        
        if step < 2:
            # Idle
            self.util = 0.0
            self.power = 65.0
            self.fb_used = 3.0
            self.fb_free = 81248.0
        elif step == 2:
            # Job attempts initialization
            self.util = 100.0
            self.power = 388.0
            self.fb_used = 81248.0
            self.fb_free = 79.0
        else:
            # Immediately crashes with Xid 43 (driver initialization reset failure)
            self.xid = 43.0
            self.status = "faulty"
            self.util = 0.0
            self.power = 70.0
            self.fb_used = 3.0
            self.fb_free = 81248.0
            logger.warning(f"GPU {self.gpu_id} failed initialization (Xid 43)")

# Initialize simulator state
gpus: Dict[str, GPUState] = {}
for node in NODES:
    for idx in range(GPU_COUNT):
        state = GPUState(node, idx)
        gpus[state.gpu_id] = state

# Stream subscriptions
subscribers: List[asyncio.Queue] = []

async def telemetry_loop():
    """Background loop that ticks every 1 second to update GPU telemetry."""
    while True:
        await asyncio.sleep(1.0)
        # Update all states
        data = {}
        for gpu_id, gpu in gpus.items():
            gpu.update()
            data[gpu_id] = {
                "node": gpu.node,
                "index": gpu.index,
                "temp": round(gpu.temp, 1),
                "mem_temp": round(gpu.mem_temp, 1),
                "util": round(gpu.util, 1),
                "mem_copy": round(gpu.mem_copy, 1),
                "fb_used": round(gpu.fb_used, 1),
                "fb_free": round(gpu.fb_free, 1),
                "power": round(gpu.power, 1),
                "clock": round(gpu.clock, 0),
                "xid": gpu.xid,
                "status": gpu.status,
                "active_sequence": gpu.active_sequence
            }
            
        # Broadcast to all SSE subscribers
        message = json.dumps(data)
        for queue in list(subscribers):
            try:
                queue.put_nowait(message)
            except Exception:
                subscribers.remove(queue)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_loop())

class FailureInjection(BaseModel):
    node: str
    gpu_index: int
    error_type: str # ecc_crash, overheat, init_fail, reset

@app.post("/api/inject_failure")
async def inject_failure(injection: FailureInjection):
    gpu_id = f"{injection.node}-{injection.gpu_index}"
    if gpu_id not in gpus:
        return {"success": False, "error": f"GPU {gpu_id} not found."}
        
    gpu = gpus[gpu_id]
    if injection.error_type == "reset":
        gpu.active_sequence = None
        gpu.sequence_step = 0
        gpu.status = "healthy"
        gpu.temp = 32.0
        gpu.mem_temp = 47.0
        gpu.xid = 0.0
        gpu.util = 0.0
        gpu.power = 64.0
        gpu.fb_used = 3.0
        gpu.fb_free = 81248.0
        logger.info(f"Reset GPU {gpu_id} back to healthy operational state.")
    else:
        gpu.active_sequence = injection.error_type
        gpu.sequence_step = 0
        logger.info(f"Injected error sequence '{injection.error_type}' on GPU {gpu_id}.")
        
    return {"success": True, "gpu_id": gpu_id, "error_type": injection.error_type}

@app.get("/telemetry/stream")
async def telemetry_stream(request: Request):
    """Server-Sent Events endpoint streaming real-time GPU telemetry."""
    async def event_generator():
        queue = asyncio.Queue()
        subscribers.append(queue)
        try:
            while True:
                # Disconnect check
                if await request.is_disconnected():
                    break
                data = await queue.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in subscribers:
                subscribers.remove(queue)

# --- GPUSitter Triage / Processing Server logic ---

class IncidentPayload(BaseModel):
    incident_id: str
    node: str
    gpu_index: int
    trigger_xid: float
    trigger_time: str

@app.post("/api/triage")
async def triage_incident(payload: IncidentPayload):
    """
    On-Call Processing Endpoint.
    Analyzes historical and current telemetry around the incident to diagnose the issue
    and recommend early-warning parameters.
    """
    gpu_id = f"{payload.node}-{payload.gpu_index}"
    logger.info(f"Triage request received for incident {payload.incident_id} on {gpu_id}")
    
    # Diagnose based on the current state of the GPU
    gpu = gpus.get(gpu_id)
    if not gpu:
        return {"incident_id": payload.incident_id, "disposition": "REJECT", "reason": "GPU not found"}
        
    disposition = "RESTART"
    rca_summary = "Unknown GPU failure"
    early_warning_rule = "None"
    
    if payload.trigger_xid == 31.0 or gpu.active_sequence == "ecc_crash":
        # Memory failure
        disposition = "REPLACE_GPU"
        rca_summary = "Uncorrectable ECC Memory Failure (Xid 31) registered on the VRAM."
        early_warning_rule = "ALERT if Power > 350W AND FB_Used > 80000MB for > 15s (predicts failure 5.5 min early)."
    elif payload.trigger_xid == 43.0 or gpu.active_sequence == "overheat":
        # Thermal shutdown
        disposition = "PAGE_TECHNICIAN"
        rca_summary = "GPU Overheating Limit Exceeded (Xid 43). GPU reached thermal shutdown threshold."
        early_warning_rule = "ALERT if Temp > 80C OR Mem_Temp > 85C (predicts thermal shutdown)."
    elif gpu.active_sequence == "init_fail":
        # Init failure
        disposition = "DRAIN_NODE"
        rca_summary = "GPU Initialization Failure (Xid 43) occurred immediately upon allocating CUDA context."
        early_warning_rule = "ALERT if Xid registered on CUDA context initialization."

    result = {
        "incident_id": payload.incident_id,
        "node": payload.node,
        "gpu_index": payload.gpu_index,
        "trigger_xid": payload.trigger_xid,
        "rca": rca_summary,
        "early_warning_rule": early_warning_rule,
        "disposition": disposition
    }
    
    logger.info(f"Triage complete for {gpu_id}. Disposition: {disposition}")
    return result

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serves the dashboard file directly."""
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r") as f:
            return f.read()
    return "Dashboard HTML file not found."

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


import os
from pydantic import BaseModel
from inference import IntrusionSystem

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
class State:
    total_packets = 0
    detected_attacks = 0
    accuracy = 98.7
    throughput = 0
    recent_logs = []
    traffic_last_minute = {
        "allow": [0] * 60,
        "block": [0] * 60,
        "throttle": [0] * 60
    }
    last_traffic_update = None


state = State()

ids = IntrusionSystem()
class PacketData(BaseModel):
    features: list
    source_ip: str = "unknown"

@app.post("/api/packet")
async def detect_packet(packet: PacketData):
    # Run detection
    print(packet)
    action, confidence = ids.predict_action(packet.features)
    # Update state
    import datetime
    now = datetime.datetime.now()
    state.total_packets += 1
    is_attack = action != "ALLOW"
    if is_attack:
        state.detected_attacks += 1
    state.throughput = state.total_packets  # Simplified, should be pkts/sec

    # Update traffic_last_minute - track actions separately
    sec = now.second
    if state.last_traffic_update is None:
        state.traffic_last_minute = {
            "allow": [0] * 60,
            "block": [0] * 60,
            "throttle": [0] * 60
        }
        state.last_traffic_update = sec
    # If time jumped forward, zero out skipped seconds for all actions
    if sec != state.last_traffic_update:
        diff = (sec - state.last_traffic_update) % 60
        for i in range(1, diff+1):
            idx = (state.last_traffic_update + i) % 60
            state.traffic_last_minute["allow"][idx] = 0
            state.traffic_last_minute["block"][idx] = 0
            state.traffic_last_minute["throttle"][idx] = 0
        state.last_traffic_update = sec
    
    # Increment the appropriate action counter
    action_lower = action.lower()
    if action_lower in state.traffic_last_minute:
        state.traffic_last_minute[action_lower][sec] += 1

    # Log event
    log = {
        "timestamp": now.strftime("%H:%M:%S"),
        "source_ip": packet.source_ip,
        "attack_type": action,
        "confidence": f"{confidence:.2f}"
    }
    state.recent_logs.insert(0, log)
    state.recent_logs = state.recent_logs[:20]
    return {"action": action, "confidence": confidence}

@app.get("/api/stats")
async def get_stats():
    return {
        "total_packets": state.total_packets,
        "detected_attacks": state.detected_attacks,
        "accuracy": round(state.accuracy, 1),
        "throughput": state.throughput,
        "traffic_last_minute": state.traffic_last_minute,  # Now returns action-specific data
    }

@app.get("/api/logs")
async def get_logs(limit: int = 8):
    return state.recent_logs[:limit]

@app.post("/api/control/start")
async def start():
    return {"status": "ok"}

@app.post("/api/control/stop")
async def stop():
    return {"status": "ok"}

@app.post("/api/control/inject")
async def inject():
    return {"status": "ok"}

# Serve static files
public_dir = os.path.join(os.path.dirname(__file__), "public")
app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(public_dir, "index.html"))

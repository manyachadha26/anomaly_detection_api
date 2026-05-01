# 🚨 Real-Time Anomaly Detection API

> A production-grade REST API that detects anomalies in real-time metric streams using three algorithms running in parallel — Z-Score, Isolation Forest, and EWMA.

Built with FastAPI + Docker. Detects CPU spikes, sustained high load, and sudden drops in under 100ms.

---

## Architecture

```
CPU Simulator (generates data)
        ↓
   POST /ingest  (FastAPI endpoint)
        ↓
   ┌─────────────────────────────┐
   │  Z-Score    │ detects spikes │
   │  Iso Forest │ complex patterns│
   │  EWMA       │ gradual drift  │
   └─────────────────────────────┘
        ↓
   Majority vote → anomaly verdict
        ↓
   GET /anomalies  (retrieve results)
```

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest` | Send a data point, get anomaly verdict |
| GET | `/anomalies` | Get recent anomalies |
| GET | `/stream` | Get full recent stream |
| GET | `/stats` | Summary statistics |
| GET | `/health` | Health check |
| DELETE | `/reset` | Reset detector state |
| GET | `/docs` | Auto-generated Swagger UI |

## Run with Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/anomaly-detection-api.git
cd anomaly-detection-api

# Start API + simulator together
docker-compose up --build
```

Open http://localhost:8000/docs for the interactive API documentation.

## Run locally (without Docker)

```bash
pip install -r requirements.txt

# Terminal 1 — start the API
uvicorn main:app --reload

# Terminal 2 — start the simulator
python simulator.py
```

## Test the API manually

```bash
# Send a normal reading
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"value": 45.2, "metric": "cpu_usage"}'

# Send an anomalous spike
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"value": 98.7, "metric": "cpu_usage"}'

# Get detected anomalies
curl http://localhost:8000/anomalies

# Get stats
curl http://localhost:8000/stats
```

## The three algorithms

**Z-Score** — compares each value to the rolling mean/std. Fast and stateless. Best for sudden spikes.

**Isolation Forest** — ML model that learns what normal looks like, then flags points that are hard to isolate. Best for complex patterns.

**EWMA** — exponentially weighted moving average that adapts to gradual drift. Catches slow creep that Z-Score misses.

**Final verdict** — majority vote across all three. 2/3 = anomaly.

## Stack

| Component | Tool |
|---|---|
| API framework | FastAPI |
| ML algorithm | scikit-learn IsolationForest |
| Containerization | Docker + Docker Compose |
| Data simulation | Custom Python simulator |

---



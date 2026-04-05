"""
FastAPI Application
Exposes the anomaly detector as a REST API.

Endpoints:
  POST /ingest          — send a data point, get anomaly result back
  GET  /anomalies       — get all detected anomalies so far
  GET  /stats           — summary stats about the stream
  GET  /health          — health check
  GET  /docs            — auto-generated Swagger UI (FastAPI built-in)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
from collections import deque
from detector import AnomalyDetector, DataPoint, AnomalyResult


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Real-Time Anomaly Detection API",
    description="""
    Detects anomalies in real-time metric streams using three algorithms:
    - Z-Score (sudden spikes)
    - Isolation Forest (complex patterns)
    - EWMA (gradual drift)
    
    Final verdict is a majority vote across all three.
    """,
    version="1.0.0",
)

# Allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state — in production this would be a database
detector = AnomalyDetector()
all_results: deque = deque(maxlen=1000)    # keep last 1000 results
anomalies: deque = deque(maxlen=500)       # keep last 500 anomalies
total_points = 0
start_time = time.time()


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# Pydantic models define the shape of API input/output
# FastAPI auto-generates docs from these
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    value: float = Field(..., description="Metric value (e.g. CPU %)", example=45.2)
    metric: str = Field(default="cpu_usage", description="Metric name", example="cpu_usage")
    timestamp: Optional[float] = Field(default=None, description="Unix timestamp (auto-filled if not provided)")

    class Config:
        json_schema_extra = {
            "example": {
                "value": 45.2,
                "metric": "cpu_usage",
            }
        }


class IngestResponse(BaseModel):
    timestamp: float
    value: float
    metric: str
    is_anomaly: bool
    confidence: float
    zscore: float
    zscore_anomaly: bool
    isolation_anomaly: bool
    ewma_anomaly: bool
    ewma_value: float
    explanation: str


class StatsResponse(BaseModel):
    uptime_seconds: float
    total_points_processed: int
    total_anomalies_detected: int
    anomaly_rate_pct: float
    points_in_memory: int


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """
    Basic health check — returns 200 if API is running.
    Used by Docker and load balancers to check if container is alive.
    """
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - start_time, 1),
        "total_points_processed": total_points,
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Detection"])
def ingest(request: IngestRequest):
    """
    Ingest a single data point and get an anomaly verdict immediately.

    Send any numeric metric value. The API scores it against all three
    detectors and returns results within milliseconds.

    - **value**: the metric reading (e.g. CPU % between 0-100)
    - **metric**: name of the metric (for labeling)
    - **timestamp**: optional unix timestamp (auto-filled if not provided)
    """
    global total_points

    point = DataPoint(
        timestamp=request.timestamp or time.time(),
        value=request.value,
        metric=request.metric,
    )

    result = detector.process(point)
    total_points += 1

    all_results.append(result)
    if result.is_anomaly:
        anomalies.append(result)

    return IngestResponse(
        timestamp=result.timestamp,
        value=result.value,
        metric=result.metric,
        is_anomaly=result.is_anomaly,
        confidence=result.confidence,
        zscore=result.zscore,
        zscore_anomaly=result.zscore_anomaly,
        isolation_anomaly=result.isolation_anomaly,
        ewma_anomaly=result.ewma_anomaly,
        ewma_value=result.ewma_value,
        explanation=result.explanation,
    )


@app.get("/anomalies", tags=["Detection"])
def get_anomalies(limit: int = 50):
    """
    Returns the most recent detected anomalies.

    - **limit**: max number of anomalies to return (default 50, max 500)
    """
    if limit > 500:
        raise HTTPException(status_code=400, detail="limit cannot exceed 500")

    recent = list(anomalies)[-limit:]
    return {
        "count": len(recent),
        "anomalies": [
            {
                "timestamp": r.timestamp,
                "value": r.value,
                "metric": r.metric,
                "confidence": r.confidence,
                "explanation": r.explanation,
            }
            for r in recent
        ],
    }


@app.get("/stream", tags=["Detection"])
def get_stream(limit: int = 100):
    """
    Returns the most recent data points (both normal and anomalous).
    Useful for visualizing the full stream with anomalies highlighted.
    """
    recent = list(all_results)[-limit:]
    return {
        "count": len(recent),
        "points": [
            {
                "timestamp": r.timestamp,
                "value": r.value,
                "is_anomaly": r.is_anomaly,
                "confidence": r.confidence,
                "ewma_value": r.ewma_value,
            }
            for r in recent
        ],
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
def get_stats():
    """
    Returns summary statistics about the anomaly detection stream.
    """
    anomaly_rate = (len(anomalies) / total_points * 100) if total_points > 0 else 0

    return StatsResponse(
        uptime_seconds=round(time.time() - start_time, 1),
        total_points_processed=total_points,
        total_anomalies_detected=len(anomalies),
        anomaly_rate_pct=round(anomaly_rate, 2),
        points_in_memory=len(all_results),
    )


@app.delete("/reset", tags=["System"])
def reset():
    """
    Resets the detector state and clears all stored data.
    Useful for starting fresh without restarting the container.
    """
    global detector, all_results, anomalies, total_points, start_time
    detector = AnomalyDetector()
    all_results = deque(maxlen=1000)
    anomalies = deque(maxlen=500)
    total_points = 0
    start_time = time.time()
    return {"status": "reset complete"}

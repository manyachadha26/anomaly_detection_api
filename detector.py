"""
Anomaly Detection Engine
Three algorithms running in parallel:
  1. Z-Score         — fast, stateless, catches sudden spikes
  2. Isolation Forest — ML-based, catches complex patterns
  3. EWMA            — adapts to gradual drift over time

Each algorithm returns a score + boolean flag.
Final verdict is a majority vote across all three.
"""

import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class DataPoint:
    """A single incoming metric reading."""
    timestamp: float
    value: float
    metric: str = "cpu_usage"


@dataclass
class AnomalyResult:
    """Result from the anomaly detection engine for one data point."""
    timestamp: float
    value: float
    metric: str
    is_anomaly: bool
    confidence: float          # 0.0 to 1.0 — how confident we are it's an anomaly
    zscore: float
    zscore_anomaly: bool
    isolation_anomaly: bool
    ewma_anomaly: bool
    ewma_value: float          # what the EWMA expected the value to be
    explanation: str           # human-readable reason


# ─────────────────────────────────────────────
# ALGORITHM 1: Z-SCORE
# ─────────────────────────────────────────────

class ZScoreDetector:
    """
    Compares each new value to the rolling mean and standard deviation.
    If the value is more than `threshold` standard deviations from the mean,
    it's an anomaly.

    Good for: sudden spikes (CPU jumps from 30% to 95%)
    Bad for: gradual drift (CPU slowly creeping up over hours)
    """

    def __init__(self, window: int = 50, threshold: float = 3.0):
        self.window = window          # how many recent values to consider
        self.threshold = threshold    # standard deviations cutoff
        self.values = deque(maxlen=window)

    def score(self, value: float) -> tuple[float, bool]:
        """Returns (z_score, is_anomaly)."""
        self.values.append(value)

        if len(self.values) < 10:
            # Not enough data yet — can't compute meaningful stats
            return 0.0, False

        mean = np.mean(self.values)
        std = np.std(self.values)

        if std < 1e-6:
            # All values are identical — no variation to score against
            return 0.0, False

        z = abs((value - mean) / std)
        return round(z, 3), z > self.threshold


# ─────────────────────────────────────────────
# ALGORITHM 2: ISOLATION FOREST
# ─────────────────────────────────────────────

class IsolationForestDetector:
    """
    An ML model that learns what "normal" looks like from recent data,
    then flags points that are hard to isolate (i.e. unusual).

    IsolationForest works by randomly partitioning data — anomalies
    require fewer cuts to isolate because they're far from the cluster.

    Good for: complex multivariate anomalies, non-gaussian distributions
    Bad for: very small datasets (needs ~50+ points to train)
    """

    def __init__(self, window: int = 100, contamination: float = 0.05):
        self.window = window
        self.contamination = contamination   # expected % of anomalies in data
        self.values = deque(maxlen=window)
        self.model = None
        self.retrain_every = 20              # retrain model every N new points
        self.points_since_retrain = 0

    def _retrain(self):
        if len(self.values) < 20:
            return
        X = np.array(self.values).reshape(-1, 1)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=50,    # fewer trees = faster inference
        )
        self.model.fit(X)

    def score(self, value: float) -> bool:
        """Returns is_anomaly."""
        self.values.append(value)
        self.points_since_retrain += 1

        if self.points_since_retrain >= self.retrain_every:
            self._retrain()
            self.points_since_retrain = 0

        if self.model is None:
            return False

        prediction = self.model.predict([[value]])
        return prediction[0] == -1   # IsolationForest: -1 = anomaly, 1 = normal


# ─────────────────────────────────────────────
# ALGORITHM 3: EWMA (Exponentially Weighted Moving Average)
# ─────────────────────────────────────────────

class EWMADetector:
    """
    Maintains a smoothed running average that gives more weight to recent values.
    Flags anomalies when actual value deviates too far from the EWMA prediction.

    Unlike Z-Score which uses a fixed window, EWMA adapts to slow drift —
    if CPU gradually climbs from 30% to 50% over an hour, EWMA follows it.
    A sudden jump to 90% would still be caught.

    alpha: smoothing factor. Higher = adapts faster, lower = smoother.
    Good for: gradual drift + sudden spikes
    """

    def __init__(self, alpha: float = 0.1, threshold_multiplier: float = 2.5):
        self.alpha = alpha
        self.threshold_multiplier = threshold_multiplier
        self.ewma = None
        self.ewma_variance = None
        self.values = deque(maxlen=50)

    def score(self, value: float) -> tuple[float, bool]:
        """Returns (ewma_value, is_anomaly)."""
        if self.ewma is None:
            self.ewma = value
            self.ewma_variance = 0.0
            self.values.append(value)
            return value, False

        # Update EWMA
        prev_ewma = self.ewma
        self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma

        # Update variance estimate
        deviation = abs(value - prev_ewma)
        self.ewma_variance = (
            self.alpha * deviation + (1 - self.alpha) * self.ewma_variance
        )

        self.values.append(value)

        if self.ewma_variance < 1e-6:
            return round(self.ewma, 3), False

        # Anomaly if deviation exceeds threshold * variance
        threshold = self.threshold_multiplier * self.ewma_variance
        is_anomaly = deviation > threshold

        return round(self.ewma, 3), is_anomaly


# ─────────────────────────────────────────────
# COMBINED DETECTOR
# ─────────────────────────────────────────────

class AnomalyDetector:
    """
    Runs all three detectors on each incoming data point.
    Final verdict: majority vote (2 out of 3 = anomaly).
    Confidence = proportion of detectors that agree.
    """

    def __init__(self):
        self.zscore = ZScoreDetector(window=50, threshold=3.0)
        self.isolation = IsolationForestDetector(window=100, contamination=0.05)
        self.ewma = EWMADetector(alpha=0.1, threshold_multiplier=2.5)

    def process(self, point: DataPoint) -> AnomalyResult:
        z_score, z_anomaly = self.zscore.score(point.value)
        iso_anomaly = self.isolation.score(point.value)
        ewma_val, ewma_anomaly = self.ewma.score(point.value)

        # Majority vote
        votes = sum([z_anomaly, iso_anomaly, ewma_anomaly])
        is_anomaly = votes >= 2
        confidence = round(votes / 3, 2)

        # Human-readable explanation
        if not is_anomaly:
            explanation = "Normal reading."
        else:
            reasons = []
            if z_anomaly:
                reasons.append(f"Z-score={z_score} (threshold 3.0)")
            if iso_anomaly:
                reasons.append("Isolation Forest flagged as outlier")
            if ewma_anomaly:
                reasons.append(f"Deviated from EWMA={ewma_val}")
            explanation = " | ".join(reasons)

        return AnomalyResult(
            timestamp=point.timestamp,
            value=point.value,
            metric=point.metric,
            is_anomaly=is_anomaly,
            confidence=confidence,
            zscore=z_score,
            zscore_anomaly=z_anomaly,
            isolation_anomaly=iso_anomaly,
            ewma_anomaly=ewma_anomaly,
            ewma_value=ewma_val,
            explanation=explanation,
        )

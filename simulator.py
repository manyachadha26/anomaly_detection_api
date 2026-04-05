"""
CPU Metrics Simulator
Generates realistic server CPU usage data and sends it to the API.
Injects anomalies at random intervals so you can see the detector in action.

Run this in a separate terminal while the API is running:
  python simulator.py

You'll see live output showing normal readings and flagged anomalies.
"""

import requests
import time
import random
import math
import argparse
from datetime import datetime


# ─────────────────────────────────────────────
# REALISTIC CPU SIMULATION
# ─────────────────────────────────────────────

class CPUSimulator:
    """
    Generates realistic CPU usage patterns:
    - Base load: 20-40% (normal server idle)
    - Daily cycle: slight increase during "business hours"
    - Random noise: ±5% normal variation
    - Injected anomalies: sudden spikes, sustained high load, drops
    """

    def __init__(self):
        self.base_load = 30.0
        self.t = 0
        self.anomaly_countdown = random.randint(30, 60)
        self.in_anomaly = False
        self.anomaly_duration = 0
        self.anomaly_type = None

    def next(self) -> tuple[float, bool]:
        """
        Returns (cpu_value, is_injected_anomaly).
        The boolean tells the simulator whether it intentionally injected one.
        """
        self.t += 1
        self.anomaly_countdown -= 1

        # Normal signal: base load + slow sine wave + noise
        slow_wave = 5 * math.sin(self.t / 20)         # slow oscillation
        noise = random.gauss(0, 2)                     # gaussian noise ±2%
        value = self.base_load + slow_wave + noise

        injected = False

        # Inject anomaly
        if self.anomaly_countdown <= 0 and not self.in_anomaly:
            self.anomaly_type = random.choice(["spike", "sustained", "drop"])
            self.anomaly_duration = random.randint(1, 5)
            self.in_anomaly = True
            self.anomaly_countdown = random.randint(40, 80)

        if self.in_anomaly:
            if self.anomaly_type == "spike":
                value += random.uniform(40, 60)    # sudden spike to 70-100%
            elif self.anomaly_type == "sustained":
                value += random.uniform(25, 40)    # sustained high load
            elif self.anomaly_type == "drop":
                value -= random.uniform(20, 28)    # sudden drop (process crash)

            self.anomaly_duration -= 1
            injected = True
            if self.anomaly_duration <= 0:
                self.in_anomaly = False

        # Clamp to 0-100%
        value = max(0.0, min(100.0, value))
        return round(value, 2), injected


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run_simulator(api_url: str, interval: float, num_points: int):
    sim = CPUSimulator()

    print(f"\n{'='*55}")
    print(f"  CPU Anomaly Simulator")
    print(f"  API: {api_url}")
    print(f"  Sending 1 point every {interval}s")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*55}\n")
    print(f"{'Time':<12} {'CPU %':<10} {'Injected':<12} {'Detected':<10} {'Confidence':<12} {'Explanation'}")
    print("-" * 90)

    sent = 0
    detected = 0

    try:
        while sent < num_points:
            value, injected = sim.next()

            try:
                response = requests.post(
                    f"{api_url}/ingest",
                    json={"value": value, "metric": "cpu_usage"},
                    timeout=5,
                )
                result = response.json()

                is_detected = result.get("is_anomaly", False)
                confidence = result.get("confidence", 0)
                explanation = result.get("explanation", "")[:60]

                if is_detected:
                    detected += 1

                # Color coding in terminal
                now = datetime.now().strftime("%H:%M:%S")
                injected_str = "⚠ INJECTED" if injected else "normal"
                detected_str = "🚨 ANOMALY" if is_detected else "ok"

                print(
                    f"{now:<12} {value:<10} {injected_str:<12} "
                    f"{detected_str:<10} {confidence:<12} {explanation}"
                )

            except requests.exceptions.ConnectionError:
                print("❌ Cannot connect to API. Is it running? (uvicorn main:app --reload)")
                time.sleep(3)
                continue

            sent += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        pass

    print(f"\n{'='*55}")
    print(f"  Sent: {sent} points")
    print(f"  Anomalies detected: {detected} ({round(detected/max(sent,1)*100, 1)}%)")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU metrics simulator")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between points")
    parser.add_argument("--points", type=int, default=500, help="Total points to send")
    args = parser.parse_args()

    run_simulator(args.url, args.interval, args.points)

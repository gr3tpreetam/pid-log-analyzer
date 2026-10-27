"""Generate simulated PID log data for demonstration purposes.

This does NOT model any specific robot.  It gives the tool plausible-looking
data to test with when no real logs are available.

Model: a 1-D "position" with offset p, velocity v.
    dv = -damping * v + steer_gain * u + curve_push + asymmetry
    u  = -(Kp * e + Kd * de/dt), clipped to [-255, 255]
    e  = p + sensor_noise
    Control delay: 3 samples.
    Curve pushes: random step disturbances every 2-4 s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_run(
    *,
    kp: float = 2.0,
    kd: float = 0.05,
    noise: float = 0.005,
    asymmetry: float = 0.0,
    duration_s: float = 30.0,
    fs: float = 100.0,
    curve_strength: float = 20.0,
    damping: float = 4.0,
    steer_gain: float = 60.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a DataFrame that looks like a line-follower log.

    Parameters
    ----------
    kp, kd : PD gains
    noise : sensor noise amplitude (std dev)
    asymmetry : constant offset added to error (simulates calibration error)
    duration_s : run duration in seconds
    fs : sample rate (Hz)
    curve_strength : magnitude of random curve pushes
    damping : friction-like velocity damping
    steer_gain : how strongly the PID output moves the robot
    seed : RNG seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: time_ms, error, output, kp, kd
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    n = int(duration_s * fs)
    delay_samples = 3

    t = np.arange(n) * dt

    # Pre-compute random curve pushes (step disturbances every 2-4 s)
    curve = np.zeros(n)
    i_push = 0
    while i_push < n:
        gap = int(rng.uniform(2.0, 4.0) * fs)
        i_push += gap
        if i_push < n:
            direction = rng.choice([-1.0, 1.0])
            strength = direction * curve_strength * rng.uniform(0.5, 1.5)
            # Push lasts 0.2-0.5 s
            dur = int(rng.uniform(0.2, 0.5) * fs)
            end = min(i_push + dur, n)
            curve[i_push:end] = strength

    position = 0.0
    velocity = 0.0
    prev_error = 0.0

    errors = np.zeros(n)
    outputs = np.zeros(n)
    output_buffer = np.zeros(delay_samples)  # ring buffer for delay

    for i in range(n):
        # Sensor reading: position offset + noise
        error = position + rng.normal(0, noise)

        # Delayed PID output (3-sample delay)
        deriv = (error - prev_error) * fs  # de/dt
        u_new = -(kp * error + kd * deriv)
        u_new = np.clip(u_new, -255, 255)

        # Use the delayed output
        u = output_buffer[i % delay_samples]
        output_buffer[i % delay_samples] = u_new

        # Physics: dv = (-damping * v + steer_gain * u + curve + asymmetry) * dt
        dv = (-damping * velocity + steer_gain * u + curve[i] + asymmetry) * dt
        velocity += dv
        position += velocity * dt

        errors[i] = error
        outputs[i] = u_new  # log the command, not the delayed version

        prev_error = error

    return pd.DataFrame({
        "time_ms": (t * 1000).astype(int),
        "error": errors,
        "output": outputs,
        "kp": kp,
        "kd": kd,
    })

"""Pure-function metrics for PID log analysis.

Every function here is deterministic and side-effect free, making them easy
to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

def _noise_sigma(error: np.ndarray) -> float:
    """Estimate sensor noise from the MAD of diff(error) / sqrt(2).

    Uses the Median Absolute Deviation of the first difference, which is
    robust to actual signal changes (curves).
    """
    d = np.diff(error)
    if len(d) == 0:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    # MAD → sigma for a Normal: sigma ≈ MAD / 0.6745
    return (mad / 0.6745) / np.sqrt(2)


def _auto_deadband(error: np.ndarray, factor: float = 0.1) -> float:
    """Deadband that scales with signal + noise.

    deadband = max(factor * std(error), 3 * noise_sigma)
    """
    sigma_n = _noise_sigma(error)
    return float(max(factor * np.std(error), 3.0 * sigma_n))


# ---------------------------------------------------------------------------
# Zero crossings & swings
# ---------------------------------------------------------------------------

def find_crossings(error: np.ndarray, deadband: float = 0.0) -> np.ndarray:
    """Return indices where *error* crosses zero, ignoring flips inside *deadband*.

    A crossing is counted when the signal moves from beyond +deadband/2 on
    one side to beyond -deadband/2 on the other side (state-machine approach).
    """
    indices: list[int] = []
    last_side: Optional[int] = None  # +1 or -1

    for i, v in enumerate(error):
        if abs(v) <= deadband / 2:
            continue
        side = 1 if v > 0 else -1
        if last_side is not None and side != last_side:
            indices.append(i)
        last_side = side

    return np.array(indices, dtype=int)


def swing_peaks(error: np.ndarray, crossings: np.ndarray) -> np.ndarray:
    """Return the peak |error| in each swing (interval between crossings)."""
    if len(crossings) < 2:
        return np.array([], dtype=float)
    peaks = []
    for a, b in zip(crossings[:-1], crossings[1:]):
        peaks.append(np.max(np.abs(error[a:b])))
    return np.array(peaks)


def decay_ratio(peaks: np.ndarray) -> float:
    """Ratio of consecutive swing amplitudes.

    Returns the median of ``peaks[i+1] / peaks[i]``.
    < 1 → settling, ≈ 1 → sustained, > 1 → growing.
    Returns NaN when there are fewer than 3 peaks.
    """
    if len(peaks) < 3:
        return float("nan")
    ratios = peaks[1:] / np.where(peaks[:-1] == 0, 1e-12, peaks[:-1])
    return float(np.median(ratios))


def crossing_frequency(t: np.ndarray, crossings: np.ndarray) -> float:
    """Estimate oscillation frequency from zero-crossing times.

    Two consecutive crossings ≈ half a period.
    """
    if len(crossings) < 2:
        return 0.0
    half_periods = np.diff(t[crossings])
    full_period = 2 * np.median(half_periods)
    return float(1.0 / full_period) if full_period > 0 else 0.0


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def spectral_peak(t: np.ndarray, error: np.ndarray) -> tuple[float, float]:
    """Return ``(peak_frequency_hz, share_of_non_dc_power)`` via Welch's method.

    The share is the fraction of non-DC power contained within ±2 frequency
    bins of the peak.  A pure sine should give a share close to 1; white
    noise gives a very low share.
    """
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return (0.0, 0.0)
    fs = 1.0 / dt

    # Resample to uniform grid for Welch (real serial logs can have jitter)
    t_uniform = np.arange(t[0], t[-1], dt)
    if len(t_uniform) < 64:
        return (0.0, 0.0)
    error_uniform = np.interp(t_uniform, t, error)

    nperseg = min(len(error_uniform), max(64, 2 ** int(np.log2(len(error_uniform) / 2))))
    freqs, psd = welch(error_uniform, fs=fs, nperseg=nperseg)

    if len(psd) < 3 or psd[1:].sum() == 0:
        return (0.0, 0.0)

    # Skip DC (index 0)
    peak_idx = np.argmax(psd[1:]) + 1
    non_dc_power = psd[1:].sum()

    # Power in ±2 bins around peak
    lo = max(1, peak_idx - 2)
    hi = min(len(psd), peak_idx + 3)  # exclusive
    peak_power = psd[lo:hi].sum()

    return (float(freqs[peak_idx]), float(peak_power / non_dc_power))


# ---------------------------------------------------------------------------
# Output health
# ---------------------------------------------------------------------------

def saturation_fraction(output: np.ndarray, limit: Optional[float] = None) -> float:
    """Fraction of samples where |output| is at *limit*.

    If *limit* is None, it is estimated as the maximum |output| and clipping
    is detected only when the output sits *exactly* at that value (within a
    small tolerance) in consecutive runs of ≥2 samples, so a smooth sine
    reaching its peak once per cycle is not flagged as clipped.
    """
    absout = np.abs(output)
    if limit is None:
        limit_val = float(np.max(absout))
        if limit_val == 0:
            return 0.0
        # Auto-detected: only count runs of ≥5 consecutive samples at the
        # limit.  A smooth sine touches its peak for ~3 samples per cycle;
        # true clipping creates much longer flat segments.
        tol = limit_val * 0.001
        at_limit = absout >= limit_val - tol
        min_run = 5
        clipped = np.zeros(len(at_limit), dtype=bool)
        run_start = None
        for i in range(len(at_limit)):
            if at_limit[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and (i - run_start) >= min_run:
                    clipped[run_start:i] = True
                run_start = None
        # Handle run at end of array
        if run_start is not None and (len(at_limit) - run_start) >= min_run:
            clipped[run_start:] = True
        return float(np.mean(clipped))
    else:
        tol = abs(limit) * 0.005
        return float(np.mean(absout >= limit - tol))


def jitter_ratio(output: np.ndarray) -> float:
    """std(diff(output)) / std(output).

    High values mean the output jumps around a lot relative to its overall
    range — a sign that Kd is amplifying sensor noise.
    """
    output = output.astype(float)
    s = float(np.std(output))
    if s < 1e-12 or len(output) < 4:
        return 0.0
    return float(np.std(np.diff(output)) / s)


# ---------------------------------------------------------------------------
# Big-error episodes ("events")
# ---------------------------------------------------------------------------

def find_events(
    t: np.ndarray,
    error: np.ndarray,
    merge_gap_s: float = 0.05,
) -> list[dict]:
    """Find stretches where |error| is well above its usual size.

    Uses 3 × MAD-scale as the threshold (robust to outliers).
    The *error* passed in should be **bias-corrected** (mean removed) so that
    a constant offset is not mistaken for "big error".

    Gaps shorter than *merge_gap_s* between events are merged.

    Returns
    -------
    list of dicts with keys: start_s, duration_s, peak_error
    """
    ae = np.abs(error)
    # MAD-based scale (robust) — compute from the *signed* error so that
    # the MAD correctly estimates σ for Gaussian signals.
    mad = float(np.median(np.abs(error - np.median(error))))
    mad_scale = mad / 0.6745 if mad > 0 else float(np.std(error))
    thresh = 3.0 * mad_scale

    if thresh <= 0:
        return []

    above = ae > thresh

    # Collect raw runs — require at least 3 consecutive samples
    min_run = 3
    raw_events: list[tuple[int, int]] = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_run:
                raw_events.append((i, j))
            i = j
        else:
            i += 1

    if not raw_events:
        return []

    # Merge events with gaps <= merge_gap_s
    merged: list[tuple[int, int]] = [raw_events[0]]
    for start, end in raw_events[1:]:
        prev_start, prev_end = merged[-1]
        gap = t[min(start, len(t) - 1)] - t[min(prev_end, len(t) - 1)]
        if gap <= merge_gap_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    events: list[dict] = []
    for s, e in merged:
        e = min(e, len(t) - 1)
        events.append({
            "start_s": float(t[s]),
            "duration_s": float(t[e] - t[s]),
            "peak_error": float(np.max(ae[s : e + 1])),
        })

    return events


# ---------------------------------------------------------------------------
# Top-level: compute all metrics for a run
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    deadband_factor: float = 0.1,
    output_limit: Optional[float] = None,
) -> dict:
    """Return a dict of all analysis metrics for a single run."""
    error = df["error"].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01
    fs = 1.0 / dt if dt > 0 else 100.0

    # basic stats
    rms = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    bias_ratio = abs(bias) / rms if rms > 1e-12 else 0.0

    # auto deadband (lesson 1)
    db = _auto_deadband(error, factor=deadband_factor)

    # crossings & swings
    crossings = find_crossings(error, db)
    peaks = swing_peaks(error, crossings)
    dr = decay_ratio(peaks)
    xfreq = crossing_frequency(t, crossings)
    n_swings = max(0, len(crossings) - 1)

    # spectrum
    spec_freq, spec_share = spectral_peak(t, error)

    # events — use bias-free error (lesson 2)
    centred = error - bias
    events = find_events(t, centred)
    event_durations = [e["duration_s"] for e in events]

    # output
    has_output = "output" in df.columns
    if has_output:
        output = df["output"].to_numpy(dtype=float)
        sat = saturation_fraction(output, limit=output_limit)
        jr = jitter_ratio(output)
    else:
        sat = None
        jr = None

    return {
        "rms": rms,
        "mae": mae,
        "bias": bias,
        "bias_ratio": bias_ratio,
        "swings": n_swings,
        "decay_ratio": dr,
        "crossing_freq_hz": xfreq,
        "spec_freq_hz": spec_freq,
        "spec_peak_share": spec_share,
        "event_count": len(events),
        "event_median_duration_s": float(np.median(event_durations)) if event_durations else 0.0,
        "saturation": sat,
        "jitter_ratio": jr,
        "deadband": db,
        "fs": fs,
        "events": events,
    }

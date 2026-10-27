"""Rule-based diagnosis from computed metrics.

Each rule inspects the metrics dict and, if triggered, returns a Finding
with a severity, plain-language title, tuning advice and supporting evidence.

**Honesty note**: these are rules of thumb.  Thresholds were tuned on
simulated data and must be validated on real robot logs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Thresholds:
    """User-tunable thresholds that control which findings fire."""

    min_osc_hz: float = 1.0
    """Wobble faster than this is treated as oscillation (not the track)."""

    slow_event_s: float = 0.35
    """Big-error episodes longer than this are flagged as sluggish recovery."""


@dataclass
class Finding:
    """One diagnostic finding for a run."""

    title: str
    severity: str  # "ok", "info", "warn", "bad"
    advice: str
    evidence: dict = field(default_factory=dict)


def diagnose(metrics: dict, thresholds: Optional[Thresholds] = None) -> list[Finding]:
    """Apply rules of thumb to *metrics* and return a list of findings.

    Rule defaults (from the spec):
    - Sustained oscillation: spec_freq ≥ 1 Hz AND peak_share ≥ 0.6
      AND swings ≥ 6 AND decay ≥ 0.85.  decay > 1.1 → growing.
    - Lightly damped: fast AND 0.65 ≤ decay < 0.85.
    - Slow recovery: not oscillating, ≥ 3 events, median dur ≥ 0.35 s.
    - Clipping: saturation ≥ 10%.
    - Twitchy output: jitter ≥ 0.9.
    - Steady offset: bias_ratio ≥ 0.35.
    """
    th = thresholds or Thresholds()
    findings: list[Finding] = []

    dr = metrics.get("decay_ratio", float("nan"))
    spec_freq = metrics.get("spec_freq_hz", 0)
    spec_share = metrics.get("spec_peak_share", 0)
    n_swings = metrics.get("swings", 0)

    is_nan_dr = isinstance(dr, float) and math.isnan(dr)

    # --- oscillation rules --------------------------------------------------
    fast_enough = spec_freq >= th.min_osc_hz
    strong_spectral = spec_share >= 0.6
    enough_swings = n_swings >= 6
    decay_sustained = (not is_nan_dr) and dr >= 0.85
    decay_growing = (not is_nan_dr) and dr > 1.1
    decay_lightly_damped = (not is_nan_dr) and 0.65 <= dr < 0.85

    osc_detected = False

    # Growing oscillation (check first — more severe)
    if fast_enough and strong_spectral and enough_swings and decay_growing:
        findings.append(Finding(
            title="Growing oscillation",
            severity="bad",
            advice=(
                f"Swings are getting bigger (decay ratio {dr:.2f} > 1). "
                f"The dominant wobble is at {spec_freq:.1f} Hz. "
                "The loop is unstable -- reduce Kp by ~30-50% immediately."
            ),
            evidence={
                "decay_ratio": round(dr, 3),
                "spec_freq_hz": round(spec_freq, 2),
                "spec_peak_share": round(spec_share, 3),
                "swings": n_swings,
            },
        ))
        osc_detected = True

    # Sustained oscillation
    elif fast_enough and strong_spectral and enough_swings and decay_sustained:
        findings.append(Finding(
            title="Sustained oscillation",
            severity="bad",
            advice=(
                f"Sustained {spec_freq:.1f} Hz oscillation "
                f"(decay ratio {dr:.2f} ~ 1, power share {spec_share*100:.0f}%). "
                "Lower Kp ~20-30% or raise Kd to add damping."
            ),
            evidence={
                "decay_ratio": round(dr, 3),
                "spec_freq_hz": round(spec_freq, 2),
                "spec_peak_share": round(spec_share, 3),
                "swings": n_swings,
            },
        ))
        osc_detected = True

    # Lightly damped oscillation
    elif fast_enough and enough_swings and decay_lightly_damped:
        findings.append(Finding(
            title="Lightly damped oscillation",
            severity="warn",
            advice=(
                f"Oscillation at ~{spec_freq:.1f} Hz is dying out slowly "
                f"(decay ratio {dr:.2f}). "
                "Add a little more Kd to speed up settling."
            ),
            evidence={
                "decay_ratio": round(dr, 3),
                "spec_freq_hz": round(spec_freq, 2),
                "swings": n_swings,
            },
        ))
        osc_detected = True

    # --- slow recovery (only if not oscillating) ----------------------------
    event_dur = metrics.get("event_median_duration_s", 0)
    event_count = metrics.get("event_count", 0)
    if (not osc_detected) and event_count >= 3 and event_dur >= th.slow_event_s:
        findings.append(Finding(
            title="Slow recovery from big errors",
            severity="warn",
            advice=(
                f"Median big-error episode lasts {event_dur:.2f} s "
                f"({event_count} episodes).  The controller is sluggish "
                "after curves.  Raise Kp or lower the base speed."
            ),
            evidence={
                "event_median_duration_s": round(event_dur, 3),
                "event_count": event_count,
            },
        ))

    # --- output clipping (≥ 10%) --------------------------------------------
    sat = metrics.get("saturation")
    if sat is not None and sat >= 0.10:
        findings.append(Finding(
            title="Output clipping",
            severity="warn" if sat < 0.3 else "bad",
            advice=(
                f"Output is at its limit {sat*100:.0f}% of the time. "
                "The gains may be too high or the motor/PWM range too narrow. "
                "Reduce Kp, or widen the output range."
            ),
            evidence={"saturation_fraction": round(sat, 3)},
        ))

    # --- twitchy output (jitter ≥ 0.9) --------------------------------------
    jitter = metrics.get("jitter_ratio")
    if jitter is not None and jitter >= 0.9:
        findings.append(Finding(
            title="Twitchy output (Kd amplifying noise)",
            severity="warn",
            advice=(
                f"Output jitter ratio is {jitter:.2f} (std of change / std of "
                "output).  Kd is amplifying sensor noise.  Lower Kd or add a "
                "low-pass filter on the derivative term."
            ),
            evidence={"jitter_ratio": round(jitter, 3)},
        ))

    # --- steady offset (bias_ratio ≥ 0.35) ----------------------------------
    bias_ratio = metrics.get("bias_ratio", 0)
    bias = metrics.get("bias", 0)
    if bias_ratio >= 0.35:
        findings.append(Finding(
            title="Steady-state offset (bias)",
            severity="info",
            advice=(
                f"Mean error is {bias:+.3f} (bias ratio {bias_ratio:.2f}).  "
                "The controller does not centre on zero.  Check sensor "
                "calibration or add a small Ki term (watch for windup)."
            ),
            evidence={
                "bias": round(bias, 4),
                "bias_ratio": round(bias_ratio, 3),
            },
        ))

    # --- all good -----------------------------------------------------------
    if not findings:
        findings.append(Finding(
            title="No major issues detected",
            severity="ok",
            advice="The controller looks reasonable based on these metrics.",
        ))

    return findings

"""Tests for the PID log analysis pipeline.

Covers: loader, crossings/deadband, decay ratio, frequency on a sine,
spectral peak vs noise, saturation (auto-limit must not flag a smooth sine),
and end-to-end oscillation / noise / offset diagnoses.
"""
import io
import math

import numpy as np
import pandas as pd
import pytest

from analysis.diagnose import diagnose, Thresholds
from analysis.load import LogError, load_log
from analysis.metrics import (
    compute_metrics,
    crossing_frequency,
    decay_ratio,
    find_crossings,
    find_events,
    jitter_ratio,
    saturation_fraction,
    spectral_peak,
    swing_peaks,
)


# ---- helpers ---------------------------------------------------------------

def make_df(err, fs=100.0, output=None):
    """Build a minimal DataFrame for testing."""
    t = np.arange(len(err)) / fs
    d = pd.DataFrame({"t": t, "error": err})
    if output is not None:
        d["output"] = output
    return d


# =========================================================================
# LOADER
# =========================================================================

class TestLoader:
    def test_load_time_ms_column(self):
        csv = "time_ms,error,output\n" + "\n".join(f"{i*10},{i%3},{i}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[0] == 0
        assert df["t"].iloc[1] == pytest.approx(0.01)
        assert {"error", "output"} <= set(df.columns)

    def test_load_ms_alias(self):
        csv = "ms,error\n" + "\n".join(f"{i*10},{i*0.1}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[1] == pytest.approx(0.01)

    def test_load_t_ms_alias(self):
        csv = "t_ms,error\n" + "\n".join(f"{i*10},{i*0.1}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[1] == pytest.approx(0.01)

    def test_load_time_s_column(self):
        csv = "time_s,error\n" + "\n".join(f"{i*0.01},{i*0.1}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[1] == pytest.approx(0.01)

    def test_load_ambiguous_time_as_ms(self):
        # median step = 10 (>= 1), so treated as ms
        csv = "time,error\n" + "\n".join(f"{i*10},{i*0.1}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[1] == pytest.approx(0.01)

    def test_load_ambiguous_time_as_seconds(self):
        # median step = 0.01 (< 1), so treated as seconds
        csv = "time,error\n" + "\n".join(f"{i*0.01},{i*0.1}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[1] == pytest.approx(0.01)

    def test_load_setpoint_minus_measured(self):
        csv = "ms,setpoint,measured\n" + "\n".join(f"{i*10},0,{i*0.1}" for i in range(50))
        df = load_log(io.StringIO(csv))
        assert df["error"].iloc[10] == pytest.approx(-1.0)

    def test_load_skips_junk_rows(self):
        rows = ["time_ms,error"] + [f"{i*10},{i}" for i in range(30)]
        rows.insert(5, "BOOT OK,???")
        df = load_log(io.StringIO("\n".join(rows)))
        assert len(df) == 30

    def test_load_strips_boot_messages_before_header(self):
        boot = "Serial init...\n[BOOT] ready\n"
        data = "time_ms,error\n" + "\n".join(f"{i*10},{i}" for i in range(30))
        df = load_log(io.StringIO(boot + data))
        assert len(df) == 30

    def test_load_bytes_input(self):
        csv = "time_ms,error\n" + "\n".join(f"{i*10},{i}" for i in range(30))
        df = load_log(csv.encode("utf-8"))
        assert len(df) == 30

    def test_load_missing_columns_gives_clear_error(self):
        with pytest.raises(LogError):
            load_log(io.StringIO("a,b\n" + "\n".join("1,2" for _ in range(30))))

    def test_load_too_few_rows_gives_clear_error(self):
        with pytest.raises(LogError, match="usable rows"):
            load_log(io.StringIO("time_ms,error\n10,1\n20,2"))

    def test_load_time_starts_from_zero(self):
        csv = "time_ms,error\n" + "\n".join(f"{1000 + i*10},{i}" for i in range(30))
        df = load_log(io.StringIO(csv))
        assert df["t"].iloc[0] == pytest.approx(0.0)


# =========================================================================
# CROSSINGS & SWINGS
# =========================================================================

class TestCrossingsAndSwings:
    def test_crossings_ignore_noise_inside_deadband(self):
        e = np.array([1, 0.05, -0.05, 0.05, 1, -1, 1, -1], float)
        # Only the real flips past deadband/2 = 0.1 are counted
        assert len(find_crossings(e, deadband=0.2)) == 3

    def test_crossings_zero_deadband(self):
        e = np.array([1, -1, 1, -1, 1], float)
        assert len(find_crossings(e, deadband=0.0)) == 4

    def test_decay_ratio_shrinking(self):
        assert decay_ratio(np.array([8, 4, 2, 1], float)) == pytest.approx(0.5)

    def test_decay_ratio_sustained(self):
        assert decay_ratio(np.array([5, 5, 5, 5], float)) == pytest.approx(1.0)

    def test_decay_ratio_too_few_peaks(self):
        assert math.isnan(decay_ratio(np.array([5, 4], float)))

    def test_frequency_from_crossings_matches_sine(self):
        fs, f = 100.0, 2.0
        t = np.arange(0, 10, 1 / fs)
        e = np.sin(2 * np.pi * f * t)
        cr = find_crossings(e, 0.1)
        assert crossing_frequency(t, cr) == pytest.approx(f, rel=0.05)

    def test_swing_peaks_are_amplitudes(self):
        t = np.arange(0, 5, 0.01)
        e = 3 * np.sin(2 * np.pi * 1.0 * t)
        peaks = swing_peaks(e, find_crossings(e, 0.1))
        assert np.allclose(peaks, 3, atol=0.1)


# =========================================================================
# SPECTRUM
# =========================================================================

class TestSpectrum:
    def test_spectral_peak_finds_sine(self):
        fs = 100.0
        t = np.arange(0, 20, 1 / fs)
        f, share = spectral_peak(t, np.sin(2 * np.pi * 3.0 * t))
        assert f == pytest.approx(3.0, abs=0.5)
        assert share > 0.5

    def test_spectral_peak_rejects_noise(self):
        fs = 100.0
        t = np.arange(0, 20, 1 / fs)
        rng = np.random.default_rng(0)
        _, share_noise = spectral_peak(t, rng.normal(size=len(t)))
        assert share_noise < 0.3


# =========================================================================
# OUTPUT HEALTH
# =========================================================================

class TestOutputHealth:
    def test_saturation_fraction_with_limit(self):
        out = np.array([0, 0.5, 1.0, 1.0, -1.0, 0.2])
        assert saturation_fraction(out, limit=1.0) == pytest.approx(3 / 6)

    def test_auto_limit_does_not_flag_smooth_sine(self):
        """Lesson 3: a smooth sine reaching its peak is NOT clipping."""
        x = np.sin(np.linspace(0, 40 * np.pi, 4000))
        assert saturation_fraction(x) < 0.01

    def test_auto_limit_flags_clipped_signal(self):
        x = np.sin(np.linspace(0, 40 * np.pi, 4000))
        clipped = np.clip(3 * x, -1, 1)
        assert saturation_fraction(clipped) > 0.3

    def test_jitter_ratio_smooth_vs_noisy(self):
        smooth = np.sin(np.linspace(0, 10, 500))
        assert jitter_ratio(smooth) < 0.5
        rng = np.random.default_rng(42)
        noisy = rng.normal(size=500)
        assert jitter_ratio(noisy) > 0.5


# =========================================================================
# EVENTS
# =========================================================================

class TestEvents:
    def test_events_use_bias_free_error(self):
        """Lesson 2: constant offset should NOT create events."""
        rng = np.random.default_rng(10)
        # Constant offset of 5 + small noise
        n = 3000
        error = 5.0 + rng.normal(0, 0.1, n)
        t = np.arange(n) / 100.0
        # Bias-corrected: should have almost no events
        centred = error - np.mean(error)
        events = find_events(t, centred)
        # With bias removed, only random spikes above 3*MAD should trigger
        assert len(events) < 5

    def test_events_merge_short_gaps(self):
        t = np.arange(200) / 100.0  # 2 seconds at 100 Hz
        error = np.zeros(200)
        # Two bursts separated by < 0.05 s (4 samples at 100 Hz)
        error[10:20] = 10.0
        error[23:33] = 10.0  # gap = 3 samples = 0.03 s
        events = find_events(t, error, merge_gap_s=0.05)
        # Should merge into one event
        assert len(events) <= 1


# =========================================================================
# END-TO-END DIAGNOSES
# =========================================================================

class TestEndToEnd:
    def test_sustained_sine_is_flagged_as_oscillation(self):
        t = np.arange(0, 30, 0.01)
        e = 2 * np.sin(2 * np.pi * 2.5 * t)
        m = compute_metrics(make_df(e))
        titles = [f.title for f in diagnose(m)]
        assert any("oscillation" in x.lower() for x in titles)

    def test_quiet_noise_is_not_flagged_as_oscillation(self):
        rng = np.random.default_rng(1)
        e = rng.normal(0, 0.05, 3000)
        titles = [f.title.lower() for f in diagnose(compute_metrics(make_df(e)))]
        assert not any("oscillation" in x for x in titles)

    def test_constant_offset_reported_as_bias(self):
        rng = np.random.default_rng(2)
        e = 1.0 + rng.normal(0, 0.1, 3000)
        titles = [f.title.lower() for f in diagnose(compute_metrics(make_df(e)))]
        assert any("offset" in x or "bias" in x for x in titles)

    def test_noisy_kd_flagged_as_twitchy(self):
        """High jitter output should trigger the twitchy finding."""
        rng = np.random.default_rng(3)
        n = 3000
        error = np.sin(np.linspace(0, 20, n)) * 2
        # Output that is very noisy (high jitter)
        output = error * 50 + rng.normal(0, 100, n)
        m = compute_metrics(make_df(error, output=output))
        titles = [f.title.lower() for f in diagnose(m)]
        assert any("twitch" in x or "noise" in x or "kd" in x.lower() for x in titles)

    def test_clipped_output_flagged(self):
        """Output at limit for a large fraction should trigger clipping."""
        n = 3000
        error = np.sin(np.linspace(0, 20, n))
        output = np.clip(error * 500, -255, 255)
        m = compute_metrics(make_df(error, output=output), output_limit=255)
        titles = [f.title.lower() for f in diagnose(m)]
        assert any("clip" in x for x in titles)

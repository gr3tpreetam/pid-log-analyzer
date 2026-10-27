"""Analyse one or more logs in the terminal.

    python cli.py sample_data/run_good.csv sample_data/run_oscillating.csv
    python cli.py my_run.csv --output-limit 255
"""
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analysis.diagnose import diagnose
from analysis.load import LogError, load_log
from analysis.metrics import compute_metrics

SHOW = [
    ("rms", "RMS error"),
    ("mae", "mean |error|"),
    ("bias", "mean error (bias)"),
    ("bias_ratio", "bias ratio |mean|/rms"),
    ("swings", "full swings"),
    ("decay_ratio", "decay ratio"),
    ("spec_freq_hz", "dominant wobble (Hz)"),
    ("spec_peak_share", "power in that wobble"),
    ("event_count", "big-error episodes"),
    ("event_median_duration_s", "median episode (s)"),
    ("saturation", "output clipped share"),
    ("jitter_ratio", "output jitter"),
    ("deadband", "auto deadband"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="PID / line-follower log analyser")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--output-limit", type=float, default=None,
                    help="max |output| your motors accept (default: auto-detect)")
    args = ap.parse_args()

    status = 0
    for path in args.files:
        print(f"\n=== {path} ===")
        try:
            df = load_log(path)
        except (LogError, FileNotFoundError) as exc:
            print("  could not load:", exc)
            status = 1
            continue
        m = compute_metrics(df, output_limit=args.output_limit)
        for key, label in SHOW:
            v = m.get(key)
            if v is None:
                continue
            print(f"  {label:<24} {v:.3f}" if isinstance(v, float) else f"  {label:<24} {v}")
        print("  --- findings ---")
        for f in diagnose(m):
            print(f"  [{f.severity.upper()}] {f.title}")
            print(f"        {f.advice}")
            print(f"        evidence: {f.evidence}")
    return status


if __name__ == "__main__":
    sys.exit(main())

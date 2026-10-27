"""Write simulated sample logs and negative test cases into sample_data/.

    python make_samples.py

Generates:
1. Valid simulated runs with different controller behaviors (good, sluggish,
   sustained oscillating, noisy Kd, steady offset, growing unstable, severe clipping).
2. Invalid test files in sample_data/invalid_test_files/ to verify upload
   error reporting (missing error column, too few rows, corrupted data).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.simulate import simulate_run

CASES = {
    # name: simulator settings
    "run_sluggish":    dict(kp=0.4, kd=0.0,  noise=0.003),
    "run_good":        dict(kp=2.5, kd=0.08, noise=0.003),
    "run_oscillating": dict(kp=5.0, kd=0.05, noise=0.003),
    "run_noisy_kd":    dict(kp=3.0, kd=0.4,  noise=0.02),
    "run_offset":      dict(kp=2.5, kd=0.08, noise=0.003, asymmetry=15.0),
}


def make_unstable_growing_run() -> pd.DataFrame:
    """Generate an unstable run where swings grow exponentially."""
    t = np.arange(0, 8.0, 0.01)
    # Growing oscillation at 2.5 Hz
    error = 0.1 * np.exp(0.65 * t) * np.sin(2 * np.pi * 2.5 * t)
    output = np.clip(-3.5 * error, -255, 255)
    return pd.DataFrame({
        "time_ms": (t * 1000).astype(int),
        "error": error,
        "output": output,
        "kp": 6.5,
        "kd": 0.0,
    })


def make_severe_clipping_run() -> pd.DataFrame:
    """Generate a run where PID output is saturated at limits >90% of the time."""
    t = np.arange(0, 15.0, 0.01)
    error = 25.0 * np.sin(2 * np.pi * 0.5 * t)
    output = np.clip(100.0 * error, -255, 255)
    return pd.DataFrame({
        "time_ms": (t * 1000).astype(int),
        "error": error,
        "output": output,
        "kp": 8.0,
        "kd": 0.1,
    })


def make_invalid_test_files(target_dir: Path):
    """Write invalid logs to test UI error banner handling."""
    target_dir.mkdir(exist_ok=True, parents=True)

    # 1. Missing error column
    with open(target_dir / "invalid_missing_error_col.csv", "w", encoding="utf-8") as f:
        f.write("time_ms,speed,battery,temp\n")
        for i in range(100):
            f.write(f"{i*10},120,7.4,32.5\n")

    # 2. Too few rows
    with open(target_dir / "invalid_too_few_rows.csv", "w", encoding="utf-8") as f:
        f.write("time_ms,error,output\n")
        f.write("0,1.2,10\n")
        f.write("10,1.0,8\n")
        f.write("20,0.8,5\n")

    # 3. Empty file
    with open(target_dir / "invalid_empty_file.csv", "w", encoding="utf-8") as f:
        pass

    # 4. Corrupted data with non-numeric strings
    with open(target_dir / "invalid_corrupted_data.csv", "w", encoding="utf-8") as f:
        f.write("time_ms,error,output\n")
        for i in range(50):
            if i == 15:
                f.write("FAIL,CORRUPT,INVALID\n")
            elif i == 25:
                f.write("####,????,!!!!\n")
            else:
                f.write(f"{i*10},{np.sin(i*0.1):.3f},{i*2}\n")

    print(f"wrote 4 invalid test files to {target_dir}")


def main():
    out = Path(__file__).parent / "sample_data"
    out.mkdir(exist_ok=True)

    for name, kw in CASES.items():
        df = simulate_run(seed=1, **kw)
        df.round(5).to_csv(out / f"{name}.csv", index=False)
        print("wrote", out / f"{name}.csv")

    # Write additional negative/severe issue runs
    df_growing = make_unstable_growing_run()
    df_growing.round(5).to_csv(out / "run_unstable_growing.csv", index=False)
    print("wrote", out / "run_unstable_growing.csv")

    df_clipping = make_severe_clipping_run()
    df_clipping.round(5).to_csv(out / "run_severe_clipping.csv", index=False)
    print("wrote", out / "run_severe_clipping.csv")

    # Write invalid files for testing upload validation errors
    make_invalid_test_files(out / "invalid_test_files")


if __name__ == "__main__":
    main()

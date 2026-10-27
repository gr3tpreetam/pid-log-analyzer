# PID Log Analyzer
### *Rule-Based Diagnosis & Telemetry for Line-Follower Robots*

[![Python](https://img.shields.io/badge/Python-3.10%2B-00f0ff?style=for-the-badge&logo=python&logoColor=black)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-33%2F33%20Passing-39ff14?style=for-the-badge&logo=pytest&logoColor=black)](tests/)
[![UI](https://img.shields.io/badge/UI-Cyber%20Neon%20Dashboard-ff0055?style=for-the-badge&logo=streamlit&logoColor=white)](app.py)
[![Demo Video](https://img.shields.io/badge/Demo_Video-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](#-project-demo-video)
[![Status](https://img.shields.io/badge/Status-End--Term%20Project-bc13fe?style=for-the-badge)]()

---

## 📹 Project Demo Video

> **Google Drive Link:** [Click here to watch the PID Log Analyzer Demo Video](PASTE_YOUR_GOOGLE_DRIVE_SHAREABLE_LINK_HERE)  
> 
> *Note for evaluators: Viewing permissions are configured as **"Anyone with the link can view"** so the video streams directly without access barriers.*

---

## 📌 The Problem
Tuning a line-follower robot by watching it run on the track only shows the **symptom** (it wobbles, oscillates, or overshoots curves). 
- General serial plotters (*Teleplot, PlotJuggler, pid-scope*) plot raw lines, but **do not diagnose why** the robot behaves poorly or which gain ($K_p$, $K_d$, or $K_i$) needs adjustment.
- Drone blackbox tools (*Betaflight Blackbox, INAV toolkit*) perform automated diagnostics, but are built strictly for quadcopter flight dynamics.
- **I couldn't find an offline diagnostics tool built specifically for ground line-follower robots.**

This project bridges that gap: upload a CSV log recorded from your robot's microcontroller (Arduino, STM32, ESP32) and get **plain-language findings with the exact mathematical evidence** that triggered them.

---

## ⚡ Key Control-Theory Insights

A line follower **has no step input**—its setpoint is perpetually $\text{error} = 0$, while track curves act as continuous external disturbances. Classic step-response metrics (*rise time, settling time, percent overshoot*) cannot be computed directly.

This tool evaluates controller health using four signal-processing foundations:

| Diagnostic Metric | Engineering Principle | Target Question |
|---|---|---|
| **Zero-Crossing Decay Ratio** | Evaluates peak ratios across successive sign transitions shielded by an auto-adaptive deadband. | Is the wobble settling ($<1.0$), sustained ($\approx 1.0$), or exponentially growing ($>1.1$)? |
| **Welch Power Spectrum (FFT)** | Computes dominant repeating frequencies and non-DC spectral power share. | Is the wobble a high-frequency controller vibration ($>1.0$ Hz) or slow track curvature ($<1.0$ Hz)? |
| **Big-Error Curve Episodes** | Identifies episodes where $\|error\| > 3 \times \text{MAD}$ scale, evaluated post-bias correction. | How sluggish is the robot when recovering from sharp turns? |
| **Actuation Saturation & Jitter** | Calculates output clipping percentage and jitter ratio ($\sigma_{\Delta u} / \sigma_u$). | Are motors saturated at $\pm 255$, or is high $K_d$ amplifying sensor noise? |
| **Steady-State Bias Ratio** | Computes $\|\text{mean}\| / \text{RMS}$ error. | Is there motor trim imbalance or sensor calibration offset? |

---

## 🎮 Interface & Telemetry Dashboard

The dashboard is engineered with a **Cyber-Robotic / Neon Mecha** aesthetic (`app.py`):
- **Typography**: Google Fonts `Orbitron` (HUD headers, cyber badges) and `Share Tech Mono` (digital telemetry registers).
- **Interactive Waveforms**: High-contrast Plotly chart with a glowing Electric Cyan (`#00f0ff`) error trace, Neon Lime (`#39ff14`) motor output trace, and glowing Neon Amber (`#ffaa00`) curve-episode highlight bands.
- **Start Analysis Workflow**: Prominent primary button ensuring analysis only executes on command with real-time toast feedback.
- **Multi-Run Matrix**: Ranked side-by-side comparison table and neon bar chart comparing different gain configurations.
- **CLI Mode**: Full terminal-based batch diagnostics via `cli.py`.

---

## 🚀 Quick Start (Windows PowerShell)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Generate Simulated Benchmark Runs
No robot required to test the software! Generates 7 benchmark runs and invalid test files:
```powershell
python make_samples.py
```

### 3. Launch the Neon Cyber Dashboard
```powershell
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. (If using Brave Browser, remember to toggle Brave Shields OFF for `localhost`).

### 4. Run CLI Diagnostics in Terminal
```powershell
python cli.py sample_data/run_good.csv sample_data/run_oscillating.csv sample_data/run_unstable_growing.csv
```

### 5. Execute Automated Unit Tests (33 Tests)
```powershell
python -m pytest tests/test_analysis.py -v
```

---

## 📊 Benchmark Datasets & Negative Reports

The simulator generates calibrated test cases covering healthy, degraded, and catastrophic controller states:

| Benchmark File | Primary Finding | Severity | Root Cause & Tuning Advice |
|---|---|---|---|
| `run_good.csv` | **No major issues detected** | `[OK]` | Well-tuned $K_p=2.5, K_d=0.08$. Damped settling. |
| `run_unstable_growing.csv` | **Growing oscillation** | `[BAD]` | Loop unstable ($K_p=6.5, K_d=0$). Decay ratio $1.14 > 1$. Reduce $K_p$ by 30-50%. |
| `run_severe_clipping.csv` | **Output clipping** | `[BAD]` | Motors pegged at limit 93% of the time. Reduce $K_p$ or widen PWM range. |
| `run_oscillating.csv` | **Sustained oscillation** | `[BAD]` | $2.1$ Hz limit cycle ($K_p=5.0$). Decay ratio $1.00$. Lower $K_p$ or raise $K_d$. |
| `run_sluggish.csv` | **Slow recovery from big errors** | `[WARN]` | Low gain ($K_p=0.4, K_d=0$). Episodes last 0.50s. Increase $K_p$. |
| `run_noisy_kd.csv` | **Twitchy output** | `[WARN]` | High derivative gain ($K_d=0.4$). Jitter ratio $1.70$. Lower $K_d$ or add LPF. |
| `run_offset.csv` | **Steady-state offset (bias)** | `[INFO]` | Motor asymmetry/sensor zero-shift. Mean error $+0.100$. Calibrate or add small $K_i$. |

### Invalid Upload Test Suite (`sample_data/invalid_test_files/`)
Drag and drop these into the UI to verify robust error rejection:
- `invalid_missing_error_col.csv` $\rightarrow$ Reports missing error column.
- `invalid_too_few_rows.csv` $\rightarrow$ Rejects files with $<20$ samples.
- `invalid_empty_file.csv` $\rightarrow$ Catches zero-byte logs.

---

## 💾 Hardware Logging Format (Arduino / STM32 / ESP32)

Output comma-separated telemetry over UART/Serial at $\sim 100$ Hz. Save the serial terminal output as `.csv` (via PuTTY, TeraTerm, or Arduino Serial Monitor):

```c
// At boot / setup:
Serial.println("time_ms,error,output,kp,kd");

// Inside control loop (~100 Hz):
void loop() {
    uint32_t now = millis();
    int16_t error = read_line_sensor();  // setpoint - measured
    int16_t output = compute_pid(error); // motor correction (-255 to 255)
    
    Serial.print(now);        Serial.print(",");
    Serial.print(error);      Serial.print(",");
    Serial.print(output);     Serial.print(",");
    Serial.print(Kp);         Serial.print(",");
    Serial.println(Kd);
    
    delay(10);
}
```

---

## 🛠️ Project Architecture

```
Mern Project/
├── analysis/                 # Core signal processing & diagnostic engine
│   ├── __init__.py           # Package namespace
│   ├── load.py               # Robust CSV cleaner (aliases, junk rows, boot noise)
│   ├── metrics.py            # Zero-crossings, Welch FFT, MAD events, clipping, jitter
│   ├── diagnose.py           # Rules of thumb, thresholds, and evidence formatting
│   └── simulate.py           # Physics simulation of 1-D line-follower dynamics
├── sample_data/              # Calibrated simulated runs & negative test cases
│   ├── run_good.csv          # Healthy baseline
│   ├── run_oscillating.csv   # Sustained limit cycle
│   ├── run_unstable_growing.csv # Catastrophic unstable growth
│   ├── run_severe_clipping.csv  # 93% saturated actuator
│   ├── run_sluggish.csv      # Sluggish recovery after curves
│   ├── run_noisy_kd.csv      # Noisy jittery actuation
│   ├── run_offset.csv        # Steady-state calibration bias
│   └── invalid_test_files/   # Malformed CSVs for validation testing
├── tests/
│   └── test_analysis.py      # 33 unit tests for loader, metrics, and diagnoses
├── app.py                    # Streamlit web app with Cyber-Neon Mecha HUD
├── cli.py                    # Terminal batch diagnostics CLI
├── make_samples.py           # Benchmark dataset generator
├── requirements.txt          # Python dependencies
└── README.md                 # Engineering documentation
```

---

## ⚠️ Limitations
- **Heuristic Rules of Thumb**: Findings provide targeted engineering hints based on signal patterns, not an absolute formal control proof. Thresholds can be fine-tuned in the sidebar.
- **Track Layout Blindness**: The analyzer cannot physically "see" the track. A slow weave caused by a sinusoidal track layout looks identical to slow controller drift. The default threshold treats wobble slower than $1.0$ Hz as track layout.
- **Comparison Context**: Multi-run RMS ranking is strictly valid when runs are performed on the same track at the same base speed.

---

## 🎓 Academic Submission Note
Built as an **End-Term Software Project**. AI tools were used for code structuring, styling assistance, and rapid drafting; all underlying mathematical logic, physics modeling, signal processing rules, and debugging were validated and verified end-to-end.

"""Load and clean CSV logs from line-follower robots.

Handles common messiness: serial boot messages before the header,
``time_ms`` vs ``time_s``, ``error`` vs ``setpoint`` + ``measured``,
and non-numeric junk rows.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class LogError(Exception):
    """Raised when a log file cannot be turned into a usable DataFrame."""


# Aliases mapped to (canonical name, divisor to get seconds)
_TIME_ALIASES_MS = {"time_ms", "ms", "t_ms"}
_TIME_ALIASES_S = {"time_s"}


def _find_header_start(lines: list[str]) -> int:
    """Return the index of the first line that looks like a CSV header.

    Serial bootloaders often print text like "BOOT OK" or "[ready]" before
    the actual header.  We scan for the first line whose first token is a
    plausible column name.
    """
    known = {
        "time_ms", "ms", "t_ms", "time_s", "t", "time",
        "error", "setpoint", "measured", "output", "kp", "ki", "kd",
    }
    for idx, line in enumerate(lines):
        tokens = [tok.strip().lower() for tok in line.split(",")]
        if any(tok in known for tok in tokens):
            return idx
    return 0  # fall back to the very first line


def load_log(source: Union[str, Path, io.IOBase, bytes]) -> pd.DataFrame:
    """Read a CSV log and return a tidy DataFrame with at least ``t`` and ``error``.

    Parameters
    ----------
    source : path-like, file-like or bytes
        The CSV file to read.

    Returns
    -------
    pd.DataFrame
        Always has columns ``t`` (seconds, starting from 0) and ``error``.
        May also have ``output``, ``kp``, ``ki``, ``kd`` if they were present.

    Raises
    ------
    LogError
        If the required columns cannot be found or there are fewer than 20
        usable rows.
    """
    # --- read raw text -------------------------------------------------------
    if isinstance(source, bytes):
        raw = source.decode("utf-8", errors="replace")
    elif isinstance(source, (str, Path)):
        raw = Path(source).read_text(encoding="utf-8", errors="replace")
    else:
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

    # --- strip junk lines before the header ----------------------------------
    lines = raw.splitlines(keepends=True)
    hdr = _find_header_start(lines)
    cleaned = "".join(lines[hdr:])

    # --- parse, coerce errors to NaN, then drop bad rows ---------------------
    try:
        df = pd.read_csv(io.StringIO(cleaned), on_bad_lines="skip")
    except Exception as exc:
        raise LogError(f"CSV parse error: {exc}") from exc

    # Strip whitespace from column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Coerce all columns to numeric; junk rows become NaN and are dropped
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(how="all")

    # --- time column ---------------------------------------------------------
    ms_cols = _TIME_ALIASES_MS & set(df.columns)
    s_cols = _TIME_ALIASES_S & set(df.columns)

    if ms_cols:
        tcol = next(iter(ms_cols))
        df["t"] = df[tcol].astype(float) / 1000.0
    elif s_cols:
        tcol = next(iter(s_cols))
        df["t"] = df[tcol].astype(float)
    elif "t" in df.columns:
        pass  # already have it
    elif "time" in df.columns:
        # Ambiguous: decide by median step size.  If median step >= 1 it is ms.
        vals = df["time"].dropna().astype(float)
        if len(vals) > 1 and np.median(np.diff(vals)) >= 1.0:
            df["t"] = vals / 1000.0
        else:
            df["t"] = vals
    else:
        raise LogError(
            "No time column found. Need one of: time_ms, ms, t_ms, time_s, t, time."
        )

    # --- error column --------------------------------------------------------
    if "error" not in df.columns:
        if "setpoint" in df.columns and "measured" in df.columns:
            df["error"] = df["setpoint"] - df["measured"]
        else:
            raise LogError(
                "No error column found.  Need 'error' or 'setpoint' + 'measured'."
            )

    # --- drop rows where critical columns are NaN ----------------------------
    df = df.dropna(subset=["t", "error"])
    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise LogError(
            f"Only {len(df)} usable rows after cleaning (need at least 20)."
        )

    # --- zero-offset time ----------------------------------------------------
    df["t"] = df["t"] - df["t"].iloc[0]

    return df

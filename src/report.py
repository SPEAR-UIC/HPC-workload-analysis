"""
Text Report Generator for HPC Workload Analysis.

This module writes **plain-text summary reports** describing per-queue
statistics.  The reports are consumed by humans (not parsed by other
modules) so they aim for readability.

The single public function ``write_detailed_queue_summary`` iterates
over every queue in the dataset and writes:

* Job count and total core-hours (with percentage share).
* Walltime efficiency (``runtime / walltime``).
* Descriptive statistics (mean, median, P5, P95, min, max, std) for:
  - nodes_used
  - nodes_requested
  - walltime_seconds
  - runtime_seconds
  - wait_seconds
  - used_core_hours
* Unique user / project counts per queue and overall.

Output file example
-------------------
::

    Polaris 2024 – Detailed Queue Summary
    ======================================

    Total jobs:       120,000
    Total core-hours: 5,432,100.00
    Queues:           8

    ------------------------------------------------------------------------
    Queue: small
    ------------------------------------------------------------------------
      Jobs:           18,000
      Total Core-Hrs: 540,000.00 (9.94% of total)
      Walltime Efficiency (runtime/walltime):
        Mean: 72.30%  Median: 78.15%  Std: 18.44%
      ...
"""

import os
import pandas as pd
import numpy as np
from utils import _fmt, _seconds_to_hms


def write_detailed_queue_summary(df, out_path, title):
    """
    Write a rich per-queue statistics report to a plain-text file.

    For each queue the report contains: job count, total core-hours,
    share of total core-hours, mean/median/P5/P95/min/max for several
    numeric columns, walltime efficiency, and unique user/project counts.

    Parameters
    ----------
    df : pd.DataFrame
        Job dataset (must contain ``queue_name`` and ``used_core_hours``).
    out_path : str
        Full path to the output text file.
    title : str
        Title line written at the top of the report.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Columns to summarise – each maps to a human label and an optional
    # formatting function (_seconds_to_hms for time columns, None for
    # plain numbers).
    stat_cols = {
        "nodes_used": ("Nodes Used", None),
        "nodes_requested": ("Nodes Requested", None),
        "walltime_seconds": ("Walltime", _seconds_to_hms),
        "runtime_seconds": ("Runtime", _seconds_to_hms),
        "wait_seconds": ("Wait Time", _seconds_to_hms),
        "used_core_hours": ("Core-Hours per Job", None),
    }

    queues = sorted(df["queue_name"].dropna().unique())
    total_hours = df["used_core_hours"].sum() if "used_core_hours" in df.columns else 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(f"Total jobs:       {len(df):,}\n")
        f.write(f"Total core-hours: {_fmt(total_hours)}\n")
        f.write(f"Queues:           {len(queues)}\n")
        f.write("\n")

        for queue in queues:
            qdf = df[df["queue_name"] == queue]
            n_jobs = len(qdf)
            q_hours = (
                qdf["used_core_hours"].sum() if "used_core_hours" in qdf.columns else 0
            )
            share = (q_hours / total_hours * 100) if total_hours > 0 else 0

            f.write("-" * 72 + "\n")
            f.write(f"Queue: {queue}\n")
            f.write("-" * 72 + "\n")
            f.write(f"  Jobs:           {n_jobs:,}\n")
            f.write(f"  Total Core-Hrs: {_fmt(q_hours)} ({share:.2f}% of total)\n")

            # Efficiency: runtime / walltime
            if "runtime_seconds" in qdf.columns and "walltime_seconds" in qdf.columns:
                eff = (
                    qdf["runtime_seconds"] / qdf["walltime_seconds"].replace(0, np.nan)
                ).dropna()
                if len(eff) > 0:
                    f.write(f"  Walltime Efficiency (runtime/walltime):\n")
                    f.write(
                        f"    Mean: {eff.mean():.2%}  Median: {eff.median():.2%}  "
                        f"Std: {eff.std():.2%}\n"
                    )

            for col, (label, fmt_fn) in stat_cols.items():
                if col not in qdf.columns:
                    continue
                s = qdf[col].dropna()
                if s.empty:
                    continue

                mean_v = s.mean()
                med_v = s.median()
                std_v = s.std()
                p5_v = s.quantile(0.05)
                p95_v = s.quantile(0.95)
                min_v = s.min()
                max_v = s.max()

                if fmt_fn is not None:
                    f.write(f"  {label}:\n")
                    f.write(f"    Mean: {fmt_fn(mean_v)}  Median: {fmt_fn(med_v)}\n")
                    f.write(f"    P5: {fmt_fn(p5_v)}  P95: {fmt_fn(p95_v)}\n")
                    f.write(f"    Min: {fmt_fn(min_v)}  Max: {fmt_fn(max_v)}\n")
                else:
                    f.write(f"  {label}:\n")
                    f.write(
                        f"    Mean: {_fmt(mean_v)}  Median: {_fmt(med_v)}  "
                        f"Std: {_fmt(std_v)}\n"
                    )
                    f.write(f"    P5: {_fmt(p5_v)}  P95: {_fmt(p95_v)}\n")
                    f.write(f"    Min: {_fmt(min_v)}  Max: {_fmt(max_v)}\n")
            # Unique users / projects if available
            if "username_genid" in qdf.columns:
                f.write(f"  Unique Users:    {qdf['username_genid'].nunique():,}\n")
            if "project_name_genid" in qdf.columns:
                f.write(f"  Unique Projects: {qdf['project_name_genid'].nunique():,}\n")

            f.write("\n")
        # total unique users / projects across all queues
        if "username_genid" in df.columns:
            f.write(f"Total Unique Users:    {df['username_genid'].nunique():,}\n")
        if "project_name_genid" in df.columns:
            f.write(f"Total Unique Projects: {df['project_name_genid'].nunique():,}\n")
    print(f"  \u2705 Detailed queue summary saved: {out_path}")

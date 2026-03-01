"""
Queue-Level Core-Hour Analysis for HPC Workload Analysis.

This module generates **per-queue breakdowns** of core-hours at
multiple time granularities (daily, weekly, monthly) and writes both
visualisation files and plain-text summary reports.

Entry point
-----------
``generate_queue_analysis(df, save_dir, machine_name, color_map, ...)``
is called from ``orchestrator.distributions_analysis`` when the
``--queue-analysis`` flag is active.

Granularity logic
-----------------
The function inspects the date range in the data:

* **Multiple years** → ``general_queue_analysis`` (monthly stacked bar /
  area / pie) **plus** yearly drill-down via ``yearly_queue_analysis``.
* **Single year, multiple months** → ``yearly_queue_analysis`` (weekly
  charts) **plus** monthly drill-down via ``monthly_queue_analysis``.
* **Single month** → ``monthly_queue_analysis`` (daily charts).

At every granularity level the following outputs are produced:

* Stacked-bar chart of core-hours by queue.
* Cumulative stacked-area chart.
* Pie chart of overall queue shares.
* Plain-text detailed queue summary (all jobs).
* Plain-text detailed queue summary (successful jobs only).

Dependencies
------------
plotting (stacked bar / area / pie), report (text summary), utils
(pivoting helpers, colour map, successful-job filter).
"""

import os
import pandas as pd

# ---- Project-local imports --------------------------------------------------
from plotting import (
    plot_pie_chart_queue_percentages,
    plot_stacked_bar_queue_percentages,
    plot_stacked_area_queue_percentages,
)
from report import write_detailed_queue_summary
from utils import (
    _build_color_map,
    _pivot_daily_core_hours,
    _pivot_weekly_core_hours,
    _pivot_monthly_core_hours,
    _get_successful_jobs,
)


# =============================================================================
#  DAILY CORE-HOUR ANALYSIS  (stacked bar + area + pie for each day)
# =============================================================================


def daily_core_hours_analysis(df, save_dir, prefix, label, color_map):
    """Generate daily-granularity core-hour charts (bar, area, pie)."""
    pivoted_daily = _pivot_daily_core_hours(df)

    plot_stacked_bar_queue_percentages(
        pivoted_daily,
        os.path.join(save_dir, f"{prefix}_daily_stacked_bar.png"),
        f"{label} – Daily Core-Hours by Queue (Stacked Bar)",
        color_map,
    )
    plot_stacked_area_queue_percentages(
        pivoted_daily,
        os.path.join(save_dir, f"{prefix}_daily_cumulative_area.png"),
        f"{label} – Cumulative Daily Core-Hours by Queue (Area)",
        color_map,
    )
    plot_pie_chart_queue_percentages(
        pivoted_daily,
        os.path.join(save_dir, f"{prefix}_daily_pie_chart.png"),
        f"{label} – Overall Core-Hours by Queue (Pie Chart)",
        color_map,
    )


# =============================================================================
#  WEEKLY CORE-HOUR ANALYSIS  (stacked bar + area + pie for each week)
# =============================================================================


def weekly_core_hours_analysis(df, save_dir, prefix, label, color_map):
    """Generate weekly-granularity core-hour charts (bar, area, pie)."""
    pivoted_weekly = _pivot_weekly_core_hours(df)

    plot_stacked_bar_queue_percentages(
        pivoted_weekly,
        os.path.join(save_dir, f"{prefix}_weekly_stacked_bar.png"),
        f"{label} – Weekly Core-Hours by Queue (Stacked Bar)",
        color_map,
    )
    plot_stacked_area_queue_percentages(
        pivoted_weekly,
        os.path.join(save_dir, f"{prefix}_weekly_cumulative_area.png"),
        f"{label} – Cumulative Weekly Core-Hours by Queue (Area)",
        color_map,
    )
    plot_pie_chart_queue_percentages(
        pivoted_weekly,
        os.path.join(save_dir, f"{prefix}_weekly_pie_chart.png"),
        f"{label} – Overall Weekly Core-Hours by Queue (Pie Chart)",
        color_map,
    )


# =============================================================================
#  MONTHLY CORE-HOUR ANALYSIS  (stacked bar + area + pie for each month)
# =============================================================================


def monthly_core_hours_analysis(df, save_dir, prefix, label, color_map):
    """Generate monthly-granularity core-hour charts (bar, area, pie)."""
    pivoted_monthly = _pivot_monthly_core_hours(df)

    plot_stacked_bar_queue_percentages(
        pivoted_monthly,
        os.path.join(save_dir, f"{prefix}_monthly_stacked_bar.png"),
        f"{label} – Monthly Core-Hours by Queue (Stacked Bar)",
        color_map,
    )
    plot_stacked_area_queue_percentages(
        pivoted_monthly,
        os.path.join(save_dir, f"{prefix}_monthly_cumulative_area.png"),
        f"{label} – Cumulative Monthly Core-Hours by Queue (Area)",
        color_map,
    )
    plot_pie_chart_queue_percentages(
        pivoted_monthly,
        os.path.join(save_dir, f"{prefix}_monthly_pie_chart.png"),
        f"{label} – Overall Monthly Core-Hours by Queue (Pie Chart)",
        color_map,
    )


# =============================================================================
#  MONTHLY DRILL-DOWN
#  For every (year, month) in the data: daily charts + detailed summaries.
# =============================================================================


def monthly_queue_analysis(df, save_dir, machine_name, color_map=None):
    """Generate daily core-hour charts and text summaries for each month."""
    # Parse timestamps and extract year/month columns
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    df["year"] = df["start_timestamp"].dt.year
    df["month"] = df["start_timestamp"].dt.month
    for year in sorted(df["year"].dropna().unique()):
        for month in sorted(df["month"].dropna().unique()):
            monthly_df = df[(df["year"] == year) & (df["month"] == month)]
            if monthly_df.empty:
                continue
            prefix = machine_name + f"_{year}_{month:02d}_queue_analysis"
            label = f"{machine_name} {year}-{month:02d}"

            # ── Daily core-hour charts ──
            daily_core_hours_analysis(
                monthly_df,
                save_dir,
                prefix,
                label,
                color_map,
            )

            # ── Text summaries (all jobs + successful only) ──
            write_detailed_queue_summary(
                monthly_df,
                os.path.join(save_dir, f"{prefix}_detailed_summary.txt"),
                f"{label} – Detailed Queue Summary",
            )

            # Successful-jobs-only summary
            monthly_df_success = _get_successful_jobs(monthly_df)
            write_detailed_queue_summary(
                monthly_df_success,
                os.path.join(
                    save_dir, f"{prefix}_successful_jobs_detailed_summary.txt"
                ),
                f"{label} – Detailed Queue Summary (Successful Jobs Only)",
            )


# =============================================================================
#  YEARLY DRILL-DOWN
#  For every year in the data: weekly charts + detailed summaries.
# =============================================================================


def yearly_queue_analysis(df, save_dir, machine_name, color_map=None):
    """Generate weekly core-hour charts and text summaries for each year."""

    # Parse timestamps and extract a year column
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    df["year"] = df["start_timestamp"].dt.year
    for year in sorted(df["year"].dropna().unique()):
        yearly_df = df[df["year"] == year]
        prefix = machine_name + f"_{year}_queue_analysis"
        label = f"{machine_name} {(year)}"

        # ── Weekly core-hour charts ──
        weekly_core_hours_analysis(yearly_df, save_dir, prefix, label, color_map)

        # ── Text summaries (all jobs + successful only) ──
        write_detailed_queue_summary(
            yearly_df,
            os.path.join(save_dir, f"{prefix}_detailed_summary.txt"),
            f"{label} – Detailed Queue Summary",
        )

        # Successful-jobs-only summary
        yearly_df_success = _get_successful_jobs(yearly_df)
        write_detailed_queue_summary(
            yearly_df_success,
            os.path.join(save_dir, f"{prefix}_successful_jobs_detailed_summary.txt"),
            f"{label} – Detailed Queue Summary (Successful Jobs Only)",
        )


# =============================================================================
#  GENERAL (FULL-RANGE) QUEUE ANALYSIS
#  Picks daily / weekly / monthly granularity based on the date range.
# =============================================================================


def general_queue_analysis(df, save_dir, machine_name, color_map=None):
    """
    Auto-select granularity and generate core-hour charts + summaries.

    * < 30 days  → daily charts.
    * < 1 year   → weekly charts.
    * ≥ 1 year   → monthly charts.
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = machine_name + "_queue_analysis"
    label = machine_name

    # Determine the date range of the dataset
    min_date = pd.to_datetime(df["start_timestamp"], errors="coerce").min()
    max_date = pd.to_datetime(df["start_timestamp"], errors="coerce").max()

    # ---- Choose granularity based on date span ------------------------------
    if (max_date - min_date).days < 30:

        # ── DAILY ────────────────────────────────────────────────────────────
        daily_core_hours_analysis(df, save_dir, prefix, label, color_map)

    elif (max_date - min_date).days < 365:

        # ── WEEKLY ────────────────────────────────────────────────────────────
        weekly_core_hours_analysis(df, save_dir, prefix, label, color_map)
    else:

        # ── MONTHLY ────────────────────────────────────────────────────────────
        monthly_core_hours_analysis(df, save_dir, prefix, label, color_map)

    # ---- Text summaries (all jobs + successful only) ------------------------
    write_detailed_queue_summary(
        df,
        os.path.join(save_dir, f"{prefix}_detailed_summary.txt"),
        f"{label} – Detailed Queue Summary",
    )

    df_success = _get_successful_jobs(df)
    # Successful-jobs-only summary
    write_detailed_queue_summary(
        df_success,
        os.path.join(save_dir, f"{prefix}_successful_jobs_detailed_summary.txt"),
        f"{label} – Detailed Queue Summary (Successful Jobs Only)",
    )


# =============================================================================
#  PUBLIC ENTRY POINT  –  called from orchestrator.distributions_analysis
# =============================================================================


def generate_queue_analysis(
    df, save_dir, machine_name, color_map=None, do_yearly=False, do_monthly=False
):
    """
    Generate the complete queue core-hour analysis.

    Automatically chooses the right granularity based on the date range
    and creates all chart + summary outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``start_timestamp``, ``queue_name``, ``used_core_hours``.
    save_dir : str
        Directory where outputs are saved.
    machine_name : str
        Machine label for titles / filenames.
    color_map : dict or None
        ``{queue_name: (r,g,b)}``; built automatically if ``None``.
    do_yearly : bool
        Forward flag (currently used to decide sub-period granularity).
    do_monthly : bool
        Forward flag (currently used to decide sub-period granularity).
    """
    if df.empty:
        print(f"  ⚠️  No data for queue analysis, skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Build colour map from available queue names if not provided
    if color_map is None:
        color_map = _build_color_map(df["queue_name"].dropna().unique())

    # Decide analysis scope based on the number of distinct years / months
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    years = df["start_timestamp"].dt.year.nunique()
    months = df["start_timestamp"].dt.month.nunique()
    if years > 1:
        general_queue_analysis(df, save_dir, machine_name, color_map)
    elif months > 1:
        yearly_queue_analysis(df, save_dir, machine_name, color_map)
    else:
        monthly_queue_analysis(df, save_dir, machine_name, color_map)

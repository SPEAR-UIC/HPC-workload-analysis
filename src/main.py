"""
Main Entry Point for HPC Job Log Analysis.

This module is the **command-line entry point** for the entire analysis
pipeline.  It reads a preprocessed job log CSV, loads per-machine
configuration files, and delegates work to the orchestrator and
system-utilization modules.

High-level workflow
-------------------
1. Parse CLI arguments  (→ ``utils._parse_arguments``).
2. Read the compressed CSV (``jobs_preprocessed.csv.gz``).
3. Load machine configuration  (→ ``utils._get_machine_config``).
4. Run the **full-dataset** analysis  (→ ``main_workflow``).
5. Optionally split the data by **year**  (→ ``run_per_year_analysis``)
   and/or by **month**  (→ ``run_per_month_analysis``) and repeat
   the analysis for each slice.

The analysis performed by ``main_workflow`` includes:
* Time-distribution plots (by hour, weekday, day-of-year)
* Job-size and walltime distribution bar charts
* 2-D heatmap (job-size vs. walltime)
* Temporal scatter plot of jobs
* Optional per-queue distribution analysis
* Optional queue core-hour breakdown (stacked bar / area / pie)
* Optional machine utilization over time

CLI flags
---------
See ``--help`` or ``utils._parse_arguments`` for the full list.
Key flags:

* ``--path``                  – path to the ``.csv.gz`` input file
* ``--machine-name``          – label used in titles / filenames
* ``--queue-analysis``        – per-queue core-hour breakdown
* ``--machine-utilization``   – node-utilization time series
* ``--yearly``                – split by year
* ``--monthly``               – split by month
* ``--full-queue-analysis``   – per-queue distribution plots

Usage examples
--------------
::

    # Basic run (distribution plots only)
    python main.py --path ../data/Polaris/jobs_preprocessed.csv.gz \
                   --machine-name Polaris

    # Full analysis with all extras
    python main.py --path ../data/Polaris/jobs_preprocessed.csv.gz \
                   --machine-name Polaris \
                   --queue-analysis --machine-utilization \
                   --yearly --monthly --full-queue-analysis
"""

# ---- Standard-library & third-party imports --------------------------------
import os
import pandas as pd

# ---- Project-local imports --------------------------------------------------
from utils import (
    _parse_arguments,
    _get_machine_config,
)
from orchestrator import workOnAllQueues
from system_utilization import machine_utilization


# =============================================================================
#  MAIN WORKFLOW  –  called once for the full dataset, once per year, or
#                    once per month depending on the CLI flags.
# =============================================================================
def main_workflow(
    df,
    output_path,
    job_dim_df,
    walltime_df,
    max_nodes,
    queue_name_df,
    do_queue_analysis=False,
    machine_name="",
    year=None,
    month=None,
    do_machine_utilization=False,
    do_yearly=False,
    do_monthly=False,
    do_full_queue_analysis=False,
):
    """
    Execute the complete analysis pipeline for the given DataFrame slice.

    This is a thin wrapper that:
    1. Calls ``workOnAllQueues`` (orchestrator) for distribution / queue
       analysis.
    2. Optionally calls ``machine_utilization`` (system_utilization) for
       node-usage time-series analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Job log data (preprocessed).
    output_path : str
        Directory where all output (plots, reports) is written.
    job_dim_df : pd.DataFrame
        Node-count bin definitions (from ``job_dimension.csv``).
    walltime_df : pd.DataFrame
        Walltime bin definitions (from ``walltime.csv``).
    max_nodes : int
        Total machine capacity (from ``max_nodes.csv``).
    queue_name_df : pd.DataFrame or None
        Active queue definitions (from ``queue_names.csv``); ``None``
        disables queue-analysis.
    do_queue_analysis : bool
        Enable per-queue core-hour breakdown.
    machine_name : str
        Label used in plot titles and filenames.
    year, month : int or None
        If set, restrict data and label output accordingly.
    do_machine_utilization : bool
        Compute and plot node utilization over time.
    do_yearly / do_monthly : bool
        Forward flags to the orchestrator for sub-period analysis.
    do_full_queue_analysis : bool
        Generate per-queue distribution plots for every queue.
    """
    # ---- Step 1: distribution & queue analysis (orchestrator) ---------------
    workOnAllQueues(
        df,
        output_path=output_path,
        queue_name_df=queue_name_df,
        job_dim_df=job_dim_df,
        walltime_df=walltime_df,
        max_nodes=max_nodes,
        machine_name=machine_name,
        year=year,
        month=month,
        do_yearly=do_yearly,
        do_monthly=do_monthly,
        do_full_queue_analysis=do_full_queue_analysis,
        do_queue_analysis=do_queue_analysis,
    )

    # ---- Step 2: machine utilization (optional) -----------------------------
    if do_machine_utilization:
        df_usage = df.copy()  # work on a copy to avoid side-effects
        machine_utilization(
            df_usage,
            output_path=output_path,
            max_nodes=max_nodes,
            machine_name=machine_name,
            year=year,
            month=month,
        )


# =============================================================================
#  PER-YEAR ANALYSIS HELPER
#  Splits the dataset by calendar year and runs main_workflow on each slice.
# =============================================================================
def run_per_year_analysis(
    df,
    output_path,
    job_dim_df,
    walltime_df,
    max_nodes,
    queue_name_df,
    do_queue_analysis,
    do_machine_utilization,
    machine_name,
    do_full_queue_analysis,
):
    """
    Iterate over every calendar year present in *df* and call
    ``main_workflow`` for each one.

    Output is written to ``output_path/{YYYY}_analysis/``.
    """
    print("Running per-year analysis...")
    df["year"] = pd.to_datetime(df["start_timestamp"]).dt.year
    years = df["year"].unique()
    for year in years:
        print(f"Processing year: {year}")
        year_df = df[(df["start_timestamp"].dt.year == year)].copy()
        # Create output directory for the year in output_path/{year}_analysis/
        year_output_path = os.path.join(output_path, f"{year}_analysis")
        os.makedirs(year_output_path, exist_ok=True)
        main_workflow(
            year_df,
            output_path=year_output_path,
            job_dim_df=job_dim_df,
            walltime_df=walltime_df,
            max_nodes=max_nodes,
            queue_name_df=queue_name_df,
            do_queue_analysis=do_queue_analysis,
            machine_name=machine_name,
            year=year,
            month=None,
            do_machine_utilization=do_machine_utilization,
            do_full_queue_analysis=do_full_queue_analysis,
        )


# =============================================================================
#  PER-MONTH ANALYSIS HELPER
#  Splits the dataset by (year, month) pairs and runs main_workflow on each.
# =============================================================================
def run_per_month_analysis(
    df,
    output_path,
    input_path,
    job_dim_df,
    walltime_df,
    max_nodes,
    queue_name_df,
    do_queue_analysis,
    do_machine_utilization,
    machine_name,
    do_full_queue_analysis,
):
    """
    Iterate over every (year, month) combination in *df* and call
    ``main_workflow`` for each one.

    Output is written to ``output_path/{YYYY}_analysis/{MM}_analysis/``.
    """
    # Identify all unique year-month combinations, sorted chronologically
    print("Running per-month analysis...")
    df["year"] = pd.to_datetime(df["start_timestamp"]).dt.year
    df["month"] = pd.to_datetime(df["start_timestamp"]).dt.month
    year_months = df[["year", "month"]].drop_duplicates()
    # Order by year and month
    year_months = year_months.sort_values(by=["year", "month"])
    for _, row in year_months.iterrows():
        year = row["year"]
        month = row["month"]
        print(f"Processing year: {year}, month: {month}")
        month_df = df[
            (
                (df["start_timestamp"].dt.year == year)
                & (df["start_timestamp"].dt.month == month)
            )
        ].copy()
        # Create output directory for the month in output_path/{year}_{month}_analysis/
        month_output_path = os.path.join(
            output_path, f"{year}_analysis/{month}_analysis"
        )
        os.makedirs(month_output_path, exist_ok=True)
        main_workflow(
            month_df,
            output_path=month_output_path,
            job_dim_df=job_dim_df,
            walltime_df=walltime_df,
            queue_name_df=queue_name_df,
            max_nodes=max_nodes,
            do_queue_analysis=do_queue_analysis,
            machine_name=machine_name,
            year=year,
            month=month,
            do_machine_utilization=do_machine_utilization,
            do_full_queue_analysis=do_full_queue_analysis,
        )


# =============================================================================
#  MAIN FUNCTION  –  reads the CSV, loads config, and drives the analysis.
# =============================================================================
def main(
    input_path,
    output_path,
    do_queue_analysis,
    do_machine_utilization,
    do_yearly,
    machine_name,
    do_monthly,
    do_full_queue_analysis,
):
    """
    Top-level analysis driver.

    1. Reads the gzipped CSV at *input_path*.
    2. Loads machine config files from the same directory.
    3. Runs the full-dataset analysis.
    4. Optionally runs per-year and per-month breakdowns.

    Parameters
    ----------
    input_path : str
        Path to ``jobs_preprocessed.csv.gz``.
    output_path : str
        Base output directory (e.g. ``analysis_output/Polaris``).
    do_queue_analysis : bool
        Run per-queue core-hour breakdown.
    do_machine_utilization : bool
        Compute node utilization time series.
    do_yearly : bool
        Split analysis by calendar year.
    machine_name : str
        Machine label for titles and filenames.
    do_monthly : bool
        Split analysis by calendar month.
    do_full_queue_analysis : bool
        Generate per-queue distribution plots for every queue.
    """

    # ---- Read input data ----------------------------------------------------
    print("Reading compressed CSV...")
    df = pd.read_csv(input_path, compression="gzip")
    print("✅ Done reading CSV file:", input_path)

    # ---- Load machine-specific configs (bins, capacity) ---------------------
    job_dim_df, walltime_df, max_nodes = _get_machine_config(input_path)

    # ---- Optionally load queue definitions ----------------------------------
    if do_queue_analysis or do_full_queue_analysis:
        # If queue analysis is enabled, we need the queue_name_df to filter and label queues.
        queue_name_df = pd.read_csv(
            os.path.join(os.path.dirname(input_path), "queue_names.csv")
        )
    else:
        queue_name_df = None

    # ---- Parse timestamp columns --------------------------------------------
    for col in ["start_timestamp", "queued_timestamp", "end_timestamp"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---- Run full-dataset analysis ------------------------------------------
    main_workflow(
        df,
        output_path=output_path,
        job_dim_df=job_dim_df,
        walltime_df=walltime_df,
        max_nodes=max_nodes,
        queue_name_df=queue_name_df,
        do_queue_analysis=do_queue_analysis,
        machine_name=machine_name,
        year=None,
        month=None,
        do_machine_utilization=do_machine_utilization,
        do_yearly=do_yearly,
        do_monthly=do_monthly,
        do_full_queue_analysis=do_full_queue_analysis,
    )
    if do_yearly:
        # ---- Per-year slice analysis ----------------------------------------
        run_per_year_analysis(
            df,
            output_path=output_path,
            job_dim_df=job_dim_df,
            walltime_df=walltime_df,
            max_nodes=max_nodes,
            queue_name_df=queue_name_df,
            do_queue_analysis=do_queue_analysis,
            do_machine_utilization=do_machine_utilization,
            machine_name=machine_name,
            do_full_queue_analysis=do_full_queue_analysis,
        )
    if do_monthly:
        # ---- Per-month slice analysis ---------------------------------------
        run_per_month_analysis(
            df,
            output_path=output_path,
            input_path=input_path,
            job_dim_df=job_dim_df,
            walltime_df=walltime_df,
            max_nodes=max_nodes,
            queue_name_df=queue_name_df,
            do_queue_analysis=do_queue_analysis,
            do_machine_utilization=do_machine_utilization,
            machine_name=machine_name,
            do_full_queue_analysis=do_full_queue_analysis,
        )


# =============================================================================
#  CLI ENTRY POINT
#  Parse arguments, create output directory, and launch the analysis.
# =============================================================================
if __name__ == "__main__":

    args = _parse_arguments()

    # Collect all boolean flags into a dict for easy forwarding
    analysis_flags = dict(
        do_queue_analysis=args.queue_analysis,
        do_machine_utilization=args.machine_utilization,
        do_yearly=args.yearly,
        machine_name=args.machine_name,
        do_monthly=args.monthly,
        do_full_queue_analysis=args.full_queue_analysis,
    )

    # Log active flags for transparency
    print("Analysis flags:")
    for flag, value in analysis_flags.items():
        print(f"  {flag}: {value}")

    # ---- Prepare output directory -------------------------------------------
    # Convention: ../analysis_output/<machine_name>/
    output_path = os.path.join("../analysis_output", args.machine_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # ---- Launch analysis ----------------------------------------------------
    main(input_path=args.path, output_path=output_path, **analysis_flags)

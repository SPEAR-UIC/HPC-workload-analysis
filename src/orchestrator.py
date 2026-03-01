"""
Analysis Orchestrator for HPC Workload Analysis.

This module is the **central coordinator** between data preparation and
visualisation.  It receives a cleaned DataFrame from ``main.py`` and
fans out work to the plotting and per-queue analysis modules.

Key functions
-------------
``workOnAllQueues``
    Top-level entry called by ``main_workflow``.  Computes time
    distributions (hourly / weekday / yearly), then delegates to
    ``distributions_analysis``.

``distributions_analysis``
    Computes bin counts, generates the four main distribution plots,
    and optionally triggers per-queue breakdowns and queue core-hour
    analysis.

``print_distribution_analysis_over_queue_names``
    Iterates over every queue in the DataFrame and produces a full
    set of distribution plots for each one.

Output produced
---------------
All files are written under the *output_path* passed in:

* ``distribution_over_days.png``
* ``distribution_over_hours.png``
* ``distribution_over_years.png``
* ``job_size_distribution.png``
* ``distribution_over_walltime.png``
* ``<machine>_job_size_vs_walltime_heatmap.png``
* ``<machine>_distribution_over_time*.png``
* ``queue_<name>/`` sub-directories (if ``do_full_queue_analysis``)
* ``queue_analysis/`` sub-directory (if ``do_queue_analysis``)
"""

import os
import sys
import pandas as pd

# ---- Project-local imports --------------------------------------------------
from plotting import (
    plot_distribution_over_days,
    plot_distribution_over_hours,
    plot_distribution_over_years,
    plot_job_size_distribution,
    plot_walltime_distribution,
    plot_heatmap_job_size_vs_walltime,
    plot_heatmap_distribution_over_time,
)
from utils import (
    _add_distributions,
    _make_counts,
    _build_color_map,
)
from single_queue_analysis import generate_queue_analysis


# =============================================================================
#  PER-QUEUE DISTRIBUTION ANALYSIS
#  Generates job-size, walltime, heatmap, and temporal scatter plots for
#  each individual queue present in the DataFrame.
# =============================================================================
def print_distribution_analysis_over_queue_names(
    new_df,
    output_path,
    job_dim_df,
    walltime_df,
    global_max_nodes,
    max_time,
    machine_name="",
    year=None,
    month=None,
    queue_name_df=None,
):
    """
    Generate a full suite of distribution plots **per queue**.

    For every unique ``queue_name`` in *new_df* the function creates a
    sub-directory ``queue_<name>/`` and writes:

    * ``job_size_distribution.png``
    * ``distribution_over_walltime.png``
    * ``<machine>_job_size_vs_walltime_heatmap.png``
    * ``<machine>_distribution_over_time*.png``

    Parameters
    ----------
    new_df : pd.DataFrame
        Job dataset (already filtered to relevant columns).
    output_path : str
        Base output directory (a ``queue_<name>/`` folder is created
        inside).
    job_dim_df : pd.DataFrame
        Node-count bin definitions.
    walltime_df : pd.DataFrame
        Walltime bin definitions.
    global_max_nodes : int
        Machine capacity – used for colour scaling on scatter plots.
    max_time : int
        Largest walltime value – used for Y-axis limits.
    machine_name : str
        Label for plot titles.
    year, month : int or None
        If set, included in plot titles.
    """
    if queue_name_df is not None:
        valid_queues = set(queue_name_df["queue_name"].dropna().unique())
        print(
            f"  Allowed queues from queue_names.csv ({len(valid_queues)}): "
            f"{sorted(valid_queues)}"
        )
        new_df = new_df[new_df["queue_name"].isin(valid_queues)].copy()
        print(f"  Filtered to {new_df.shape[0]:,} rows after queue filtering.")
    else:
        raise ValueError(
            "queue_name_df is required for per-queue distribution analysis."
        )

    for name in new_df["queue_name"].unique():

        # ---- Isolate jobs for this queue ------------------------------------
        queue_df = new_df.copy()
        queue_df = queue_df[queue_df["queue_name"] == name]

        # ---- Create output sub-directory ------------------------------------
        queue_path = os.path.join(output_path, f"queue_{name}")
        os.makedirs(queue_path, exist_ok=True)

        print(f"Processing queue: {name} with {queue_df.shape[0]} jobs.")

        # ---- Compute bin counts for this queue ------------------------------
        queue_job_size_counts, queue_walltime_counts, queue_combined_counts = (
            _make_counts(queue_df, job_dim_df, walltime_df)
        )

        # ---- Generate plots -------------------------------------------------

        plot_job_size_distribution(
            queue_job_size_counts,
            queue_path,
            year=year,
            month=month,
            machine_name=machine_name,
        )
        plot_walltime_distribution(
            queue_walltime_counts,
            queue_path,
            year=year,
            month=month,
            machine_name=machine_name,
        )
        plot_heatmap_job_size_vs_walltime(
            job_dim_df,
            walltime_df,
            queue_combined_counts,
            queue_path,
            year=year,
            month=month,
            machine_name=machine_name,
        )
        plot_heatmap_distribution_over_time(
            queue_df,
            queue_path,
            global_max_nodes,
            max_time,
            year=year,
            month=month,
            machine_name=machine_name,
        )


# =============================================================================
#  MAIN DISTRIBUTION ANALYSIS ORCHESTRATOR
#  Computes overall bin counts, generates global plots, and optionally
#  triggers per-queue and queue core-hour analyses.
# =============================================================================
def distributions_analysis(
    job_dim_df,
    walltime_df,
    max_nodes,
    df,
    output_path,
    machine_name="",
    year=None,
    month=None,
    queue_name_df=None,
    do_yearly=False,
    do_monthly=False,
    do_full_queue_analysis=False,
    do_queue_analysis=False,
):
    """
    Central distribution analysis driver.

    Steps:
    1. Prepare a lightweight copy of *df* (only the columns needed).
    2. Build human-readable ``range_label`` columns on the bin DataFrames.
    3. Compute overall job-size, walltime, and combined counts.
    4. Plot the four main distribution charts.
    5. If ``do_full_queue_analysis``, call
       ``print_distribution_analysis_over_queue_names``.
    6. If ``do_queue_analysis``, filter to valid queues (from
       ``queue_name_df``) and call ``generate_queue_analysis``.

    Parameters
    ----------
    job_dim_df : pd.DataFrame
        Node-count bin definitions.
    walltime_df : pd.DataFrame
        Walltime bin definitions.
    max_nodes : int
        Machine capacity.
    df : pd.DataFrame
        Full job dataset.
    output_path : str
        Output directory.
    machine_name : str
        Label for titles / filenames.
    year, month : int or None
        Optional time filter.
    queue_name_df : pd.DataFrame or None
        Valid queues; required when ``do_queue_analysis`` is True.
    do_yearly, do_monthly : bool
        Forwarded to ``generate_queue_analysis``.
    do_full_queue_analysis : bool
        If True, generate per-queue distribution plots.
    do_queue_analysis : bool
        If True, generate queue core-hour breakdown.
    """
    # ---- Keep only the columns needed for plotting --------------------------
    new_df = df.copy()
    new_df = new_df.filter(
        items=[
            "nodes_used",
            "walltime_seconds",
            "queued_timestamp",
            "queue_name",
            "start_timestamp",
            "start_year",
            "start_month",
            "used_core_hours",
        ]
    )
    new_df["queued_timestamp"] = pd.to_datetime(new_df["queued_timestamp"])

    # Maximum walltime in the dataset (used as Y-axis limit on scatter plots)
    max_time = df["walltime_seconds"].max()

    # ---- Build human-readable range labels (e.g. "1-10", "61-600") ----------
    job_dim_df["range_label"] = job_dim_df.apply(
        lambda r: f"{r['min']}-{r['max']}", axis=1
    )
    walltime_df["range_label"] = walltime_df.apply(
        lambda r: f"{r['min']}-{r['max']}", axis=1
    )

    # ---- Compute overall bin counts -----------------------------------------
    job_size_counts, walltime_counts, combined_counts = _make_counts(
        df, job_dim_df, walltime_df
    )

    # ---- Plot the four main distribution charts -----------------------------
    plot_job_size_distribution(
        job_size_counts, output_path, year=year, month=month, machine_name=machine_name
    )
    plot_walltime_distribution(
        walltime_counts, output_path, year=year, month=month, machine_name=machine_name
    )
    plot_heatmap_job_size_vs_walltime(
        job_dim_df,
        walltime_df,
        combined_counts,
        output_path,
        year=year,
        month=month,
        machine_name=machine_name,
    )

    plot_heatmap_distribution_over_time(
        df,
        output_path,
        max_nodes,
        max_time,
        year=year,
        month=month,
        machine_name=machine_name,
    )

    # ---- (Optional) Per-queue distribution plots ----------------------------
    if do_full_queue_analysis:
        print_distribution_analysis_over_queue_names(
            new_df,
            output_path,
            job_dim_df,
            walltime_df,
            max_nodes,
            max_time,
            machine_name=machine_name,
            year=year,
            month=month,
            queue_name_df=queue_name_df,
        )

    # ---- (Optional) Queue core-hour breakdown (stacked bar / area / pie) ----
    if do_queue_analysis:
        # Verify that the required columns exist
        required = {"start_timestamp", "queue_name", "used_core_hours"}
        missing = required - set(df.columns)
        if missing:
            print(f"ERROR: Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        if queue_name_df is not None:
            # Filter to only the queues defined in queue_names.csv
            valid_queues = set(queue_name_df["queue_name"].dropna().unique())
            print(
                f"  Allowed queues from queue_names.csv ({len(valid_queues)}): "
                f"{sorted(valid_queues)}"
            )

            before = len(df)
            df_for_analysis = df[df["queue_name"].isin(valid_queues)].copy()
            print(
                f"  Filtered to {len(df_for_analysis):,} rows "
                f"(dropped {before - len(df_for_analysis):,})."
            )

            if df_for_analysis.empty:
                print("ERROR: No data left after queue filtering.")
                return

            # Build a consistent colour map across all valid queues
            all_queues = df_for_analysis["queue_name"].dropna().unique()
            color_map = _build_color_map(all_queues)

            queue_analysis_dir = os.path.join(output_path, "queue_analysis")
            os.makedirs(queue_analysis_dir, exist_ok=True)

            generate_queue_analysis(
                df_for_analysis,
                queue_analysis_dir,
                machine_name,
                color_map,
                do_yearly=do_yearly,
                do_monthly=do_monthly,
            )

        else:
            raise ValueError(
                "queue_name_df is required for queue analysis to provide color mapping."
            )


# =============================================================================
#  MAIN ANALYSIS WORKFLOW FOR ALL QUEUES
#  This is the top-level function called by main.py's main_workflow().
# =============================================================================
def workOnAllQueues(
    df,
    output_path,
    job_dim_df,
    walltime_df,
    max_nodes,
    queue_name_df,
    machine_name="",
    year=None,
    month=None,
    do_yearly=False,
    do_monthly=False,
    do_full_queue_analysis=False,
    do_queue_analysis=False,
):
    """
    Main analysis workflow for processing all queues in a dataset.

    Generates time distributions (hourly, daily, yearly), distribution analyses
    (job size, walltime), and optionally per-queue breakdowns.

    Parameters
    ----------
    df : pd.DataFrame
        Full job dataset
    output_path : str
        Path for technical analysis output file
    path : str
        Base directory for output files
    input_path : str
        Source file path (for logging)
    reduced_queues : int, default=0
        0=full analysis, 1=lite analysis, 2+=skip queue breakdown
    top_stats : bool, default=False
        If True, generate only top-level stats (for project/user analysis)
    global_save : bool, default=False
        If True, accumulate stats in global_save_values
    global_save_values : dict, optional
        Mutable state for multi-file accumulation

    Returns
    -------
    dict
        Time distribution values {'days': {...}, 'hours': {...}, 'dayofyear': {...}}

    Outputs
    -------
    Creates in path:
    - distribution_over_days.png
    - distribution_over_hours.png
    - distribution_over_years.png
    - (plus distributions_analysis outputs)
    """

    # =============================================================================
    #  MAIN ANALYSIS WORKFLOW
    # =============================================================================

    # ---- Filter data to the requested time window (year / month) ------------
    if year is not None and month is not None:
        df = df[
            (
                (df["start_timestamp"].dt.year == year)
                & (df["start_timestamp"].dt.month == month)
            )
        ].copy()
    elif year is not None:
        df = df[df["start_timestamp"].dt.year == year].copy()

    # ---- Initialise empty time-distribution containers ----------------------
    values = {}
    values["days"] = {
        "Monday": [],
        "Tuesday": [],
        "Wednesday": [],
        "Thursday": [],
        "Friday": [],
        "Saturday": [],
        "Sunday": [],
    }
    values["hours"] = {i: [] for i in range(24)}  # 0..23
    values["dayofyear"] = {i: [] for i in range(1, 367)}  # 1..366

    print("Number of unique queue names:", df["queue_name"].nunique())

    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")

    # ---- Aggregate job counts by time bucket --------------------------------
    _add_distributions(df, values)

    # ---- Print diagnostic info about the dataset ----------------------------
    if "start_timestamp" in df.columns:
        print(
            "Machine worked from",
            df["start_timestamp"].min(),
            "to",
            df["start_timestamp"].max(),
        )
    if "nodes_used" in df.columns:
        print("Max nodes used:", df["nodes_used"].max())
        print("Min nodes used:", df["nodes_used"].min())
    if "walltime_seconds" in df.columns:
        print("Max walltime (seconds):", df["walltime_seconds"].max())
        print("Min walltime (seconds):", df["walltime_seconds"].min())

    # ---- Time-distribution plots (hourly / weekday / yearly) ----------------
    plot_distribution_over_days(values["days"], output_path=output_path)
    plot_distribution_over_hours(values["hours"], output_path=output_path)
    plot_distribution_over_years(values["dayofyear"], output_path=output_path)

    # ---- Distribution analysis (job-size / walltime / heatmap / queue) ------
    distributions_analysis(
        job_dim_df,
        walltime_df,
        max_nodes,
        df,
        output_path=output_path,
        machine_name=machine_name,
        queue_name_df=queue_name_df,
        year=year,
        month=month,
        do_yearly=do_yearly,
        do_monthly=do_monthly,
        do_full_queue_analysis=do_full_queue_analysis,
        do_queue_analysis=do_queue_analysis,
    )

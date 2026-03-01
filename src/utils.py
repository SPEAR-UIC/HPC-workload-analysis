"""
Utility Functions for HPC Workload Analysis.

This module provides shared helper functions used across the entire analysis
pipeline.  It is imported by nearly every other module in the project.

Responsibilities
----------------
- **CLI argument parsing** (`_parse_arguments`): defines and parses every
  command-line flag accepted by ``main.py``.
- **Machine configuration loading** (`_get_machine_config`): reads the three
  CSV config files (``job_dimension.csv``, ``walltime.csv``, ``max_nodes.csv``)
  that describe a machine's node bins, walltime bins, and total node count.
- **Distribution computation** (`_computeJobSizeDistribution`,
  `_computeWalltimeDistribution`, `_computeCombinedDistribution`,
  `_make_counts`): bins jobs into node-size and walltime categories and
  produces count dictionaries consumed by the plotting module.
- **Time-distribution aggregation** (`_add_distributions`): groups job counts
  by hour-of-day, day-of-week, and day-of-year for temporal pattern plots.
- **Data-pivoting helpers** (`_pivot_daily_core_hours`,
  `_pivot_weekly_core_hours`, `_pivot_monthly_core_hours`): reshape DataFrames
  for stacked-bar / stacked-area / pie charts.
- **Colour-map generation** (`_build_color_map`, `generate_distinct_colors`):
  creates perceptually distinct colours via LCH → sRGB conversion so every
  queue gets a unique, reproducible colour.
- **Formatting helpers** (`_fmt`, `_seconds_to_hms`,
  `_collapse_small_categories`, `_get_successful_jobs`): miscellaneous
  formatting and filtering utilities.

Dependencies
------------
- pandas, numpy, argparse, os  (standard / common)
- colour  (``pip install colour-science``) for perceptual colour generation
"""

import argparse
import os

import pandas as pd
import numpy as np
from colour import Lab_to_XYZ, XYZ_to_sRGB
from colour.models import LCHab_to_Lab  # type: ignore[attr-defined]


# =============================================================================
#  COLOUR GENERATION (Glasbey-style farthest-point sampling)
# =============================================================================


def generate_distinct_colors(n_colors=50):
    """
    Generate *n_colors* perceptually distinct RGB colours.

    The algorithm works in the CIE-LCH(ab) colour space:

    1. Build a dense grid of (L, C, h) candidates.
    2. Convert LCH → Lab → XYZ → sRGB and discard out-of-gamut colours.
    3. Greedily pick the colour that is farthest (in Euclidean RGB
       distance) from all colours already in the palette
       ("Glasbey-style" farthest-point sampling).

    Parameters
    ----------
    n_colors : int, default 50
        Number of colours to generate.

    Returns
    -------
    np.ndarray, shape (n_colors, 3)
        RGB values in [0, 1].
    """
    # ---- Step 1: build a wide LCH grid of candidate colours ----
    L_vals = np.linspace(20, 90, 20)  # Lightness
    C_vals = np.linspace(20, 90, 20)  # Chroma (saturation)
    h_vals = np.linspace(0, 360, 60)  # Hue angle

    candidates = np.array(
        [[L, C, h] for L in L_vals for C in C_vals for h in h_vals],
        dtype=np.float64,
    )

    # ---- Step 2: convert to sRGB and keep only valid colours ----
    Lab = LCHab_to_Lab(candidates)
    XYZ = Lab_to_XYZ(Lab)
    RGB = XYZ_to_sRGB(XYZ)

    # Discard colours outside the [0, 1] sRGB gamut
    mask = np.all((RGB >= 0) & (RGB <= 1), axis=1)
    RGB = RGB[mask]

    # ---- Step 3: greedy farthest-point sampling ----
    palette = [RGB[0]]
    remaining = RGB.copy()

    for _ in range(n_colors - 1):
        # For each remaining colour find its min distance to the palette
        distances = np.array(
            [
                np.min(np.linalg.norm(color - np.array(palette), axis=1))
                for color in remaining
            ]
        )
        # Pick the colour that maximises that minimum distance
        idx = np.argmax(distances)
        palette.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)

    return np.array(palette)


# =============================================================================
#  DATA PIVOTING HELPERS
#  Used by single_queue_analysis.py to build stacked-bar / area / pie charts.
# =============================================================================


def _pivot_daily_core_hours(df):
    """
    Pivot *df* so each row is one calendar day and each column is a queue.

    Values are the sum of ``used_core_hours`` for that queue on that day.
    A ``total_runtime_hours`` column with the row-wise total is appended.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``start_timestamp``, ``queue_name``, ``used_core_hours``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (one row per day), columns = queue names + total.
    """
    df = df.copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    df["start_date"] = df["start_timestamp"].dt.date

    grouped = (
        df.groupby(["start_date", "queue_name"])["used_core_hours"].sum().reset_index()
    )
    pivoted = grouped.pivot_table(
        index="start_date",
        columns="queue_name",
        values="used_core_hours",
        fill_value=0,
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted.sort_index(inplace=True)
    pivoted["total_runtime_hours"] = pivoted.sum(axis=1)
    return pivoted


def _pivot_weekly_core_hours(df):
    """
    Pivot *df* so each row is one ISO week (starting Monday) and each
    column is a queue, with values = sum of ``used_core_hours`` that week.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``start_timestamp``, ``queue_name``, ``used_core_hours``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (one row per week), columns = queue names + total.
    """
    df = df.copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    # ISO week start = Monday
    df["start_week"] = (
        df["start_timestamp"].dt.to_period("W").apply(lambda p: p.start_time)
    )
    grouped = (
        df.groupby(["start_week", "queue_name"])["used_core_hours"].sum().reset_index()
    )
    pivoted = grouped.pivot_table(
        index="start_week",
        columns="queue_name",
        values="used_core_hours",
        fill_value=0,
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted.sort_index(inplace=True)
    pivoted["total_runtime_hours"] = pivoted.sum(axis=1)
    return pivoted


def _pivot_monthly_core_hours(df):
    """
    Pivot *df* so each row is one calendar month and each column is a queue,
    with values = sum of ``used_core_hours`` that month.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``start_timestamp``, ``queue_name``, ``used_core_hours``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (one row per month), columns = queue names + total.
    """
    df = df.copy()
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
    df["start_month"] = df["start_timestamp"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        df.groupby(["start_month", "queue_name"])["used_core_hours"].sum().reset_index()
    )
    pivoted = grouped.pivot_table(
        index="start_month",
        columns="queue_name",
        values="used_core_hours",
        fill_value=0,
    )
    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted.sort_index(inplace=True)
    pivoted["total_runtime_hours"] = pivoted.sum(axis=1)
    return pivoted


# =============================================================================
#  COLOUR MAP BUILDER
# =============================================================================


def _build_color_map(queue_names):
    """
    Build a reproducible {queue_name: (r, g, b)} colour map.

    Sorts queue names alphabetically, generates one distinct colour per
    queue via ``generate_distinct_colors``, and always reserves a grey
    ``(0.7, 0.7, 0.7)`` entry for the catch-all "Other" category.

    Parameters
    ----------
    queue_names : array-like of str
        Unique queue names present in the dataset.

    Returns
    -------
    dict
        ``{queue_name: (r, g, b)}`` with values in [0, 1].
    """
    names = sorted(set(queue_names))
    if "Other" not in names:
        names.append("Other")
    palette = generate_distinct_colors(n_colors=len(names))
    cmap = {n: tuple(palette[i]) for i, n in enumerate(names)}
    cmap["Other"] = (0.7, 0.7, 0.7)
    return cmap


# =============================================================================
#  FORMATTING HELPERS
# =============================================================================


def _fmt(val, unit=""):
    """Format a numeric value: commas for ≥1, four decimals for <1, optional unit suffix."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1:
        return f"{val:,.2f}{unit}"
    return f"{val:.4f}{unit}"


def _seconds_to_hms(s):
    """Convert seconds to 'Xh Ym Zs' string."""
    if pd.isna(s):
        return "N/A"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"


def _get_successful_jobs(df):
    """
    Return a copy of *df* containing only successful jobs.

    Success is defined as ``exit_code == 0`` (normal exit) or
    ``exit_code == -29`` (walltime-terminated but still considered valid
    on some schedulers).

    Raises
    ------
    ValueError
        If *df* does not contain an ``exit_code`` column.
    """
    if "exit_code" in df.columns:
        return df[(df["exit_code"] == 0) | (df["exit_code"] == -29)].copy()
    raise ValueError(
        "DataFrame must contain 'exit_code' column to filter successful jobs."
    )


def _collapse_small_categories(series, threshold_percent=1.0):
    """
    Group small categories into "Other" for cleaner pie charts.

    Categories contributing less than threshold_percent of the total
    are combined into a single "Other" category.

    Parameters
    ----------
    series : pd.Series
        Category counts/values
    threshold_percent : float, default=1.0
        Minimum percentage to show separately

    Returns
    -------
    pd.Series
        Series with small categories collapsed into "Other"

    Examples
    --------
    >>> data = pd.Series({'A': 50, 'B': 40, 'C': 5, 'D': 3, 'E': 2})
    >>> collapse_small_categories(data, threshold_percent=5.0)
    A       50
    B       40
    Other   10
    dtype: int64
    """
    total = series.sum()
    if total == 0:
        return series

    above_threshold = series[series / total * 100 >= threshold_percent].copy()
    other_total = series[series / total * 100 < threshold_percent].sum()

    if other_total > 0:
        above_threshold["Other"] = other_total

    return above_threshold


# =============================================================================
#  JOB SIZE DISTRIBUTION COMPUTATION
# =============================================================================
def _computeJobSizeDistribution(df, job_dim_df, job_size_counts):
    """
    Count jobs falling into each node size bin.

    Populates job_size_counts dict in-place with counts for each range.

    Parameters
    ----------
    df : pd.DataFrame
        Job dataset with 'nodes_used' column
    job_dim_df : pd.DataFrame
        Bin definitions with 'min', 'max', 'range_label' columns
    job_size_counts : dict
        Output dictionary to populate {range_label: count}

    Examples
    --------
    >>> counts = {}
    >>> computeJobSizeDistribution(df, bins_df, counts)
    >>> counts
    {'1-10': 500, '11-100': 1200, ...}
    """
    for _, row in job_dim_df.iterrows():
        min_nodes = row["min"]
        if row["max"] == "infinity":
            max_nodes = float("inf")
        else:
            max_nodes = row["max"]
        count = df[
            (df["nodes_used"] >= min_nodes) & (df["nodes_used"] <= max_nodes)
        ].shape[0]
        job_size_counts[row["range_label"]] = count


# =============================================================================
#  WALLTIME DISTRIBUTION COMPUTATION
# =============================================================================
def _computeWalltimeDistribution(df, walltime_df, walltime_counts):
    """
    Count jobs falling into each walltime bin.

    Populates walltime_counts dict in-place with counts for each range.

    Parameters
    ----------
    df : pd.DataFrame
        Job dataset with 'walltime_seconds' column
    walltime_df : pd.DataFrame
        Bin definitions with 'min', 'max', 'range_label' columns
    walltime_counts : dict
        Output dictionary to populate {range_label: count}

    Examples
    --------
    >>> counts = {}
    >>> computeWalltimeDistribution(df, walltime_bins, counts)
    >>> counts
    {'0-3600': 800, '3601-7200': 1500, ...}
    """
    for _, row in walltime_df.iterrows():
        min_time = row["min"]
        if row["max"] == "infinity":
            max_time = float("inf")
        else:
            max_time = row["max"]
        count = df[
            (df["walltime_seconds"] >= min_time) & (df["walltime_seconds"] <= max_time)
        ].shape[0]
        walltime_counts[row["range_label"]] = count


# =============================================================================
#  COMBINED 2D DISTRIBUTION (Job Size x Walltime)
# =============================================================================
def _computeCombinedDistribution(df, job_dim_df, walltime_df, combined_counts):
    """
    Compute 2D distribution grid of jobs by node size AND walltime.

    Creates a count for each combination of node range and walltime range.
    Useful for heatmap visualizations showing job characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Job dataset with 'nodes_used' and 'walltime_seconds' columns
    job_dim_df : pd.DataFrame
        Node bin definitions
    walltime_df : pd.DataFrame
        Walltime bin definitions
    combined_counts : dict
        Output dictionary to populate {"{node_range}_{walltime_range}": count}

    Examples
    --------
    >>> counts = {}
    >>> computeCombinedDistribution(df, node_bins, time_bins, counts)
    >>> counts
    {'1-10_0-3600': 150, '1-10_3601-7200': 200, ...}
    """
    for _, row in job_dim_df.iterrows():
        for _, wrow in walltime_df.iterrows():
            name = f"{row['range_label']}_{wrow['range_label']}"
            if row["max"] == "infinity":
                max_nodes = float("inf")
            else:
                max_nodes = row["max"]
            min_nodes = row["min"]
            if wrow["max"] == "infinity":
                max_time = float("inf")
            else:
                max_time = wrow["max"]
            min_time = wrow["min"]
            count = df[
                (df["nodes_used"] >= min_nodes)
                & (df["nodes_used"] <= max_nodes)
                & (df["walltime_seconds"] >= min_time)
                & (df["walltime_seconds"] <= max_time)
            ].shape[0]
            combined_counts[name] = count


# =============================================================================
#  COMPUTE ALL DISTRIBUTION COUNTS
# =============================================================================
def _make_counts(df, job_dim_df, walltime_df, debug=False):
    """
    Compute all three distribution types: job size, walltime, and combined.

    Convenience function that initializes count dictionaries and calls
    all three distribution computation functions.

    Parameters
    ----------
    df : pd.DataFrame
        Job dataset
    job_dim_df : pd.DataFrame
        Node bin definitions with 'range_label' column
    walltime_df : pd.DataFrame
        Walltime bin definitions with 'range_label' column
    debug : bool, default=False
        If True, print computed counts

    Returns
    -------
    tuple
        (job_size_counts, walltime_counts, combined_counts)
        Each is a dict {label: count}

    Examples
    --------
    >>> job_counts, wall_counts, combined = make_counts(df, job_bins, wall_bins)
    """
    job_size_counts = {r["range_label"]: 0 for _, r in job_dim_df.iterrows()}
    walltime_counts = {r["range_label"]: 0 for _, r in walltime_df.iterrows()}
    combined_counts = {}

    # Compute job size distribution
    _computeJobSizeDistribution(df, job_dim_df, job_size_counts)

    # Compute walltime distribution
    _computeWalltimeDistribution(df, walltime_df, walltime_counts)

    # Compute combined distribution
    _computeCombinedDistribution(df, job_dim_df, walltime_df, combined_counts)

    return job_size_counts, walltime_counts, combined_counts


# =============================================================================
#  TIME DISTRIBUTION AGGREGATION
# =============================================================================
def _add_distributions(df, values):
    """
    Populate time distribution arrays from job data.

    Computes job counts grouped by time period (hour, weekday, or day of year)
    and appends the counts to the values dictionary. This enables statistical
    analysis of job submission patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Job dataset with 'start_timestamp' column (must be datetime)
    values : dict
        Output dictionary with structure:
        - {'hours': {0: [], 1: [], ..., 23: []}}
        - {'days': {'Monday': [], ..., 'Sunday': []}}
        - {'dayofyear': {1: [], ..., 366: []}}

    Notes
    -----
    For 'hours': counts how many jobs started at each hour, for each day
    For 'days': counts how many jobs started on each weekday, for each week
    For 'dayofyear': counts how many jobs started on each day, for each year
    """
    new_df = df.copy()
    new_df["start_timestamp"] = pd.to_datetime(
        new_df["start_timestamp"], errors="coerce"
    )
    new_df = new_df.dropna(subset=["start_timestamp"])
    new_df = new_df.sort_values("start_timestamp")
    new_df["date"] = new_df["start_timestamp"].dt.date
    new_df["hour"] = new_df["start_timestamp"].dt.hour
    new_df["week"] = new_df["start_timestamp"].dt.isocalendar().week
    new_df["weekday"] = new_df["start_timestamp"].dt.day_name()
    new_df["year"] = new_df["start_timestamp"].dt.year
    new_df["dayofyear"] = new_df["start_timestamp"].dt.dayofyear

    counts_hours = new_df.groupby(["date", "hour"]).size().unstack(fill_value=0)

    for hour in range(24):
        values["hours"][hour].extend(counts_hours.get(hour, pd.Series()).tolist())

    counts_weekdays = new_df.groupby(["week", "weekday"]).size().unstack(fill_value=0)

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    for day in weekday_order:
        values["days"][day].extend(counts_weekdays.get(day, pd.Series()).tolist())

    counts_dayofyear = (
        new_df.groupby(["year", "dayofyear"]).size().unstack(fill_value=0)
    )

    for day in range(1, 367):
        values["dayofyear"][day].extend(counts_dayofyear.get(day, pd.Series()).tolist())


# =============================================================================
#  MACHINE CONFIGURATION LOADER
# =============================================================================


def _get_machine_config(input_path):
    """
    Load the three machine-specific configuration files.

    Expected CSV files in the **same directory** as *input_path*:

    * ``job_dimension.csv`` — node-count bin boundaries
      (columns: ``name``, ``min``, ``max``)
    * ``walltime.csv`` — walltime bin boundaries
      (columns: ``name``, ``min``, ``max``; ``max`` may be ``"infinity"``)
    * ``max_nodes.csv`` — single-value file with the total node count
      (column: ``max_nodes``)

    A ``range_label`` column (e.g. ``"1-10"``) is added to both
    ``job_dim_df`` and ``walltime_df`` after loading.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file (e.g. ``data/Polaris/jobs_preprocessed.csv.gz``).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, int)
        ``(job_dim_df, walltime_df, max_nodes)``

    Raises
    ------
    FileNotFoundError
        If the directory or any of the three files is missing.
    """
    base_dir = os.path.dirname(input_path)
    # check path exists
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    if not os.path.isfile(os.path.join(base_dir, "job_dimension.csv")):
        raise FileNotFoundError(f"Missing job_dimension.csv in {base_dir}")
    if not os.path.isfile(os.path.join(base_dir, "walltime.csv")):
        raise FileNotFoundError(f"Missing walltime.csv in {base_dir}")
    if not os.path.isfile(os.path.join(base_dir, "max_nodes.csv")):
        raise FileNotFoundError(f"Missing max_nodes.csv in {base_dir}")

    job_dim_df = pd.read_csv(os.path.join(base_dir, "job_dimension.csv"))
    walltime_df = pd.read_csv(os.path.join(base_dir, "walltime.csv"))
    max_nodes = pd.read_csv(os.path.join(base_dir, "max_nodes.csv"))["max_nodes"].iloc[
        0
    ]

    job_dim_df["range_label"] = job_dim_df.apply(
        lambda r: f"{r['min']}-{r['max']}", axis=1
    )
    walltime_df["range_label"] = walltime_df.apply(
        lambda r: f"{r['min']}-{r['max']}", axis=1
    )
    print(f"✅ Loaded machine config from {base_dir}:")
    return job_dim_df, walltime_df, max_nodes


# =============================================================================
#  COMMAND-LINE ARGUMENT PARSER
# =============================================================================


def _parse_arguments():
    """
    Parse CLI arguments for ``main.py``.

    Returns
    -------
    argparse.Namespace
        Attributes:

        * ``path`` (str)              – path to a ``.csv.gz`` input file.
        * ``machine_name`` (str)      – machine label (e.g. ``"Polaris"``).
        * ``queue_analysis`` (bool)   – run per-queue core-hour breakdown.
        * ``machine_utilization`` (bool) – compute and plot node utilization.
        * ``yearly`` (bool)           – split analysis by calendar year.
        * ``monthly`` (bool)          – split analysis by calendar month.
        * ``full_queue_analysis`` (bool) – generate per-queue distribution
          plots (job-size, walltime, heatmap) for **every** queue.
    """
    # -------------------------------------------------------------
    #  ARGPARSE SETUP
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="HPC Job Log Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    Single file:
        python infoFromFile.py path/to/jobs_preprocessed.csv.gz

    All files in directory:
        python infoFromFile.py path/to/machine/ --all

    With global aggregation:
        python infoFromFile.py path/to/machine/ --all --save

    Skip user/project analysis and yearly report:
        python infoFromFile.py path/to/file.csv.gz --no-top-projects --no-top-users --no-yearly

    Run only basic workflow (no extras):
    python infoFromFile.py path/to/file.csv.gz --no-top-projects --no-top-users \\
        --no-queue-analysis --no-machine-utilization --no-yearly --no-active-queues
        """,
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to input file (jobs_preprocessed.csv.gz) or directory containing multiple files",
    )
    parser.add_argument(
        "--machine-name",
        type=str,
        required=True,
        help="Name of the machine (e.g. Polaris, Aurora). Used in plot titles and filenames.",
    )
    parser.add_argument(
        "--queue-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run queue utilization analysis (reduced_queues=False when on, True when off) (default: on)",
    )
    parser.add_argument(
        "--machine-utilization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run machine utilization analysis (default: off)",
    )
    parser.add_argument(
        "--yearly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run per-year breakdown analysis (default: off)",
    )
    parser.add_argument(
        "--monthly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run monthly breakdown analysis (default: off, not implemented yet)",
    )
    parser.add_argument(
        "--full-queue-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run queue utilization analysis (reduced_queues=False when on, True when off) (default: on)",
    )

    return parser.parse_args()

"""
Machine / System Utilization Analysis.

This module computes **node-level utilization** over time using an
event-based approach and optionally detects intervals where the
scheduler exceeded the machine's physical node count ("over-capacity").

Algorithm
---------
1. For every job, emit two events:
   * ``+nodes_used`` at ``start_timestamp``
   * ``-nodes_used`` at ``end_timestamp``
2. Sort all events chronologically and take a cumulative sum to get
   the exact number of nodes in use at every event boundary.
3. Compute **time-weighted** mean and standard deviation.
4. Walk the event series to identify contiguous intervals where
   ``nodes_used > max_nodes`` (over-capacity).
5. Resample to a fixed resolution (default ``15min``) for plotting.

Outputs
-------
* ``<machine>_utilization_over_time*.png``
  (via ``plotting.plot_machine_utilization``)
* ``utilization_stats.txt`` (optional, when ``print_stats=True``)
* ``over_100_intervals_with_jobs.txt`` (if over-capacity detected
  and ``print_stats=True``)

Public API
----------
``machine_utilization(df, output_path, max_nodes, ...)``
    Main entry point – called from ``main.main_workflow``.

``jobs_running_at(df, ts)``
    Helper that returns all jobs overlapping timestamp *ts*.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from plotting import plot_machine_utilization

# =============================================================================
#  CONFIGURATION
# =============================================================================
RESOLUTION = "15min"  # time-series resampling resolution for plotting
DEBUG = True  # if True, print extra diagnostic messages
# Minimum duration (in seconds) of an over-capacity interval to report.
# Shorter spikes are treated as scheduling artefacts and ignored.
MIN_OVER_CAPACITY_DURATION_SEC = 60


# =============================================================================
#  HELPER: find all jobs running at a given timestamp
# =============================================================================


def jobs_running_at(df, ts):
    """
    Return jobs whose execution interval contains *ts*.

    A job is considered running if ``start_timestamp <= ts < end_timestamp``.
    Handles timezone-aware and timezone-naive timestamps transparently.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``job_name``, ``nodes_used``, ``start_timestamp``,
        ``end_timestamp``.
    ts : pd.Timestamp
        Point in time to query.

    Returns
    -------
    pd.DataFrame
        Subset of *df* with columns ``['job_name', 'nodes_used',
        'start_timestamp', 'end_timestamp']``.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    mask = (df["start_timestamp"] <= ts) & (ts < df["end_timestamp"])
    return df.loc[mask, ["job_name", "nodes_used", "start_timestamp", "end_timestamp"]]


# =============================================================================
#  MAIN UTILIZATION FUNCTION
# =============================================================================


def machine_utilization(
    df_all,
    output_path,
    max_nodes=0,
    print_stats=False,
    print_plot=True,
    machine_name="",
    year=None,
    month=None,
):
    """
    Compute and optionally plot machine utilization over time.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full job dataset (must contain ``start_timestamp``,
        ``end_timestamp``, ``nodes_used``).
    output_path : str
        Directory for output files.
    max_nodes : int
        Physical machine capacity (node count).  If <= 0 the maximum
        ``nodes_used`` in the data is used instead.
    print_stats : bool, default False
        Write ``utilization_stats.txt`` and over-capacity reports.
    print_plot : bool, default True
        Generate the utilization line plot.
    machine_name : str
        Label for plot titles.
    year, month : int or None
        Included in plot titles / filenames when set.

    Returns
    -------
    tuple of (int, int, float, float)
        ``(min_nodes_used, max_nodes_used, mean_nodes, std_nodes)``.
    """
    df = df_all.copy()

    # ---- Pre-processing -----------------------------------------------------
    # Fall back to the max observed value if machine capacity not provided
    if max_nodes <= 0:
        max_nodes = df.nodes_used.max()

    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], utc=True)
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], utc=True)

    # Drop invalid jobs (end before they started)
    df = df[df["end_timestamp"] > df["start_timestamp"]]

    # ---- Build event-based utilization series --------------------------------
    # Each job produces two events: +N at start, -N at end.
    start_times = df["start_timestamp"].values.astype("datetime64[ns]")
    end_times = df["end_timestamp"].values.astype("datetime64[ns]")

    event_times = np.concatenate([start_times, end_times])  # all events
    event_deltas = np.concatenate(
        [df["nodes_used"].values, -df["nodes_used"].values]  # +N at start
    )  # -N at end

    # Sort by time
    order = np.argsort(event_times)
    event_times = event_times[order]
    event_deltas = event_deltas[order]

    # Group concurrent events and compute the cumulative node count
    event_df = pd.DataFrame({"delta": event_deltas}, index=pd.to_datetime(event_times))
    event_df = event_df.groupby(level=0).sum()

    utilization_events = event_df.cumsum()  # running total of nodes in use
    utilization_events = utilization_events.rename(columns={"delta": "nodes_used"})

    # ---- Time-weighted statistics -------------------------------------------
    # Weight each node-count value by the duration until the next event.
    event_times = utilization_events.index.values
    event_values = utilization_events["nodes_used"].values

    durations = (event_times[1:] - event_times[:-1]) / np.timedelta64(1, "s")  # seconds
    values = np.asarray(
        event_values[:-1], dtype=np.float64
    )  # node count during each interval

    total_time = durations.sum()  # total observation window in seconds

    # Weighted mean
    mean_nodes = np.sum(values * durations) / total_time
    # Weighted variance and standard deviation
    variance = np.sum(((values - mean_nodes) ** 2) * durations) / total_time
    std_nodes = np.sqrt(variance)

    # ---- (Optional) Write utilization statistics to file --------------------
    if print_stats:
        with open(os.path.join(output_path, "utilization_stats.txt"), "w") as f:
            f.write(f"Max nodes available: {max_nodes}\n")
            f.write(
                f"Max utilization: {utilization_events['nodes_used'].max()} nodes "
                f"({utilization_events['nodes_used'].max() / max_nodes * 100:.2f} %)\n"
            )
            f.write(
                f"Min utilization: {utilization_events['nodes_used'].min()} nodes "
                f"({utilization_events['nodes_used'].min() / max_nodes * 100:.2f} %)\n"
            )
            f.write(
                f"Mean utilization (time-weighted): {mean_nodes:.2f} nodes "
                f"({mean_nodes / max_nodes * 100:.2f} %)\n"
            )
            f.write(
                f"Std  utilization (time-weighted): {std_nodes:.2f} nodes "
                f"({std_nodes / max_nodes * 100:.2f} %)\n"
            )

    # ---- Detect over-capacity intervals (event-based) -----------------------
    # An over-capacity interval starts when nodes_used first exceeds
    # max_nodes and ends when it drops back to or below max_nodes.
    utilization_events = utilization_events.sort_index()
    over_mask = utilization_events["nodes_used"] > max_nodes

    starts = over_mask & ~over_mask.shift(fill_value=False)
    ends = ~over_mask & over_mask.shift(fill_value=False)

    start_times = utilization_events.index[starts]
    end_times = utilization_events.index[ends]

    # Close open interval if needed
    if len(end_times) < len(start_times):
        end_times = end_times.append(pd.Index([utilization_events.index[-1]]))

    over_capacity_periods = []

    for start, end in zip(start_times, end_times):
        duration_sec = (end - start).total_seconds()

        # ⬇️ Skip short spikes
        if duration_sec < MIN_OVER_CAPACITY_DURATION_SEC:
            continue

        interval = utilization_events.loc[start:end]

        jobs_start = jobs_running_at(df, start)
        jobs_end = jobs_running_at(df, end)

        over_capacity_periods.append(
            {
                "start_timestamp": start,
                "end_timestamp": end,
                "duration_seconds": duration_sec,
                "max_nodes_used": int(interval["nodes_used"].max()),
                "mean_nodes_used": interval["nodes_used"].mean(),
                "jobs_at_start": jobs_start,
                "jobs_at_end": jobs_end,
            }
        )
    over_capacity_df = pd.DataFrame(over_capacity_periods)

    # ---- (Optional) Write over-capacity report ------------------------------
    if print_stats:
        if not over_capacity_df.empty:
            report_path = os.path.join(output_path, "over_100_intervals_with_jobs.txt")

            with open(report_path, "w") as f:
                for idx, (_, r) in enumerate(over_capacity_df.iterrows(), start=1):
                    f.write("=" * 80 + "\n")
                    f.write(
                        f"Over-capacity interval {idx}\n"
                        f"Start: {r.start_timestamp}\n"
                        f"End  : {r.end_timestamp}\n"
                        f"Duration: {r.duration_seconds/60:.2f} min\n"
                        f"Max nodes used: {r.max_nodes_used}\n\n"
                    )

                    f.write("Jobs running at START:\n")
                    for _, j in r.jobs_at_start.iterrows():
                        f.write(
                            f"  - {j.job_name}: {j.nodes_used} nodes "
                            f"(start: {j.start_timestamp}, end: {j.end_timestamp})\n"
                        )

                    f.write("\nJobs running at END:\n")
                    for _, j in r.jobs_at_end.iterrows():
                        f.write(
                            f"  - {j.job_name}: {j.nodes_used} nodes "
                            f"(start: {j.start_timestamp}, end: {j.end_timestamp})\n"
                        )
                    f.write("\n")

            if DEBUG:
                print(
                    f"⚠ Found {len(over_capacity_df)} over-capacity intervals\n"
                    f"📄 Report written to {report_path}"
                )
        else:
            print("✅ No real over-capacity utilization detected.")

    # ---- Resample for plotting (fixed time grid) ----------------------------
    # Forward-fill the event series onto a regular grid so the plot is smooth.
    utilization_plot = (
        utilization_events["nodes_used"].resample(RESOLUTION).ffill().fillna(0)
    )
    utilization_pct = utilization_plot / max_nodes * 100  # convert to %

    if print_plot:
        plot_machine_utilization(
            machine_name,
            utilization_pct,
            output_path=output_path,
            mean_nodes_pct=mean_nodes / max_nodes * 100,
            std_nodes_pct=std_nodes / max_nodes * 100,
            year=year,
            month=month,
        )

    min_used_nodes = utilization_events["nodes_used"].min()
    max_used_nodes = utilization_events["nodes_used"].max()

    return min_used_nodes, max_used_nodes, mean_nodes, std_nodes

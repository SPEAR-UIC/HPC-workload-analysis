"""
Visualisation Module for HPC Workload Analysis.

This module contains **every** plotting function used by the pipeline.
All plots are written to disk as PNG files – nothing is displayed
interactively (``matplotlib.use("Agg")``).

Plot catalogue
--------------
Temporal distributions (called by ``orchestrator.workOnAllQueues``):
    * ``plot_distribution_over_hours``  – bar chart by hour of day.
    * ``plot_distribution_over_days``   – bar chart by weekday.
    * ``plot_distribution_over_years``  – smoothed line chart by day-of-year.

Job-size & walltime distributions (called by ``orchestrator.distributions_analysis``):
    * ``plot_job_size_distribution``       – bar chart of node-bin counts.
    * ``plot_walltime_distribution``       – bar chart of walltime-bin counts.
    * ``plot_heatmap_job_size_vs_walltime``– 2-D heatmap (log colour scale).
    * ``plot_heatmap_distribution_over_time`` – temporal scatter plot
      (X = date, Y = walltime, colour / size = nodes).

Machine utilization (called by ``system_utilization.machine_utilization``):
    * ``plot_machine_utilization`` – time-series with rolling mean, LOWESS
      trend, mean±std band, and 50 % capacity line.

Queue core-hour breakdown (called by ``single_queue_analysis``):
    * ``plot_stacked_bar_queue_percentages``  – stacked bar chart.
    * ``plot_stacked_area_queue_percentages`` – cumulative stacked area.
    * ``plot_pie_chart_queue_percentages``    – pie chart.

Dependencies
------------
matplotlib, seaborn, numpy, pandas, statsmodels (for LOWESS).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LogNorm
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils import _collapse_small_categories

# Force a non-interactive backend – all output goes to files.
matplotlib.use("Agg")

# Constants for the temporal scatter-plot marker sizing.
MIN_SIZE = 5  # minimum marker area (points²)
MAX_SIZE = 500  # maximum marker area (points²)


# =============================================================================
#  INTERNAL HELPER FUNCTIONS
# =============================================================================


def mean_std_percent(array_dict: dict) -> tuple[pd.Series, pd.Series]:
    """
    Convert ``{bin_label: [count_period1, count_period2, ...]}`` to
    (mean %, std %) across periods.

    Each row (= one period) is normalised to 100 % first so that periods
    with different total job counts are comparable.

    Returns
    -------
    tuple of (pd.Series, pd.Series)
        ``(mean_pct, std_pct)`` indexed by bin label.
    """

    # Build a DataFrame: columns = bins, rows = periods
    df = pd.DataFrame(
        {k: pd.Series(v, dtype="float64") for k, v in array_dict.items()}
    ).fillna(0.0)

    if df.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    # ---- Row-wise normalisation to percentages ------------------------------
    row_sum = df.sum(axis=1)
    nonzero_mask = row_sum != 0  # avoid division by zero
    df.loc[nonzero_mask] = (
        df.loc[nonzero_mask].div(row_sum[nonzero_mask], axis=0) * 100.0
    )
    # Rows that sum to zero stay at zero.

    return df.mean(), df.std()


def _scale_marker_size(series, min_nodes_used=1, max_nodes_used=1):
    """
    Linearly map node counts to scatter-plot marker areas.

    The smallest job gets ``MIN_SIZE`` and the machine-capacity job gets
    ``MAX_SIZE``.  If all values are equal the minimum size is returned.
    """
    if max_nodes_used <= min_nodes_used:
        return pd.Series([MIN_SIZE] * len(series))

    return MIN_SIZE + (series - min_nodes_used) / (max_nodes_used - min_nodes_used) * (
        MAX_SIZE - MIN_SIZE
    )


# =============================================================================
#  TEMPORAL DISTRIBUTION PLOTS  (hour / weekday / day-of-year)
# =============================================================================


def plot_distribution_over_hours(array, output_path):
    """
    Bar chart of mean job percentage by **hour of day** (0-23).

    Error bars show the standard deviation across all dates in the
    dataset.

    Saves ``distribution_over_hours.png`` into *output_path*.
    """
    mean, std = mean_std_percent(array)

    plt.figure(figsize=(10, 6))
    plt.bar(
        mean.index,
        np.asarray(mean.values, dtype=float),
        yerr=np.asarray(std.values, dtype=float),
        capsize=4,
    )
    plt.xlabel("Hour of the Day")
    plt.ylabel("Percentage of Jobs")
    plt.title("Distribution of Jobs over Hours (mean ± std)")
    plt.xticks(range(24))

    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, "distribution_over_hours.png")
    plt.savefig(out)
    plt.close()
    print("✅ Done plotting distribution over hours on:", out)


def plot_distribution_over_days(array, output_path):
    """
    Bar chart of mean job percentage by **weekday** (Mon–Sun).

    Saves ``distribution_over_days.png`` into *output_path*.
    """
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    mean, std = mean_std_percent(array)
    mean = mean.reindex(order)
    std = std.reindex(order)

    plt.figure(figsize=(10, 6))
    plt.bar(
        mean.index,
        np.asarray(mean.values, dtype=float),
        yerr=np.asarray(std.values, dtype=float),
        capsize=4,
    )
    plt.xlabel("Day of the Week")
    plt.ylabel("Percentage of Jobs")
    plt.title("Distribution of Jobs over Days (mean ± std)")

    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, "distribution_over_days.png")
    plt.savefig(out)
    plt.close()
    print("✅ Done plotting distribution over days on:", out)


def plot_distribution_over_years(array, output_path, window=7):
    """
    Smoothed line chart of mean job percentage by **day of year** (1-366).

    A rolling mean (with ± std band) is applied with *window* days
    for visual smoothing.

    Saves ``distribution_over_years.png`` into *output_path*.
    """
    mean, std = mean_std_percent(array)
    mean = mean.reindex(range(1, 367), fill_value=0)
    std = std.reindex(range(1, 367), fill_value=0)

    # Apply smoothing
    mean_s = mean.rolling(window, center=True, min_periods=1).mean()
    std_s = std.rolling(window, center=True, min_periods=1).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(mean_s.index, np.asarray(mean_s.values, dtype=float), linewidth=1.5)
    plt.fill_between(mean_s.index, mean_s - std_s, mean_s + std_s, alpha=0.3)

    # Month labels on x-axis
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_positions = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    plt.xticks(month_positions, month_names)
    plt.xlim(1, 366)
    plt.xlabel("Day of the Year")
    plt.ylabel("Percentage of Jobs")
    plt.title("Distribution of Jobs over the Year (rolling mean ± std)")

    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, "distribution_over_years.png")
    plt.savefig(out, bbox_inches="tight", dpi=450)
    plt.close()
    print("✅ Smoothed yearly distribution saved to:", out)


# =============================================================================
#  JOB SIZE & WALLTIME DISTRIBUTION PLOTS
# =============================================================================


def plot_job_size_distribution(
    job_size_counts, output_path, year=None, month=None, machine_name=""
):
    """
    Bar chart of job counts (as percentages) per **node-count bin**.

    Each bar is annotated with its exact percentage value.

    Saves ``job_size_distribution.png`` into *output_path*.
    """
    plt.figure(figsize=(10, 6))
    job_size_series = pd.Series(job_size_counts)
    job_size_series = job_size_series / job_size_series.sum() * 100

    job_size_series.plot(kind="bar", color="teal")
    plt.xlabel("Nodes used")
    plt.ylabel("Percentage of Jobs")

    for i, v in enumerate(job_size_series):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha="center")

    os.makedirs(output_path, exist_ok=True)
    if year and month:
        plt.title(
            f"Distribution of Jobs over Node Sizes ({machine_name} {year}-{month:02d})",
            fontsize=16,
            fontweight="bold",
        )
    elif year:
        plt.title(
            f"Distribution of Jobs over Node Sizes ({machine_name} {year})",
            fontsize=16,
            fontweight="bold",
        )
    else:
        plt.title(
            f"Distribution of Jobs over Node Sizes ({machine_name})",
            fontsize=16,
            fontweight="bold",
        )
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, f"job_size_distribution.png"), dpi=300)
    plt.close()
    print(
        "✅ Done plotting job size distribution on:",
        os.path.join(output_path, f"job_size_distribution.png"),
    )


def plot_walltime_distribution(
    walltime_counts, output_path, year=None, month=None, machine_name=""
):
    """
    Bar chart of job counts (as percentages) per **walltime bin**.

    Bins are colour-coded and a legend maps position labels
    (``pos1``, ``pos2``, …) to the actual walltime ranges.

    Saves ``distribution_over_walltime.png`` into *output_path*.
    """
    plt.figure(figsize=(12, 6))

    if isinstance(walltime_counts, dict):
        counts: pd.Series = pd.Series(walltime_counts, dtype=float)
        counts = counts / counts.sum()
    else:
        counts = pd.Series(walltime_counts).value_counts(normalize=True).sort_index()

    # Sort by numeric min of each "A-B" range
    counts.index = pd.Categorical(
        counts.index,
        categories=sorted(counts.index, key=lambda x: int(x.split("-")[0])),
        ordered=True,
    )
    counts = counts.sort_index()

    # Create position labels and colors
    positions = [f"pos{i+1}" for i in range(len(counts))]
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i % 20) for i in range(len(counts))]

    ax = counts.plot(kind="bar", color=colors)
    ax.set_xticklabels(positions, rotation=45)

    plt.xlabel("Walltime Position")
    plt.ylabel("Percentage of Jobs")

    # Build legend
    handles = [
        mpatches.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(counts))
    ]
    labels = [f"{positions[i]}: {counts.index[i]}" for i in range(len(counts))]
    plt.legend(
        handles,
        labels,
        title="Walltime ranges",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    os.makedirs(output_path, exist_ok=True)

    if year and month:
        plt.title(
            f"Distribution of Jobs over Walltime ({machine_name} {year}-{month:02d})",
            fontsize=16,
            fontweight="bold",
        )
    elif year:
        plt.title(
            f"Distribution of Jobs over Walltime ({machine_name} {year})",
            fontsize=16,
            fontweight="bold",
        )
    else:
        plt.title(
            f"Distribution of Jobs over Walltime ({machine_name})",
            fontsize=16,
            fontweight="bold",
        )
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, f"distribution_over_walltime.png"), dpi=300)
    plt.close()
    print(
        "✅ Done plotting distribution over walltime on:",
        os.path.join(output_path, f"distribution_over_walltime.png"),
    )


def format_seconds_range(min_sec, max_sec):
    """
    Convert a (min_sec, max_sec) pair to a compact human-readable label.

    Examples: ``"5m–30m"``, ``">12h"``.
    """

    def fmt(sec):
        if sec >= 3600:
            return f"{int(sec/3600)}h"
        elif sec >= 60:
            return f"{int(sec/60)}m"
        else:
            return f"{int(sec)}s"

    if max_sec == float("inf"):
        return f">{fmt(min_sec)}"

    return f"{fmt(min_sec)}–{fmt(max_sec)}"


def plot_heatmap_job_size_vs_walltime(
    job_dim_df,
    walltime_df,
    combined_counts,
    output_path,
    year=None,
    month=None,
    machine_name="",
):
    """
    2-D heatmap of job counts binned by node-count **and** walltime.

    Uses a ``LogNorm`` colour scale to handle wide count ranges.
    Cells with zero jobs are left blank (NaN).

    Saves ``<machine>_job_size_vs_walltime_heatmap.png`` into
    *output_path*.
    """

    # ---- Build the count matrix (rows = node bins, cols = walltime bins) ----
    heatmap_data = pd.DataFrame(
        0,
        index=job_dim_df["range_label"],
        columns=walltime_df["range_label"],
    )

    for _, row in job_dim_df.iterrows():
        for _, wrow in walltime_df.iterrows():
            name = f"{row['range_label']}_{wrow['range_label']}"
            heatmap_data.at[row["range_label"], wrow["range_label"]] = (
                combined_counts.get(name, 0)
            )

    # Replace zeros with NaN so LogNorm doesn't choke on log(0)
    heatmap_data = heatmap_data.replace(0, np.nan)

    if heatmap_data.isna().all().all():
        print("⚠️ Skipping heatmap: all values are zero")
        return

    # ---- Convert walltime column headers to human-readable labels -----------
    formatted_columns = []
    for _, row in walltime_df.iterrows():
        min_sec = row["min"]
        max_sec = row["max"]

        if str(max_sec).lower() == "infinity":
            max_sec = float("inf")

        formatted_columns.append(format_seconds_range(min_sec, max_sec))

    heatmap_data.columns = formatted_columns

    # ---- Draw heatmap -------------------------------------------------------
    vmax = heatmap_data.max().max()

    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        cbar_kws={"label": "Number of Jobs"},
        norm=LogNorm(vmin=1, vmax=vmax),
        annot_kws={"size": 13, "weight": "bold"},
    )

    plt.xlabel("Walltime", fontsize=13, fontweight="bold")
    plt.ylabel("Nodes used", fontsize=13, fontweight="bold")

    plt.xticks(rotation=45, ha="center", fontsize=11, fontweight="bold")
    plt.yticks(fontsize=11, fontweight="bold")

    # ---- Style the colour bar -----------------------------------------------
    cbar = ax.collections[0].colorbar
    assert cbar is not None, "Colorbar was not created"
    cbar.set_label("Number of Jobs", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    # ---- Save ---------------------------------------------------------------
    os.makedirs(output_path, exist_ok=True)
    if month and year:
        plt.title(
            f"{machine_name} — Job Size vs Walltime Distribution ({year}-{month:02d})",
            fontsize=16,
            fontweight="bold",
        )
    elif year:
        plt.title(
            f"{machine_name} — Job Size vs Walltime Distribution ({year})",
            fontsize=16,
            fontweight="bold",
        )
    else:
        plt.title(
            f"{machine_name} — Job Size vs Walltime Distribution",
            fontsize=16,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"{machine_name}_job_size_vs_walltime_heatmap.png"),
        dpi=300,
    )
    plt.close()

    print(
        "✅ Done plotting job size vs walltime heatmap on:",
        os.path.join(output_path, f"{machine_name}_job_size_vs_walltime_heatmap"),
    )


# =============================================================================
#  TEMPORAL SCATTER PLOT  (date vs. walltime, coloured by nodes)
# =============================================================================


def plot_heatmap_distribution_over_time(
    df, path, global_max_nodes, max_time, year=None, month=None, machine_name=""
):
    """
    Scatter plot of jobs over time.

    * **X-axis**: queued timestamp.
    * **Y-axis**: walltime (log scale; jobs < 100 s are excluded).
    * **Colour**: nodes used (viridis colourmap, 1 … *global_max_nodes*).
    * **Size**: proportional to nodes used.

    Saves ``<machine>_distribution_over_time*.png`` into *path*.
    """
    # Only keep jobs with walltime >= 100 s (removes noise from tiny jobs)
    df = df.copy()
    df = df[df["walltime_seconds"] >= 100]
    df["queued_timestamp"] = pd.to_datetime(df["queued_timestamp"], errors="coerce")

    # ---- Apply optional year / month filter for better aesthetics -----------
    if year is not None:
        df = df[df["queued_timestamp"].dt.year == year]
    if month is not None:
        df = df[df["queued_timestamp"].dt.month == month]
    if df.empty:
        print(
            f"⚠️ No data to plot for {machine_name} {year if year else ''} {month if month else ''}"
        )
        return

    os.makedirs(path, exist_ok=True)

    plt.figure(figsize=(15, 8))
    sizes = _scale_marker_size(df["nodes_used"], max_nodes_used=global_max_nodes)

    scatter = plt.scatter(
        df["queued_timestamp"],
        df["walltime_seconds"],
        c=df["nodes_used"],
        s=sizes,
        cmap="viridis",
        alpha=0.6,
        edgecolors="none",
        vmin=1,
        vmax=global_max_nodes,
    )
    ax = plt.gca()
    ax.set_xlim(df["queued_timestamp"].min(), df["queued_timestamp"].max())
    # ---- Fix colour bar styling ---------------------------------------------
    cbar = plt.colorbar(scatter)
    cbar.set_label("Nodes Used", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=13)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    # ---- X-axis date formatting (scatter plot) --------------------------------

    locator = mdates.AutoDateLocator(
        minticks=6, maxticks=15  # increase this → more ticks → more concentrated
    )
    ax.xaxis.set_major_locator(locator)

    if year and month:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.yscale("log")
    plt.ylim(bottom=100, top=max_time * 1.1)
    plt.xlabel("Date", fontsize=16, fontweight="bold")
    plt.ylabel("Walltime (seconds)", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    if year and month:
        plt.title(
            f"{machine_name} — Job Distribution Over {year}-{month:02d}",
            fontsize=16,
            fontweight="bold",
        )
    elif year:
        plt.title(
            f"{machine_name} — Job Distribution Over {year}",
            fontsize=16,
            fontweight="bold",
        )
    else:
        plt.title(
            f"{machine_name} — Job Distribution Over Time",
            fontsize=16,
            fontweight="bold",
        )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            path,
            f"{machine_name}_distribution_over_time{f'_{year}_{month:02d}' if year and month else f'_{year}' if year else ''}.png",
        ),
        dpi=300,
    )
    plt.close()
    print(
        "✅ Done plotting job distribution over time on:",
        os.path.join(
            path,
            f"{machine_name}_distribution_over_time{f'_{year}_{month:02d}' if year and month else f'_{year}' if year else ''}.png",
        ),
    )


# =============================================================================
#  MACHINE UTILIZATION PLOT  (time series with trend indicators)
# =============================================================================


def plot_machine_utilization(
    machine_name,
    utilization_pct,
    output_path,
    mean_nodes_pct,
    std_nodes_pct,
    rolling_window="1D",
    year=None,
    month=None,
):
    """
    Time-series plot of machine utilization (% of nodes in use).

    Overlays:
    * Raw utilization curve.
    * Horizontal mean line with ± std shaded band.
    * 50 % capacity reference line.
    * Rolling average (default 1-day window).
    * LOWESS trend (fraction = 0.01).

    Saves ``<machine>_utilization_over_time*.png`` into *output_path*.
    """
    plt.figure(figsize=(18, 5))

    # ---- (1) Raw utilization ------------------------------------------------
    plt.plot(
        utilization_pct.index,
        utilization_pct,
        linewidth=0.8,
        alpha=0.6,
        label="utilization (%)",
    )

    # ---- (2) Mean line + ± std band -----------------------------------------
    plt.axhline(
        mean_nodes_pct,
        color="orange",
        linestyle="--",
        label=f"Mean: {mean_nodes_pct:.2f}%",
    )
    plt.fill_between(
        utilization_pct.index,
        mean_nodes_pct - std_nodes_pct,
        mean_nodes_pct + std_nodes_pct,
        color="orange",
        alpha=0.2,
        label=f"Std: {std_nodes_pct:.2f}%",
    )

    # ---- (3) 50 % capacity reference line -----------------------------------
    plt.axhline(50, color="red", linestyle="--", label="50% Capacity")

    # ---- (4) Rolling average ------------------------------------------------
    utilization_roll = utilization_pct.rolling(rolling_window, min_periods=1).mean()
    plt.plot(
        utilization_roll.index,
        utilization_roll,
        color="green",
        linestyle="-",
        linewidth=1.2,
        label=f"Rolling Mean ({rolling_window})",
    )

    # ---- (5) LOWESS trend ---------------------------------------------------
    utilization_lowess = lowess(
        utilization_pct.values, np.arange(len(utilization_pct)), frac=0.01
    )
    plt.plot(
        utilization_pct.index,
        utilization_lowess[:, 1],
        color="purple",
        linestyle="-",
        linewidth=1.5,
        label="LOWESS Trend",
    )

    # ---- X-axis date formatting ---------------------------------------------
    ax = plt.gca()
    locator = mdates.AutoDateLocator(
        minticks=6, maxticks=15  # increase this → more ticks → more concentrated
    )
    ax.xaxis.set_major_locator(locator)

    if year and month:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.set_xlim(utilization_pct.index.min(), utilization_pct.index.max())
    plt.xticks(rotation=30, fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.ylim(0, utilization_pct.max() * 1.1 if utilization_pct.max() > 100 else 100)
    plt.ylabel("Nodes in use (%)", fontsize=16, fontweight="bold")
    plt.xlabel("Time", fontsize=16, fontweight="bold")
    if month and year:
        plt.title(
            f"{machine_name} — Machine Utilization Over {year}-{month:02d}\nMean: {mean_nodes_pct:.2f}%, Std: {std_nodes_pct:.2f}%",
            fontsize=16,
            fontweight="bold",
        )
    elif year:
        plt.title(
            f"{machine_name} — Machine Utilization Over {year}\nMean: {mean_nodes_pct:.2f}%, Std: {std_nodes_pct:.2f}%",
            fontsize=16,
            fontweight="bold",
        )
    else:
        plt.title(
            f"{machine_name} — Machine Utilization Over Time\nMean: {mean_nodes_pct:.2f}%, Std: {std_nodes_pct:.2f}%",
            fontsize=16,
            fontweight="bold",
        )
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    legend = plt.legend(fontsize=13, title_fontsize=14, loc="lower center")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_path,
            f"{machine_name}_utilization_over_time{f'_{year}_{month:02d}' if year and month else f'_{year}' if year else ''}.png",
        ),
        dpi=300,
    )
    plt.close()
    print(f"✅ Plot saved to {output_path}")


# =============================================================================
#  QUEUE CORE-HOUR BREAKDOWN PLOTS
# =============================================================================


# ---- Stacked Bar (daily core-hours by queue) --------------------------------
def plot_stacked_bar_queue_percentages(pivoted, out_path, title, color_map, top_n=10):
    """
    Daily stacked-bar chart of core-hours by queue.

    Only the *top_n* queues (by total core-hours) are shown individually;
    the rest are collapsed into an "Other" category.

    Saves the chart to *out_path*.
    """
    data_cols = [c for c in pivoted.columns if c != "total_runtime_hours"]
    col_totals = pivoted[data_cols].sum().sort_values(ascending=False)
    top_cols = col_totals.head(top_n).index.tolist()

    plot_df = pivoted[top_cols].copy()
    other_cols = [c for c in data_cols if c not in top_cols]
    if other_cols:
        plot_df["Other"] = pivoted[other_cols].sum(axis=1)

    fig, ax = plt.subplots(figsize=(18, 7))
    colors = [color_map.get(c, "gray") for c in plot_df.columns]

    plot_df.plot(
        kind="bar", stacked=True, ax=ax, color=colors, width=0.9, edgecolor="none"
    )

    # X-axis formatting: show a manageable number of date tick labels
    n_ticks = min(len(plot_df), 30)
    step = max(1, len(plot_df) // n_ticks)
    tick_positions = list(range(0, len(plot_df), step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [plot_df.index[i].strftime("%Y-%m-%d") for i in tick_positions],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Core-Hours")
    ax.set_title(title)
    ax.legend(title="Queue", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Stacked bar chart saved: {out_path}")


# ---- Cumulative Stacked Area (daily core-hours by queue) --------------------
def plot_stacked_area_queue_percentages(
    pivoted, out_path, title, color_map, top_n=10, year=None, month=None
):
    """
    Cumulative stacked-area chart of core-hours by queue.

    Each day’s value is a **running total** (cumsum).  Shows the *top_n*
    queues plus any queue with ≥  3 % share; the rest collapse into "Other".

    Saves the chart to *out_path*.
    """
    data_cols = [c for c in pivoted.columns if c != "total_runtime_hours"]
    col_totals = pivoted[data_cols].sum().sort_values(ascending=False)
    total = col_totals.sum()

    # Top N + anything ≥ 3 %
    top_cols = col_totals.head(top_n).index.tolist()
    for col, val in col_totals.items():
        if val / total >= 0.03 and col not in top_cols:
            top_cols.append(col)

    plot_df = pivoted[top_cols].copy()
    other_cols = [c for c in data_cols if c not in top_cols]
    if other_cols:
        plot_df["Other"] = pivoted[other_cols].sum(axis=1)

    # Cumulative sum so each day = running total
    plot_df = plot_df.cumsum()

    fig, ax = plt.subplots(figsize=(18, 7))
    colors = [
        color_map.get(c, matplotlib.colormaps["tab20"](i % 20))
        for i, c in enumerate(plot_df.columns)
    ]

    plot_df.plot(kind="area", stacked=True, ax=ax, color=colors, linewidth=0.3)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Core-Hours")
    ax.set_title(title)
    ax.legend(title="Queue", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Cumulative stacked area chart saved: {out_path}")


# ---- Pie Chart (overall core-hours share by queue) --------------------------
def plot_pie_chart_queue_percentages(
    pivoted, out_path, title, color_map, threshold_pct=1.0, year=None, month=None
):
    """
    Pie chart showing overall core-hours share per queue.

    Queues contributing < *threshold_pct* % are grouped into "Other".

    Saves the chart to *out_path*.
    """
    data_cols = [c for c in pivoted.columns if c != "total_runtime_hours"]
    totals = pivoted[data_cols].sum().sort_values(ascending=False)
    collapsed = _collapse_small_categories(totals, threshold_percent=threshold_pct)

    colors = [color_map.get(c, "gray") for c in collapsed.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    pie_result = ax.pie(
        collapsed,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.8,
    )
    wedges = pie_result[0]
    ax.legend(
        wedges,
        collapsed.index,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        reverse=True,
    )
    ax.set_title(title)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Pie chart saved: {out_path}")

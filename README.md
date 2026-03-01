# HPC Workload Analysis

A Python analysis pipeline for characterising **High-Performance Computing (HPC) job scheduler logs**.  
It ingests raw job data, cleans it, computes per-queue and machine-wide statistics, and produces a comprehensive set of publication-ready plots and text reports.

---

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Input Data Requirements](#input-data-requirements)  
   - [Raw Job Log CSV](#raw-job-log-csv)  
   - [Machine Configuration Files](#machine-configuration-files)  
4. [Preprocessing](#preprocessing)  
5. [Running the Analysis](#running-the-analysis)  
   - [CLI Arguments](#cli-arguments)  
6. [Output Description](#output-description)  
   - [Directory Layout](#output-directory-layout)  
   - [Plots Produced](#plots-produced)  
   - [Text Reports](#text-reports)  
7. [Module Reference](#module-reference)  
8. [Dependencies](#dependencies)

---

## Overview

The pipeline has **two stages**:

|   Stage   | Script | Purpose |
|---------|--------|---------|
| **1 — Preprocessing** | `src/preprocessor.py` | Reads raw scheduler CSVs, normalises columns, derives metrics, filters anomalies, deduplicates, and writes a compressed `*_preprocessed.csv.gz`. |
| **2 — Analysis** | `src/main.py` | Reads a preprocessed CSV plus machine-config files and generates all plots and reports. |

### What the analysis produces

* **Temporal distribution** plots — job submission patterns by hour, weekday, and day-of-year.
* **Job-size distribution** — bar chart of jobs across configurable node-count bins.
* **Walltime distribution** — bar chart of jobs across configurable walltime bins.
* **Job-size × Walltime heatmap** — 2-D heat map with log colour scale.
* **Temporal scatter plot** — every job on a timeline (x = date, y = walltime, colour/size = nodes).
* **Machine utilization** — time-series of node usage % with rolling/LOWESS trends.
* **Per-queue core-hour breakdown** — stacked bar, stacked area, and pie charts.
* **Detailed text summaries** — per-queue statistics (mean/median/P5/P95 for nodes, walltime, runtime, wait time, core-hours, efficiency).

All analyses can optionally be repeated **per year** and **per month**.

---

## Repository Structure

```
HPC-workload-analysis/
├── README.md                   # This file
├── LICENSE
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
├── data/                       # Input data (one sub-folder per machine)
│   ├── Polaris/
│   │   ├── *.csv.gz            # Raw job log files (from scheduler)
│   │   ├── jobs_preprocessed.csv.gz  # Merged preprocessed output
│   │   ├── job_dimension.csv   # Node-count bin definitions
│   │   ├── walltime.csv        # Walltime bin definitions
│   │   ├── max_nodes.csv       # Machine capacity (single value)
│   │   └── queue_names.csv     # Valid queue definitions
│   └── Aurora/
│       └── ...                 # Same structure
├── src/                        # Source code (main pipeline)
│   ├── main.py                 # CLI entry point & top-level driver
│   ├── orchestrator.py         # Central coordinator between data & plots
│   ├── preprocessor.py         # Raw → preprocessed CSV converter
│   ├── plotting.py             # All visualisation functions
│   ├── report.py               # Text report generator
│   ├── single_queue_analysis.py  # Per-queue core-hour breakdown
│   ├── system_utilization.py   # Node-utilization computation & plots
│   └── utils.py                # Shared helpers, CLI parser, config loader
├── analysis_output/            # Generated output (one sub-folder per machine)
│   ├── Polaris/
│   │   ├── *.png               # Full-range plots
│   │   ├── 2024_analysis/      # Per-year plots
│   │   │   ├── *.png
│   │   │   └── 6_analysis/     # Per-month plots (if --monthly)
│   │   └── queue_analysis/     # Queue breakdown (if --queue-analysis)
│   └── Aurora/
│       └── ...
└── Additional/                 # Supplementary / advanced scripts
```

---

## Input Data Requirements

### Data Sources (ALCF)

| What | URL |
|------|-----|
| **Job log CSVs** | <https://reports.alcf.anl.gov/data/> |
| **Polaris queue definitions** | <https://docs.alcf.anl.gov/polaris/running-jobs/> |
| **Aurora queue definitions** | <https://docs.alcf.anl.gov/aurora/running-jobs-aurora/> |

* Download the raw job log files from the ALCF Reports portal.
* Use the Polaris / Aurora documentation pages to populate `queue_names.csv` (queue names, node limits, walltime limits) and `job_dimension.csv` (node-count bins matching the queue boundaries).
* The walltime bins in `walltime.csv` were derived from the queue walltime limits listed in the documentation — set them to whatever boundaries make sense for your analysis.

### Raw Job Log CSV

Each machine folder under `data/` must contain one or more raw CSV files exported from the job scheduler (plain or gzip-compressed).  
The following columns are **required** (case-insensitive):

| Column | Type | Description |
|--------|------|-------------|
| `JOB_NAME` | string | Human-readable job identifier |
| `USERNAME_GENID` | string | Anonymised user identifier |
| `PROJECT_NAME_GENID` | string | Anonymised project identifier |
| `QUEUE_NAME` | string | Queue the job was submitted to |
| `QUEUED_TIMESTAMP` | datetime | When the job was submitted |
| `START_TIMESTAMP` | datetime | When the job started executing |
| `END_TIMESTAMP` | datetime | When the job finished |
| `WALLTIME_SECONDS` | float | Requested walltime (seconds) |
| `RUNTIME_SECONDS` | float | Actual runtime (seconds) |
| `NODES_REQUESTED` | int | Number of nodes the user asked for |
| `NODES_USED` | int | Number of nodes actually allocated |
| `USED_CORE_HOURS` | float | Core-hours consumed by the job |
| `EXIT_CODE` | int | Job exit code (0 = success) |

### Machine Configuration Files

Four small CSV files must sit **alongside** the job data in the same machine folder:

#### `job_dimension.csv` — Node-count bins

Defines how jobs are grouped by size (number of nodes).

```csv
name,min,max
tiny,1,10
small,11,24
medium,25,99
large,100,496
```

| Column | Description |
|--------|-------------|
| `name` | Human-readable label for the bin |
| `min` | Minimum node count (inclusive) |
| `max` | Maximum node count (inclusive); use `infinity` for unbounded |

#### `walltime.csv` — Walltime bins

Defines how jobs are grouped by requested walltime (in seconds).

```csv
name,min,max
shortest,1,60
short,61,600
medium-short,601,1800
medium,1801,3600
medium-long,3601,7200
long,7201,21600
very-long,21601,43200
super-extreme,43201,64800
ultra,64801,86400
mega,86401,259200
extra-infinity,259201,infinity
```

| Column | Description |
|--------|-------------|
| `name` | Label |
| `min` | Minimum walltime in seconds (inclusive) |
| `max` | Maximum walltime in seconds (inclusive); use `infinity` for unbounded |

#### `max_nodes.csv` — Machine capacity

A single-value file declaring the total number of compute nodes.

```csv
max_nodes
560
```

#### `queue_names.csv` — Valid queue definitions

Lists the queues to include in the queue-analysis breakdown.  
Only used when `--queue-analysis` is enabled.

Populate this file from the official ALCF documentation:
* **Polaris**: <https://docs.alcf.anl.gov/polaris/running-jobs/>
* **Aurora**: <https://docs.alcf.anl.gov/aurora/running-jobs-aurora/>

```csv
queue_name,min_nodes,max_nodes,min_walltime,max_walltime
debug,1,2,00:05:00,01:00:00
small,10,24,00:05:00,03:00:00
medium,25,99,00:05:00,06:00:00
large,100,496,00:05:00,24:00:00
```

| Column | Description |
|--------|-------------|
| `queue_name` | Queue identifier (must match values in the job log) |
| `min_nodes` | Minimum node allocation for this queue |
| `max_nodes` | Maximum node allocation for this queue |
| `min_walltime` | Minimum walltime (HH:MM:SS) |
| `max_walltime` | Maximum walltime (HH:MM:SS) |

### Example folder layout for a machine

```
data/Polaris/
├── ANL-ALCF-DJC-POLARIS_20220809_20221231.csv.gz   # Raw data (year 1)
├── ANL-ALCF-DJC-POLARIS_20230101_20231231.csv.gz   # Raw data (year 2)
├── jobs_preprocessed.csv.gz   # Output of preprocessor (merged)
├── job_dimension.csv          # Node bins
├── walltime.csv               # Walltime bins
├── max_nodes.csv              # Machine capacity
└── queue_names.csv            # Queue definitions
```

---

## Preprocessing

Run the preprocessor **before** the main analysis.

```bash
# Navigate to the source directory
cd src/

# Single file
python preprocessor.py --path ../data/Polaris/raw_jobs.csv.gz --single

# All CSV files in a directory (merged into one output)
python preprocessor.py --path ../data/Polaris/ --all
```

**What the preprocessor does:**

1. Reads raw CSV(s) and keeps only the required columns.
2. Normalises column names to lowercase.
3. Parses timestamp columns.
4. Derives: `wait_seconds` (time spent in queue before execution).
5. Filters out invalid rows:
   - `runtime_seconds >= 1.5 × walltime_seconds`
   - Negative runtimes or walltimes
   - Negative core-hours
6. Deduplicates by `job_name`.
7. Writes `*_preprocessed.csv.gz` (per-file and merged).

---

## Running the Analysis

```bash
cd src/

# Minimal run (distribution plots only)
python main.py --path ../data/Polaris/jobs_preprocessed.csv.gz \
               --machine-name Polaris

# Full analysis with all extras
python main.py --path ../data/Polaris/jobs_preprocessed.csv.gz \
               --machine-name Polaris \
               --queue-analysis \
               --machine-utilization \
               --yearly \
               --monthly \
               --full-queue-analysis
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--path` | **Yes** | — | Path to the preprocessed `.csv.gz` file |
| `--machine-name` | **Yes** | — | Machine label (e.g. `Polaris`, `Aurora`). Used in plot titles and filenames. |
| `--queue-analysis` / `--no-queue-analysis` | No | `False` | Generate per-queue core-hour breakdown (stacked bar, stacked area, pie charts, text summaries). The breakdown is produced for the full dataset; when combined with `--yearly` or `--monthly`, a `queue_analysis/` folder is also created inside each year / month slice. Requires `queue_names.csv`. |
| `--machine-utilization` / `--no-machine-utilization` | No | `False` | Compute and plot node utilization over time. |
| `--yearly` / `--no-yearly` | No | `False` | Repeat the analysis for each calendar year found in the data. |
| `--monthly` / `--no-monthly` | No | `False` | Repeat the analysis for each (year, month) pair found in the data. |
| `--full-queue-analysis` / `--no-full-queue-analysis` | No | `False` | Generate per-queue distribution plots (job-size, walltime, heatmap, scatter) for **every** individual queue, in queue_names.csv in the original data folder. |

Output is written to `analysis_output/<machine-name>/` (created automatically).

---

## Output Description

### Output Directory Layout

```
analysis_output/Polaris/
├── distribution_over_days.png
├── distribution_over_hours.png
├── distribution_over_years.png
├── job_size_distribution.png
├── distribution_over_walltime.png
├── Polaris_job_size_vs_walltime_heatmap.png
├── Polaris_distribution_over_time.png
├── Polaris_utilization_over_time.png          # (--machine-utilization)
│
├── queue_analysis/                            # (--queue-analysis, full-range)
│   ├── Polaris_queue_analysis_monthly_stacked_bar.png
│   ├── Polaris_queue_analysis_monthly_cumulative_area.png
│   ├── Polaris_queue_analysis_monthly_pie_chart.png
│   ├── Polaris_queue_analysis_detailed_summary.txt
│   └── Polaris_queue_analysis_successful_jobs_detailed_summary.txt
│
├── queue_<name>/                              # (--full-queue-analysis)
│   ├── job_size_distribution.png
│   ├── distribution_over_walltime.png
│   ├── *_job_size_vs_walltime_heatmap.png
│   └── *_distribution_over_time*.png
│
├── 2024_analysis/                             # (--yearly)
│   ├── distribution_over_days.png
│   ├── distribution_over_hours.png
│   ├── distribution_over_years.png
│   ├── job_size_distribution.png
│   ├── distribution_over_walltime.png
│   ├── Polaris_job_size_vs_walltime_heatmap.png
│   ├── Polaris_distribution_over_time_2024.png
│   ├── Polaris_utilization_over_time_2024.png
│   │
│   ├── queue_analysis/                        # (--queue-analysis + --yearly)
│   │   ├── *_weekly_stacked_bar.png
│   │   ├── *_weekly_cumulative_area.png
│   │   ├── *_weekly_pie_chart.png
│   │   ├── *_detailed_summary.txt
│   │   └── *_successful_jobs_detailed_summary.txt
│   │
│   └── 6_analysis/                            # (--monthly, nested under year)
│       ├── distribution_over_days.png
│       ├── ...
│       └── queue_analysis/                    # (--queue-analysis + --monthly)
│           ├── *_daily_stacked_bar.png
│           ├── *_daily_cumulative_area.png
│           ├── *_daily_pie_chart.png
│           ├── *_detailed_summary.txt
│           └── *_successful_jobs_detailed_summary.txt
└── ...
```

### Plots Produced

| Plot | Filename | Description |
|------|----------|-------------|
| **Hours distribution** | `distribution_over_hours.png` | Bar chart — mean % of jobs by hour of day (0–23) with std error bars. |
| **Weekday distribution** | `distribution_over_days.png` | Bar chart — mean % of jobs by weekday (Mon–Sun) with std error bars. |
| **Day-of-year distribution** | `distribution_over_years.png` | Smoothed line chart — rolling mean % by day-of-year (Jan–Dec) with ±std band. |
| **Job-size distribution** | `job_size_distribution.png` | Bar chart — % of jobs in each node-count bin (annotated). |
| **Walltime distribution** | `distribution_over_walltime.png` | Colour-coded bar chart with legend mapping position labels to walltime ranges. |
| **Job-size × Walltime heatmap** | `*_job_size_vs_walltime_heatmap.png` | 2-D heatmap with log colour scale; rows = node bins, columns = walltime bins. |
| **Temporal scatter** | `*_distribution_over_time*.png` | Scatter plot (x = date, y = walltime [log], colour/size = nodes used). |
| **Machine utilization** | `*_utilization_over_time*.png` | Time-series: raw utilization, mean ± std band, 50 % line, rolling average, LOWESS trend. |
| **Queue stacked bar** | `*_stacked_bar.png` | Daily/weekly/monthly core-hours by queue (top N + Other). |
| **Queue stacked area** | `*_cumulative_area.png` | Cumulative core-hours running total by queue. |
| **Queue pie chart** | `*_pie_chart.png` | Overall core-hours share per queue (small queues collapsed into Other). |

### Text Reports

| Report | Filename | Description |
|--------|----------|-------------|
| **Detailed queue summary** | `*_detailed_summary.txt` | Per-queue: job count, core-hours, share %, efficiency, mean/median/P5/P95/min/max for nodes, walltime, runtime, wait, core-hours, unique users/projects. |
| **Successful-only summary** | `*_successful_jobs_detailed_summary.txt` | Same as above but filtered to exit_code ∈ {0, −29}. |

---

## Module Reference

| Module | Role |
|--------|------|
| `main.py` | CLI entry point.  Parses arguments, reads the CSV, loads config, drives full / per-year / per-month analysis. |
| `orchestrator.py` | Central coordinator.  Computes time distributions and delegates to plotting + queue analysis. |
| `preprocessor.py` | Raw → clean CSV converter.  Run before the main analysis. |
| `plotting.py` | All matplotlib/seaborn visualisation functions (13 plot types). |
| `report.py` | Generates plain-text per-queue statistical summaries. |
| `single_queue_analysis.py` | Queue core-hour breakdown at daily / weekly / monthly granularity. |
| `system_utilization.py` | Event-based node utilization computation + over-capacity detection. |
| `utils.py` | Shared helpers: CLI parser, config loader, bin counting, pivoting, colour generation, formatting. |

---

## Dependencies

Requires **Python 3.10+**. All dependencies are listed in [`requirements.txt`](requirements.txt).

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, filtering, aggregation |
| `numpy` | Numerical operations, array handling |
| `matplotlib` | Plot rendering (Agg backend) |
| `seaborn` | Statistical heatmaps |
| `statsmodels` | LOWESS trend line in utilization plot |
| `colour-science` | Perceptually distinct colour generation (LCH space) |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

*PhD Research — HPC Workload Analysis*
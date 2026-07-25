# HPC Workload Analysis

If you use the tool, please cite the following paper:

Melanie Cornelius, Greg Cross, Shilpika, Michael E. Papka, and Zhiling Lan, "Fusing System Data to Navigate Power-Saving Opportunities", the 29th Workshop on Job Scheduling Strategies for Parallel Processing (JSSPP 2026), in conjunction with IPDPS 2026.


A modular Python pipeline for analysing **High-Performance Computing (HPC) scheduler logs**.

It transforms raw job exports into a clean dataset, computes machine-wide and per-queue statistics, and produces **publication-ready plots and detailed text reports**.

---

## ✨ Key Features

- 🧹 Automated data cleaning & validation  
- 📊 Temporal workload characterisation  
- 📦 Job size & walltime distributions  
- 🔥 Job size × walltime heatmaps  
- 🧮 Per-queue core-hour accounting  
- 🖥 Machine utilization over time  
- 📅 Optional yearly / monthly breakdowns  
- 📄 Structured text summaries for reporting  

---

# 🏗 Pipeline Overview

The workflow is composed of **two stages**:

---

## 🧹 Stage 1 — Preprocessing  
`src/preprocessor.py`

Transforms raw scheduler exports into a clean, analysis-ready dataset.

### ✔ What it does

- Reads raw HPC scheduler CSV logs  
- Normalises column names  
- Parses timestamps  
- Computes derived metrics (runtime, wait time, core-hours)  
- Filters corrupted / anomalous records  
- Removes duplicate jobs  
- Outputs compressed dataset  

### 📦 Output
```
*_preprocessed.csv.gz
```


---

## 📊 Stage 2 — Analysis  
`src/main.py`

Generates figures, statistics, and structured reports.

---

### 📅 Temporal Workload Characterisation

- **Hourly submission distribution**
- **Weekday distribution**
- **Day-of-year distribution** (seasonality)

---

### 📦 Job Characteristics

- Job-size distribution (node-count bins)
- Walltime distribution (custom bins)
- Job-size × Walltime heatmap (log colour scale)

---

### 📊 Temporal & System Behaviour

- **Timeline scatter plot**
  - x-axis: submission date  
  - y-axis: walltime (log scale)  
  - marker size/colour: node count  

- **Machine utilization**
  - Time-series of node usage (%)  
  - Rolling averages  
  - LOWESS trend estimation  

---

### 🧮 Queue-Level Resource Accounting

- Stacked bar charts (core-hours by queue)
- Stacked cumulative area charts
- Pie charts (overall share)

#### Detailed statistical summaries per queue:

- Nodes (mean / median / P5 / P95)
- Walltime
- Runtime
- Wait time
- Core-hours
- Efficiency

---

### 🔁 Optional Granularity

All analyses can be repeated:

- Per year (`--yearly`)
- Per month (`--monthly`)

---

# 📂 Repository Structure
```
HPC-workload-analysis/
├── README.md                   # This file
├── LICENSE
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
```

---

# 📥 Input Data Requirements

## Data Sources (ALCF)

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

# 🧹 Preprocessing

Run **before** analysis:

```bash
cd src/

# Single file
python preprocessor.py --path ../data/Polaris/file.csv.gz --single

# All files in directory
python preprocessor.py --path ../data/Polaris/ --all
```

---

## 🧹 Filtering Rules

The preprocessor removes rows with:

- `runtime ≥ 1.5 × walltime`
- Negative runtimes or walltimes
- Negative core-hours
- Duplicate `job_name`

---

# ▶ Running the Analysis

## Minimal run

```bash
python main.py \
  --path ../data/Polaris/jobs_preprocessed.csv.gz \
  --machine-name Polaris
  ```

  ## Full analysis

```bash
python main.py \
  --path ../data/Polaris/jobs_preprocessed.csv.gz \
  --machine-name Polaris \
  --queue-analysis \
  --machine-utilization \
  --yearly \
  --monthly \
  --full-queue-analysis
```
Output is written to:

```text
analysis_output/<machine-name>/
```
---

# 📊 Generated Plots

- Hourly distribution  
- Weekday distribution  
- Day-of-year seasonality  
- Job-size distribution  
- Walltime distribution  
- Job-size × Walltime heatmap  
- Timeline scatter plot  
- Machine utilization time-series  
- Queue stacked bar / area / pie charts  

---

# 📄 Text Reports

- Detailed per-queue statistical summary  
- Successful-jobs-only summary  

### Each report includes

- Job count  
- Core-hours  
- Share %  
- Efficiency  
- Mean / median / P5 / P95 / min / max  
- Unique users & projects  

---

# 🧠 Module Responsibilities

| Module | Purpose |
|--------|---------|
| `main.py` | CLI entry point |
| `orchestrator.py` | Coordinates computations |
| `preprocessor.py` | Cleans raw CSV logs |
| `plotting.py` | Generates visualisations |
| `report.py` | Builds text summaries |
| `single_queue_analysis.py` | Queue core-hour breakdown |
| `system_utilization.py` | Node utilization computation |
| `utils.py` | Shared helpers |

---

# ⚙ Dependencies

Tested on **Python 3.10+**

Install:

```bash
pip install -r requirements.txt
```

---

## Core Packages

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `colour-science`

---

# 🎓 Research Context

Developed as part of PhD research on **HPC workload characterisation and resource utilization modelling**.

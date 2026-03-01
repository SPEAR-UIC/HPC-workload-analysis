"""
Preprocessor for Raw HPC Job Log Data.

This script is run **before** the main analysis pipeline.  It ingests one
or more raw CSV files produced by the HPC scheduler, cleans and normalises
the data, derives several convenience columns, filters out anomalous rows,
and writes out a compressed ``*_preprocessed.csv.gz`` file ready for
``main.py``.

Workflow
--------
1. Read the raw CSV (plain or gzip-compressed).
2. Normalise column names to lowercase.
3. Parse timestamp columns (``queued_timestamp``, ``start_timestamp``,
   ``end_timestamp``).
4. Derive metrics: ``wait_seconds``, ``runtime_seconds_check``.
5. Filter out invalid rows:
   - ``runtime_seconds >= 1.5 * walltime_seconds``
   - ``runtime_seconds < 0`` or ``walltime_seconds < 0``
   - ``used_core_hours < 0``
6. Drop duplicate jobs (by ``job_name``).
7. Save the result as ``<original_name>_preprocessed.csv.gz``.

Usage
-----
::

    # Single file
    python preprocessor.py --path path/to/raw.csv.gz --single

    # All CSV files in a directory (merged into one output)
    python preprocessor.py --path path/to/directory/ --all

The ``--all`` mode concatenates every non-config CSV in the directory,
deduplicates, sorts by ``queued_timestamp``, and writes a single
``jobs_preprocessed.csv.gz``.

Expected raw columns
--------------------
See ``COLUMNS_TO_KEEP`` below.  The raw CSV must contain at least these
columns (case-insensitive).
"""

import pandas as pd
import sys
import os
import argparse

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Default path (overridden by --path CLI argument)
INPUT_PATH = "jobs.csv.gz"

# Columns to extract from the raw scheduler CSV.
# These are the minimum columns required for the analysis pipeline.
COLUMNS_TO_KEEP = [
    "JOB_NAME",  # Human-readable job identifier
    "USERNAME_GENID",  # Anonymised user identifier
    "PROJECT_NAME_GENID",  # Anonymised project identifier
    "QUEUE_NAME",  # Queue the job was submitted to
    "QUEUED_TIMESTAMP",  # When the job was submitted
    "START_TIMESTAMP",  # When the job started running
    "END_TIMESTAMP",  # When the job finished
    "WALLTIME_SECONDS",  # Requested walltime (seconds)
    "RUNTIME_SECONDS",  # Actual runtime (seconds)
    "NODES_REQUESTED",  # Nodes the user asked for
    "NODES_USED",  # Nodes actually allocated
    "USED_CORE_HOURS",  # Core-hours consumed
    "EXIT_CODE",  # Job exit code (0 = success)
]


# =============================================================================
#  CORE PREPROCESSING FUNCTION
# =============================================================================


def main(
    input_path=INPUT_PATH, columns_to_keep=COLUMNS_TO_KEEP, df_to_save=pd.DataFrame()
):
    """
    Preprocess a single raw HPC job log CSV.

    Parameters
    ----------
    input_path : str
        Path to raw CSV file (plain or ``.gz``).
    columns_to_keep : list of str
        Columns to extract from the raw data.
    df_to_save : pd.DataFrame
        Accumulator DataFrame – the preprocessed rows are appended to
        this so that multiple files can be merged in ``--all`` mode.

    Returns
    -------
    pd.DataFrame
        The accumulated, deduplicated DataFrame.
    """
    all_columns = columns_to_keep.copy()

    # ---- Step 1: Read raw CSV -----------------------------------------------
    print("Reading compressed CSV...")
    if input_path.endswith(".gz"):
        df = pd.read_csv(input_path, usecols=all_columns, compression="gzip")
    else:
        df = pd.read_csv(input_path, usecols=all_columns)
    print("✅ Done. Dataset rows read:", df.shape[0])

    # ---- Step 2: Normalise column names to lowercase ------------------------
    df.columns = [c.lower() for c in df.columns]

    # ---- Step 3: Parse timestamp columns ------------------------------------
    print("Parsing timestamps...")
    for col in ["queued_timestamp", "start_timestamp", "end_timestamp"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---- Step 4: Derive convenience metrics ---------------------------------
    print("Computing derived metrics...")
    # Wait time = how long the job sat in the queue before starting
    df["wait_seconds"] = (
        df["start_timestamp"] - df["queued_timestamp"]
    ).dt.total_seconds()
    # Cross-check runtime from timestamps
    df["runtime_seconds_check"] = (
        df["end_timestamp"] - df["start_timestamp"]
    ).dt.total_seconds()

    # Mark negative waits/runtimes as missing (data anomaly)
    df.loc[df["wait_seconds"] < 0, "wait_seconds"] = pd.NA
    df.loc[df["runtime_seconds_check"] < 0, "runtime_seconds_check"] = pd.NA

    # ---- Print all columns for debugging ------------------------------------
    print("Dataset columns after derived metrics:", df.columns.tolist())

    # ---- Step 5: Select final columns in a deterministic order --------------
    final_columns = [
        "job_name",
        "username_genid",
        "project_name_genid",
        "queue_name",
        "queued_timestamp",
        "wait_seconds",
        "start_timestamp",
        "end_timestamp",
        "walltime_seconds",
        "runtime_seconds",
        "nodes_requested",
        "nodes_used",
        "used_core_hours",
        "exit_code",
    ]

    # Keep only the columns that actually exist in the DataFrame
    df = df[[c for c in final_columns if c in df.columns]]

    # ---- Step 6: Filter anomalous rows --------------------------------------
    # Remove jobs where runtime greatly exceeds the requested walltime
    print("Filtering out rows with runtime_seconds >= 1.5 * walltime_seconds...")
    print("Rows before filtering:", df.shape[0])
    df = df[df["runtime_seconds"] < 1.5 * df["walltime_seconds"]]
    print("Rows after filtering:", df.shape[0])
    # Remove jobs with negative runtime or walltime
    print("Filtering out rows with runtime_seconds < 0 or walltime_seconds < 0...")
    print("Rows before filtering:", df.shape[0])
    df = df[(df["runtime_seconds"] >= 0) & (df["walltime_seconds"] >= 0)]
    print("Rows after filtering:", df.shape[0])
    # Remove jobs with negative core-hours
    print("Filtering out rows with used_core_hours < 0...")
    print("Rows before filtering:", df.shape[0])
    df = df[df["used_core_hours"] >= 0]
    print("Rows after filtering:", df.shape[0])

    # ---- Step 7: Accumulate and deduplicate ---------------------------------
    print("✅ Done. Preprocessed dataset read from:", input_path)
    df_to_save = pd.concat([df_to_save, df], ignore_index=True)

    # Remove duplicate jobs (identified by job_name)
    num_rows_before = df_to_save.shape[0]
    df_to_save = df_to_save.drop_duplicates(subset=["job_name"])
    num_rows_after = df_to_save.shape[0]
    print(f"✅ Dropped {num_rows_before - num_rows_after} duplicate rows.")

    # ---- Step 8: Write per-file preprocessed output -------------------------
    suffix = "_preprocessed.csv.gz"
    output_path = os.path.join(
        os.path.dirname(input_path),
        os.path.basename(input_path).rsplit(".", 1)[0] + suffix,
    )
    df.to_csv(output_path, index=False, compression="gzip")
    print("Saving preprocessed data in file:", output_path)
    print("✅ Preprocessed data saved to:", output_path)

    return df_to_save


# =============================================================================
#  CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # ---- Parse command-line arguments ----------------------------------------
    parser = argparse.ArgumentParser(
        description="Preprocess raw HPC job logs into analysis-ready CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preprocess a single file
    python preprocessor.py --path path/to/raw.csv.gz --single

    # Process all CSV files in a directory (merged output)
    python preprocessor.py --path path/to/directory --all
        """,
    )
    parser.add_argument("--path", help="Path to a single CSV file or a directory")
    parser.add_argument(
        "--all", action="store_true", help="Process ALL CSV files in --path directory"
    )
    parser.add_argument("--single", action="store_true", help="Process a single file")

    # Support legacy positional-argument style:  preprocessor.py <path> --all
    if len(sys.argv) >= 3 and sys.argv[2] in ["--all", "--single"]:
        # Legacy format: test.py path --all
        args = parser.parse_args([sys.argv[1], sys.argv[2]] + sys.argv[3:])
    else:
        args = parser.parse_args()

    if args.all:
        # ---- Batch mode: process every CSV in the given directory ------------
        print("Processing all CSV files in directory...")
        if not os.path.isdir(args.path):
            print("Error: provided path is not a directory")
            sys.exit(1)

        df_to_save = pd.DataFrame()  # accumulator
        output_file = "jobs_preprocessed.csv.gz"
        for file in os.listdir(args.path):
            print("Found file:", file)
            # Skip config files and already-preprocessed files
            if (
                (file.endswith(".csv") or file.endswith(".csv.gz"))
                and "preprocessed" not in file
                and "job_dimension" not in file
                and "walltime" not in file
                and "queue" not in file
                and "max_nodes" not in file
            ):
                print("Processing file:", file)
                df_to_save = main(
                    input_path=os.path.join(args.path, file),
                    df_to_save=df_to_save,
                )
        # Sort chronologically and write the merged output
        df_to_save = df_to_save.sort_values(by=["queued_timestamp"])
        df_to_save.to_csv(
            os.path.join(args.path, output_file), index=False, compression="gzip"
        )
        print(
            "✅ Done. Data saved to preprocessed CSV file:",
            os.path.join(args.path, output_file),
        )
        print("Final dataset rows:", df_to_save.shape[0])
    else:
        # ---- Single-file mode -----------------------------------------------
        csv_path = args.path
        # Build output filename from input (e.g. jobs.csv.gz → jobs_preprocessed.csv.gz)
        suffix = "_preprocessed.csv.gz"
        output_path = csv_path.rsplit(".")[0] + suffix
        df_to_save = pd.DataFrame()
        df_to_save = main(input_path=csv_path, df_to_save=df_to_save)
        print("✅ Done. Data saved to preprocessed CSV file:", output_path)

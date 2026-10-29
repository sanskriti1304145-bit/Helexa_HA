"""
csv_helpers.py - small helpers for reading and writing the CSV files safely.

Every tool uses these two functions instead of calling pandas directly,
so all CSV files are read and written the same way.
"""

import os
from pathlib import Path

import pandas as pd


def read_csv(path):
    """Read a CSV file and return a pandas DataFrame where every value is text.

    - dtype=str keeps IDs like "P001" and values like "70" as plain text.
    - keep_default_na=False stops pandas from turning words such as "None"
      (used in the lab_results column) into empty/missing values.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(df, path):
    """Save a DataFrame to a CSV file without risking a half-written file.

    We first write to a temporary file, then swap it into place in one step.
    If the program crashes while writing, the original file is still intact.
    """
    path = Path(path)
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, path)


def next_id(existing_ids, prefix):
    """Create the next ID in a series, e.g. ["R001", "R023"] -> "R024".

    Takes the largest number after the prefix and adds 1.
    """
    numbers = [int(i[len(prefix):]) for i in existing_ids if i.startswith(prefix)]
    next_number = max(numbers, default=0) + 1
    return f"{prefix}{next_number:03d}"


def add_missing_columns(path, columns_after):
    """Upgrade an older CSV file: add any missing column (blank for existing rows).

    columns_after: {"new_column": "column it goes after"} - the new column is placed
    right after that one (or at the end if the other column is missing).
    Returns the list of columns that were added. Existing values are never changed.
    """
    df = read_csv(path)
    added = []
    for column, after in columns_after.items():
        if column not in df.columns:
            position = df.columns.get_loc(after) + 1 if after in df.columns else len(df.columns)
            df.insert(position, column, "")
            added.append(column)
    if added:
        write_csv(df, path)
    return added

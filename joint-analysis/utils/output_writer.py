"""
output_writer.py

File-output helpers for the analysis run. Everything here writes to the
directories defined in configs.path_configs, creating them if needed, and
prints the path that was written.
"""

import os
import pandas as pd

from configs.path_configs import OUTPUT_CSV_DIR, OUTPUT_PRINT_DIR

def save_csv_to_output_dir(data: pd.DataFrame, filename: str) -> None:
    """Save the given DataFrame to the analysis output directory with the specified filename."""
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_CSV_DIR, filename)
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def save_print_to_output_dir(print_output: str, filename: str) -> None:
    """Save all print output to a file in the analysis output directory."""
    os.makedirs(OUTPUT_PRINT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_PRINT_DIR, filename)
    with open(output_path, "w") as f:
        f.write(print_output)
    print(f"Print output saved to {output_path}")
"""
Plot a TensorBoard-exported scalar CSV (columns: Wall time, Step, Value).

HOW TO USE:
Just edit the settings below, then run:  python plot_tb_csv_simple.py

This script can live anywhere. What matters is the folder you RUN it FROM
(your current directory) -- that folder must contain "tb_charts_data/".
A "tb_plots/" folder will be created there too, next to "tb_charts_data/".

Example (Windows/PowerShell):
    cd E:\\Claude
    python path\\to\\plot_tb_csv_simple.py
"""

import csv
import os

import matplotlib.pyplot as plt

# ======================= EDIT THESE ===========================

# Filename inside tb_charts_data/ (just the CSV name, no path needed)
FILENAME = "lagrangian-mean_cost_duty_rate_inflate.csv"
OUTPUT_NAME = "mean_cost_duty_rate_inflate"
TITLE = "Mean Inflate Duty-Rate Cost vs. Limit"
XLABEL = "Training Step"
YLABEL = "Mean Cost"
HLINE = 0.3
HLINE_LABEL = "Limit = 0.3"
# ================================================================

BASE_DIR = os.getcwd()  # the folder you run this script FROM (not where the .py file lives)
DATA_DIR = os.path.join(BASE_DIR, "tb_charts_data")
OUT_DIR = os.path.join(BASE_DIR, "tb_plots")


def load_csv(path):
    steps, values = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["Step"]))
            values.append(float(row["Value"]))
    return steps, values


def main():
    csv_path = os.path.join(DATA_DIR, FILENAME)
    steps, values = load_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, values, linewidth=1.8)

    if HLINE is not None:
        ax.axhline(y=HLINE, linestyle="--", color="gray", linewidth=1.2,
                   label=HLINE_LABEL if HLINE_LABEL else f"ref = {HLINE}")
        ax.legend()

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, OUTPUT_NAME + ".png")
    pdf_path = os.path.join(OUT_DIR, OUTPUT_NAME + ".pdf")
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    

if __name__ == "__main__":
    main()
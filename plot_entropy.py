#!/usr/bin/env python3
"""
plot_entropy.py
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

CSV = Path("model_comparison.csv")
IMG = Path("images"); IMG.mkdir(exist_ok=True)

# levels to show and order
KEEP = ["char-1", "char-2 (joint)", "char-3 (joint)", "word-1", "word-2 (joint)", "word-3 (joint)"]

def main():
    rows = []
    with open(CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["level"] in KEEP:
                rows.append(row)

    authors = sorted({r["author"] for r in rows})
    levels = KEEP

    # build matrix: entropy[author][level]
    data = {a: {lvl: None for lvl in levels} for a in authors}
    for r in rows:
        a, lvl = r["author"], r["level"]
        data[a][lvl] = float(r["entropy_bits"])

    # plot grouped bars
    import numpy as np
    x = np.arange(len(levels))
    width = 0.25

    plt.figure(figsize=(10,5))
    for i, a in enumerate(authors):
        y = [data[a][lvl] for lvl in levels]
        plt.bar(x + i*width - width, y, width, label=a.title())

    plt.xticks(x, levels, rotation=20)
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy by Author and Approximation Level")
    plt.legend()
    plt.tight_layout()
    out = IMG / "entropy_comparison.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_zipf.py
"""

import json, math
from pathlib import Path
import matplotlib.pyplot as plt  
from typing import Dict

AUTHORS = ["austen", "twain", "doyle"]
OUT = Path("output")
IMG = Path("images"); IMG.mkdir(exist_ok=True)

def load_word_unigrams(author: str) -> Dict[str, float]:
    p = OUT / author / "stats" / "word_ngrams.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # our analyze.py saves {"counts":..., "probs":...}
    return data["unigram"]["probs"]

def plot_zipf(author: str, probs: Dict[str, float]):
    # sort by prob desc, compute ranks
    vals = sorted([v for v in probs.values() if v > 0], reverse=True)
    ranks = range(1, len(vals)+1)

    plt.figure()
    plt.loglog(ranks, vals, marker=".", linestyle="none")
    plt.title(f"Zipf plot (word unigrams) — {author.title()}")
    plt.xlabel("Rank (log)")
    plt.ylabel("Probability (log)")
    plt.tight_layout()
    out = IMG / f"zipf_{author}_word1.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

def main():
    for a in AUTHORS:
        probs = load_word_unigrams(a)
        plot_zipf(a, probs)

if __name__ == "__main__":
    main()

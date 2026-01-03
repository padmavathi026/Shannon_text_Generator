#!/usr/bin/env python3
"""
sample.py
Generate comprehensive sample outputs for all required CLI options and combinations.
Run:
    python sample.py
"""

import sys
import subprocess
from pathlib import Path
from itertools import combinations

PY = sys.executable  # current interpreter

# Source texts for analysis
AUTHORS = {
    "austen": "austen_pride_prejudice.txt",
    "twain":  "twain_tom_sawyer.txt",
    "doyle":  "doyle_sherlock_holmes.txt",
}

# Character lengths per order for readable samples
CHAR_LEN = {0: 300, 1: 400, 2: 400, 3: 600}

# Sentence counts per word order
WORD_SENTENCES = {1: 5, 2: 5, 3: 5}

# Anchor sets per author (assignment-friendly)
ANCHORS = {
    "austen": "love,marriage",
    "twain":  "tom,huck,river",
    "doyle":  "elementary,Watson,deduce",   # case preserved; generator lowercases internally
}

SAMPLES_DIR = Path("samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd, stdout_file: Path | None = None):
    """Run a subprocess command, optionally redirecting stdout to a file. Raises on failure."""
    print(">>", " ".join(cmd))
    if stdout_file is None:
        subprocess.run(cmd, check=True)
    else:
        with open(stdout_file, "w", encoding="utf-8") as fh:
            subprocess.run(cmd, check=True, stdout=fh, text=True)

def analyze_all():
    print("\n### Analyze: building frequency tables (Parts 1–2)")
    for author, fname in AUTHORS.items():
        run([PY, "shannon_gen.py", "analyze", "--author", author, "--file", fname])

def generate_character_levels():
    print("\n### Character-level generation for all authors (char-0..char-3)")
    for author in AUTHORS.keys():
        for order in (0, 1, 2, 3):
            out_path = SAMPLES_DIR / f"{author}_char{order}.txt"
            run([PY, "shannon_gen.py", "generate",
                 "--author", author,
                 "--level", f"char -{order}",
                 "--length", str(CHAR_LEN[order]),
                 "--out", str(out_path)])

def generate_word_levels():
    print("\n### Word-level generation for all authors (word-1..word-3)")
    for author in AUTHORS.keys():
        for order in (1, 2, 3):
            out_path = SAMPLES_DIR / f"{author}_word{order}.txt"
            run([PY, "shannon_gen.py", "generate",
                 "--author", author,
                 "--level", f"word -{order}",
                 "--sentences", str(WORD_SENTENCES[order]),
                 "--out", str(out_path)])

def generate_word_levels_with_anchors():
    print("\n### With anchor words for all authors (word-1..word-3)")
    for author, anchor_str in ANCHORS.items():
        for order in (1, 2, 3):
            out_path = SAMPLES_DIR / f"{author}_word{order}_anchors.txt"
            run([PY, "shannon_gen.py", "generate",
                 "--author", author,
                 "--level", f"word -{order}",
                 "--sentences", str(WORD_SENTENCES[order]),
                 "--anchors", anchor_str,
                 "--out", str(out_path)])

def generate_compare_outputs():
    print("\n### Compare approximation levels (per author)")
    for author in AUTHORS.keys():
        out_path = SAMPLES_DIR / f"{author}_compare.txt"
        run([PY, "shannon_gen.py", "compare", "--author", author], stdout_file=out_path)

def generate_blends():
    print("\n### BONUS: Blend styles for all author pairs (word-2 & word-3, with/without anchors)")
    pairs = list(combinations(AUTHORS.keys(), 2))  # all unique 2-author pairs
    for a1, a2 in pairs:
        # word-2 (exact filename for austen+twain as requested)
        if (a1, a2) == ("austen", "twain") or (a1, a2) == ("twain", "austen"):
            out2 = SAMPLES_DIR / "blend_austen_twain_word2_3s.txt"
        else:
            out2 = SAMPLES_DIR / f"blend_{a1}_{a2}_word2.txt"
        run([PY, "shannon_gen.py", "blend",
             "--authors", f"{a1},{a2}",
             "--level", "word-2",
             "--sentences", "3",
             "--out", str(out2)])

        # word-3
        out3 = SAMPLES_DIR / f"blend_{a1}_{a2}_word3.txt"
        run([PY, "shannon_gen.py", "blend",
             "--authors", f"{a1},{a2}",
             "--level", "word-3",
             "--sentences", "3",
             "--out", str(out3)])

        # with anchors (use anchors of the first author if defined)
        a1_anchors = ANCHORS.get(a1, "")
        if a1_anchors:
            out2a = SAMPLES_DIR / f"blend_{a1}_{a2}_word2_anchors.txt"
            run([PY, "shannon_gen.py", "blend",
                 "--authors", f"{a1},{a2}",
                 "--level", "word-2",
                 "--sentences", "3",
                 "--anchors", a1_anchors,
                 "--out", str(out2a)])

            out3a = SAMPLES_DIR / f"blend_{a1}_{a2}_word3_anchors.txt"
            run([PY, "shannon_gen.py", "blend",
                 "--authors", f"{a1},{a2}",
                 "--level", "word-3",
                 "--sentences", "3",
                 "--anchors", a1_anchors,
                 "--out", str(out3a)])

def main():
    analyze_all()                       # Parts 1–2
    generate_character_levels()         # char-0..char-3
    generate_word_levels()              # word-1..word-3
    generate_word_levels_with_anchors() # anchors for word-1..word-3
    generate_compare_outputs()          # compare per author
    generate_blends()                   # bonus blends
    print("\nAll samples written to ./samples/ (existing files were overwritten).")

if __name__ == "__main__":
    main()

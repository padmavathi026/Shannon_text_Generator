#!/usr/bin/env python3
"""
build_reports.py
Generate course deliverables from existing analysis + generator:
- shannon_analysis_report.json
- model_comparison.csv
- generated_samples.txt
- style_examples.json

Requirements:
- output/<author>/stats/{word_ngrams.json,char_ngrams.json,sentence_stats.json}
- shannon_gen.py CLI available in the same environment

Run:
    python build_reports.py
"""

import os
import json
import math
import csv
import subprocess
import sys
from pathlib import Path

PY = sys.executable

AUTHORS = ["austen", "twain", "doyle"]
ROOT = Path(".")
OUTDIR = ROOT / "output"
SAMPLEDIR = ROOT / "samples"

# choose representative levels for the writeups
REP_CHAR = {"austen": 2, "twain": 2, "doyle": 3}   # char -2/-3
REP_WORD = {"austen": 3, "twain": 3, "doyle": 2}   # word -2/-3

# short anchors for style demos
ANCHORS = {
    "austen": "love,marriage",
    "twain":  "tom,huck,river",
    "doyle":  "elementary,Watson,deduce",
}

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def entropy_from_probs(prob_dict) -> float:
    """
    Compute Shannon entropy H = -sum p log2 p.
    prob_dict can have string keys; values must be probabilities summing ~1.
    Silently ignores zero/nonpositive entries.
    """
    H = 0.0
    total = sum(prob_dict.values()) or 1.0
    for p in prob_dict.values():
        p = p / total
        if p > 0.0:
            H -= p * math.log2(p)
    return H

def joint_entropy_from_ngram_probs(ngram_probs: dict) -> float:
    """
    Approx joint-entropy over n-gram distribution (not conditional).
    Good enough for a single comparative metric across orders.
    """
    return entropy_from_probs(ngram_probs)

def read_word_models(author: str):
    stats_dir = OUTDIR / author / "stats"
    wfile = stats_dir / "word_ngrams.json"
    sfile = stats_dir / "sentence_stats.json"
    data_w = load_json(wfile)
    data_s = load_json(sfile)
    return data_w, data_s

def read_char_models(author: str):
    stats_dir = OUTDIR / author / "stats"
    cfile = stats_dir / "char_ngrams.json"
    return load_json(cfile)

def parse_prob_block(block: dict) -> dict:
    """
    our analyze.py saved: {"counts": {...}, "probs": {...}}
    Return the probs dict (string keys).
    """
    return block.get("probs", {})

def ensure_samples_dir():
    SAMPLEDIR.mkdir(parents=True, exist_ok=True)

def run_cmd(args, stdout_path: Path | None = None):
    print(">>", " ".join(args))
    if stdout_path is None:
        subprocess.run(args, check=True)
    else:
        with open(stdout_path, "w", encoding="utf-8") as f:
            subprocess.run(args, check=True, stdout=f, text=True)

def main():
    ensure_samples_dir()

    # 1) Build shannon_analysis_report.json
    report = {"authors": {}}
    for a in AUTHORS:
        report["authors"][a] = {}
        # word stats
        wdata, sdata = read_word_models(a)
        wu = parse_prob_block(wdata["unigram"])
        wb = parse_prob_block(wdata["bigram"])
        wt = parse_prob_block(wdata["trigram"])

        H_w1 = entropy_from_probs(wu)
        H_w2 = joint_entropy_from_ngram_probs(wb)
        H_w3 = joint_entropy_from_ngram_probs(wt)

        report["authors"][a]["word"] = {
            "unigram": {
                "vocab_size": len(wu),
                "entropy_bits": round(H_w1, 6),
                "perplexity": round(2 ** H_w1, 6),
            },
            "bigram": {
                "ngram_count": len(wb),
                "joint_entropy_bits": round(H_w2, 6),
                "perplexity_joint": round(2 ** H_w2, 6),
            },
            "trigram": {
                "ngram_count": len(wt),
                "joint_entropy_bits": round(H_w3, 6),
                "perplexity_joint": round(2 ** H_w3, 6),
            },
            "sentence_stats": {
                "mean": sdata.get("mean", 0.0),
                "std": sdata.get("std", 0.0),
            }
        }

        # char stats
        cdata = read_char_models(a)
        cu = parse_prob_block(cdata["unigram"])
        cb = parse_prob_block(cdata["bigram"])
        ct = parse_prob_block(cdata["trigram"])

        H_c1 = entropy_from_probs(cu)
        H_c2 = joint_entropy_from_ngram_probs(cb)
        H_c3 = joint_entropy_from_ngram_probs(ct)

        report["authors"][a]["char"] = {
            "unigram": {
                "vocab_size": len(cu),
                "entropy_bits": round(H_c1, 6),
                "perplexity": round(2 ** H_c1, 6),
            },
            "bigram": {
                "ngram_count": len(cb),
                "joint_entropy_bits": round(H_c2, 6),
                "perplexity_joint": round(2 ** H_c2, 6),
            },
            "trigram": {
                "ngram_count": len(ct),
                "joint_entropy_bits": round(H_c3, 6),
                "perplexity_joint": round(2 ** H_c3, 6),
            }
        }

    with open("shannon_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 2) Build model_comparison.csv (flat table)
    rows = []
    for a in AUTHORS:
        wdata, _ = read_word_models(a)
        cdata = read_char_models(a)

        wu = parse_prob_block(wdata["unigram"])
        wb = parse_prob_block(wdata["bigram"])
        wt = parse_prob_block(wdata["trigram"])
        cu = parse_prob_block(cdata["unigram"])
        cb = parse_prob_block(cdata["bigram"])
        ct = parse_prob_block(cdata["trigram"])

        # word
        for level, probs in [
            ("word-1", wu),
            ("word-2 (joint)", wb),
            ("word-3 (joint)", wt),
        ]:
            H = entropy_from_probs(probs)
            rows.append({
                "author": a,
                "level": level,
                "unit": "word",
                "ngrams": len(probs),
                "entropy_bits": H,
                "perplexity": 2 ** H,
            })

        # char
        for level, probs in [
            ("char-1", cu),
            ("char-2 (joint)", cb),
            ("char-3 (joint)", ct),
        ]:
            H = entropy_from_probs(probs)
            rows.append({
                "author": a,
                "level": level,
                "unit": "char",
                "ngrams": len(probs),
                "entropy_bits": H,
                "perplexity": 2 ** H,
            })

    with open("model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 3) generated_samples.txt — representative runs
    #    Rebuild specific examples via shannon_gen.py so file is fresh.
    parts = []
    for a in AUTHORS:
        # char rep
        o_char = REP_CHAR[a]
        parts.append(f"## {a.title()} — char -{o_char}\n")
        tmp = SAMPLEDIR / f"{a}_char{o_char}_rep.txt"
        run_cmd([PY, "shannon_gen.py", "generate",
                 "--author", a, "--level", f"char -{o_char}",
                 "--length", "300"], stdout_path=tmp)
        parts.append(tmp.read_text(encoding="utf-8").strip() + "\n\n")

        # word rep
        o_word = REP_WORD[a]
        parts.append(f"## {a.title()} — word -{o_word}\n")
        tmp = SAMPLEDIR / f"{a}_word{o_word}_rep.txt"
        run_cmd([PY, "shannon_gen.py", "generate",
                 "--author", a, "--level", f"word -{o_word}",
                 "--sentences", "5"], stdout_path=tmp)
        parts.append(tmp.read_text(encoding="utf-8").strip() + "\n\n")

    with open("generated_samples.txt", "w", encoding="utf-8") as f:
        f.write("# Generated Samples (Representative)\n\n")
        f.writelines(parts)

    # 4) style_examples.json — short, consistent snippets per author/level
    style_examples = {"examples": []}
    for a in AUTHORS:
        for order in (1, 2, 3):
            # 2 sentences for word-level styles
            tmp = SAMPLEDIR / f"{a}_word{order}_mini.txt"
            run_cmd([PY, "shannon_gen.py", "generate",
                     "--author", a, "--level", f"word -{order}",
                     "--sentences", "2"], stdout_path=tmp)
            snippet = tmp.read_text(encoding="utf-8").strip()
            style_examples["examples"].append({
                "author": a,
                "level": f"word-{order}",
                "text": snippet
            })

        # anchors demo (word-3, except twain word-2 is often nicer)
        if a == "twain":
            order = 2
        else:
            order = 3
        tmp = SAMPLEDIR / f"{a}_word{order}_anchors_demo.txt"
        run_cmd([PY, "shannon_gen.py", "generate",
                 "--author", a,
                 "--level", f"word -{order}",
                 "--sentences", "3",
                 "--anchors", ANCHORS[a]], stdout_path=tmp)
        style_examples["examples"].append({
            "author": a,
            "level": f"word-{order}",
            "anchors": ANCHORS[a],
            "text": tmp.read_text(encoding="utf-8").strip()
        })

    with open("style_examples.json", "w", encoding="utf-8") as f:
        json.dump(style_examples, f, indent=2)

    print("\nArtifacts written:")
    print(" - shannon_analysis_report.json")
    print(" - model_comparison.csv")
    print(" - generated_samples.txt")
    print(" - style_examples.json")

if __name__ == "__main__":
    main()

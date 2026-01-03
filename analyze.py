#!/usr/bin/env python3
"""
Assignment: Approximating Natural Language — Parts 1 & 2

This script fulfills:
• Part 1: Data Collection & Preparation
    - Uses TextPreprocessor to clean, normalize, and tokenize each provided novel.
    - Saves transparent intermediate artifacts so you can verify preprocessing.
• Part 2: Statistical Analysis
    - Uses FrequencyAnalyzer to compute character and word n-gram frequencies
      (unigram, bigram, trigram) and converts counts to probabilities.
    - Computes sentence length distribution, mean, and standard deviation.

 
"""

import os
import json
import argparse
from statistics import mean, pstdev
from typing import Dict, List, Any

# Import exactly as required by the assignment
from starter_preprocess import TextPreprocessor, FrequencyAnalyzer


# Generic utilities (IO, formatting) 

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def load_text(path: str) -> str:
    """Read a UTF-8 text file into memory."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def stringify_keys(d: Dict[Any, int]) -> Dict[str, int]:
    """
    Convert n-gram dictionary keys to JSON-safe strings.
    Tuple keys (e.g., bigrams/trigrams).
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, tuple):
            out["|||".join(map(str, k))] = v
        else:
            out[str(k)] = v
    return out

def counts_to_probs(counts: Dict[Any, int], smoothing: float = 0.0) -> Dict[Any, float]:
    """
    Convert n-gram counts to probabilities (simple MLE with optional Laplace smoothing).
    Returned as a mapping with the same keys as counts.
    """
    total = sum(counts.values())
    if total == 0:
        return {}
    vocab = len(counts)
    denom = total + smoothing * vocab
    return {k: (c + smoothing) / denom for k, c in counts.items()}


# Part 1: Data Preparation  

def preprocess_text(file_path: str, out_dir: str, preserve_sentences: bool = True) -> Dict[str, Any]:
    """
    Part 1 deliverable:
      1) Read raw text
      2) Clean text 
      3) Normalize text  
      4) Tokenize 
      5) Save artifacts 

    Returns token lists for downstream Part 2 calculations.
    """
    tp = TextPreprocessor()

    # Step 1: read
    raw = load_text(file_path)

    # Step 2: clean (starter code encapsulates header/footer and basic artifacts removal)
    cleaned = tp.clean_gutenberg_text(raw)

    # Step 3: normalize (preserve punctuation needed for sentence boundaries)
    normalized = tp.normalize_text(cleaned, preserve_sentences=preserve_sentences)

    # Step 4: tokenize
    sentences: List[str] = tp.tokenize_sentences(normalized)
    words: List[str]     = tp.tokenize_words(normalized)
    chars: List[str]     = tp.tokenize_chars(normalized, include_space=True)  # include spaces per spec

    # Step 5: save artifacts
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "cleaned.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned)
    with open(os.path.join(out_dir, "normalized.txt"), "w", encoding="utf-8") as f:
        f.write(normalized)
    with open(os.path.join(out_dir, "sentences.json"), "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "words.json"), "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "chars.json"), "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    # Quick summary to confirm preprocessing scale and uniqueness
    summary = {
        "num_sentences": len(sentences),
        "num_words": len(words),
        "num_chars_including_space": len(chars),
        "unique_words": len(set(words)),
        "unique_chars": len(set(chars)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {"sentences": sentences, "words": words, "chars": chars}


# Part 2: Statistical Analysis

def build_char_stats(chars: List[str], analyzer: FrequencyAnalyzer) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    4.1 Character-level analysis:
      - Unigram, bigram, trigram counts via FrequencyAnalyzer.calculate_ngrams
      - Convert counts to probabilities for completeness
    """
    c1 = analyzer.calculate_ngrams(chars, 1)
    c2 = analyzer.calculate_ngrams(chars, 2)
    c3 = analyzer.calculate_ngrams(chars, 3)
    return {
        "unigram": {"counts": stringify_keys(c1), "probs": stringify_keys(counts_to_probs(c1))},
        "bigram":  {"counts": stringify_keys(c2), "probs": stringify_keys(counts_to_probs(c2))},
        "trigram": {"counts": stringify_keys(c3), "probs": stringify_keys(counts_to_probs(c3))},
    }

def build_word_stats(words: List[str], analyzer: FrequencyAnalyzer) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    4.2 Word-level analysis:
      - Unigram, bigram, trigram counts via FrequencyAnalyzer.calculate_ngrams
      - Convert counts to probabilities for completeness
    """
    w1 = analyzer.calculate_ngrams(words, 1)
    w2 = analyzer.calculate_ngrams(words, 2)
    w3 = analyzer.calculate_ngrams(words, 3)
    return {
        "unigram": {"counts": stringify_keys(w1), "probs": stringify_keys(counts_to_probs(w1))},
        "bigram":  {"counts": stringify_keys(w2), "probs": stringify_keys(counts_to_probs(w2))},
        "trigram": {"counts": stringify_keys(w3), "probs": stringify_keys(counts_to_probs(w3))},
    }

def build_sentence_stats(sentences: List[str], tp: TextPreprocessor) -> Dict[str, Any]:
    """
    4.3 Sentence structure 
    """
    lengths = tp.get_sentence_lengths(sentences)
    if not lengths:
        return {"lengths": [], "mean": 0.0, "std": 0.0}
    return {"lengths": lengths, "mean": float(mean(lengths)), "std": float(pstdev(lengths))}


# Driver: one-author pipeline 

def analyze_author(author: str, file_path: str, out_root: str = "./output") -> None:
    """
    Orchestrates both parts for a single author.

    Outputs:
      ./output/<author>/prep/
          cleaned.txt
          normalized.txt
          sentences.json
          words.json
          chars.json
          summary.json
      ./output/<author>/stats/
          char_ngrams.json
          word_ngrams.json
          sentence_stats.json
    """
    author_dir = os.path.join(out_root, author.lower())
    prep_dir   = os.path.join(author_dir, "prep")
    stats_dir  = os.path.join(author_dir, "stats")
    ensure_dir(author_dir)
    ensure_dir(prep_dir)
    ensure_dir(stats_dir)

    # Part 1: preprocessing
    tokens = preprocess_text(file_path, prep_dir, preserve_sentences=True)
    sentences, words, chars = tokens["sentences"], tokens["words"], tokens["chars"]

    # Part 2: analysis
    fa = FrequencyAnalyzer()
    char_stats = build_char_stats(chars, fa)
    word_stats = build_word_stats(words, fa)
    sent_stats = build_sentence_stats(sentences, TextPreprocessor())

    with open(os.path.join(stats_dir, "char_ngrams.json"), "w", encoding="utf-8") as f:
        json.dump(char_stats, f, ensure_ascii=False)
    with open(os.path.join(stats_dir, "word_ngrams.json"), "w", encoding="utf-8") as f:
        json.dump(word_stats, f, ensure_ascii=False)
    with open(os.path.join(stats_dir, "sentence_stats.json"), "w", encoding="utf-8") as f:
        json.dump(sent_stats, f, ensure_ascii=False)

    print(f"[DONE] {author}: Part 1 → {prep_dir} | Part 2 → {stats_dir}")


# CLI 

def main():
    parser = argparse.ArgumentParser(
        description="Parts 1 & 2: preprocessing and statistical analysis for three classic authors."
    )
    parser.add_argument("--author", type=str, help="austen | twain | doyle")
    parser.add_argument("--file", type=str, help="path to the .txt for the author")
    parser.add_argument("--all", action="store_true", help="process all three provided texts")
    parser.add_argument("--out", type=str, default="./output", help="output root directory")
    args = parser.parse_args()

    if args.all:
        jobs = [
            ("austen", "austen_pride_prejudice.txt"),
            ("twain",  "twain_tom_sawyer.txt"),
            ("doyle",  "doyle_sherlock_holmes.txt"),
        ]
        for a, p in jobs:
            analyze_author(a, p, out_root=args.out)
    else:
        if not args.author or not args.file:
            raise SystemExit("Provide --author and --file, or use --all.")
        analyze_author(args.author, args.file, out_root=args.out)

if __name__ == "__main__":
    main()

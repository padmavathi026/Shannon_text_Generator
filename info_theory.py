#!/usr/bin/env python3
# info_theory.py
import math
import numpy as np

def demonstrate_shannon_concepts():
    probs = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    H = -sum(p * math.log2(p) for p in probs.values())
    PP = 2 ** H
    print(f"Entropy: {H:.3f} bits | Perplexity: {PP:.3f}")

class InformationAnalyzer:
    def calculate_entropy(self, probs: dict) -> float:
        total = sum(probs.values()) or 1.0
        H = 0.0
        for p in probs.values():
            p = p / total
            if p > 0:
                H -= p * math.log2(p)
        return H

    def calculate_perplexity(self, entropy_bits: float) -> float:
        return 2 ** entropy_bits

    def analyze_zipf_distribution(self, freq: dict):
        if not freq:
            return {'alpha': 0.0, 'r_squared': 0.0}
        vals = np.array(sorted(freq.values(), reverse=True), dtype=float)
        vals = vals[vals > 0]
        if len(vals) < 2:
            return {'alpha': 0.0, 'r_squared': 0.0}
        ranks = np.arange(1, len(vals) + 1, dtype=float)
        x = np.log(ranks)
        y = np.log(vals / vals.sum())
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        y_hat = a + b * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        alpha = float(-b)  # slope ~ -alpha
        return {'alpha': alpha, 'r_squared': float(r2)}

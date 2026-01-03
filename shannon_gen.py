#!/usr/bin/env python3
"""
shannon_gen.py — Part 4: Command-Line Interface
"""

import os
import re
import json
import argparse
import random
from typing import Any, Dict, List, Tuple, Optional

# Import your existing code
from analyze import analyze_author
from generator import TextGenerator

# small utilities & parsing 

def die(msg: str):
    raise SystemExit(f"[ERROR] {msg}")

def ensure_exists(path: str, desc: str):
    if not os.path.exists(path):
        die(f"Missing {desc}: {path}")

def parse_level(level_str: str) -> Tuple[str, int]:
    """
    Returns (kind, order) where kind in {"char","word"} and order in {0,1,2,3}.
    """
    s = level_str.strip().lower()
    s = s.replace(" ", "")          # tolerate "char -2"
    s = s.replace("_", "")          # be forgiving
    m = re.match(r"^(char|word)(-?)([0-3])$", s)
    if m:
        return m.group(1), int(m.group(3))
    if s in ("zerochar", "zero", "0", "char0", "char-0"):
        return "char", 0
    if s in ("char1english", "char1", "char-1"):
        return "char", 1
    if s in ("char2", "char-2"):
        return "char", 2
    if s in ("char3", "char-3"):
        return "char", 3
    if s in ("word1", "word-1"):
        return "word", 1
    if s in ("word2", "word-2"):
        return "word", 2
    if s in ("word3", "word-3"):
        return "word", 3
    die(f"Unrecognized --level '{level_str}'. Use char-0..char-3 or word-1..word-3.")

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# minimal blending (BONUS): combine word models 

def _string_to_tuple_key(k: str) -> Tuple[str, ...]:
    return tuple(k.split("|||"))

def _flatten_prob_json(prob_or_counts: Dict[str, Any]) -> Dict[Any, float | int]:
    out = {}
    for k, v in prob_or_counts.items():
        if "|||" in k:
            out[_string_to_tuple_key(k)] = v
        else:
            out[k] = v
    return out

def _sum_dicts(a: Dict[Any, float | int], b: Dict[Any, float | int]) -> Dict[Any, float | int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out

def _cdf_from_counts(counts: Dict[Any, float | int]) -> List[Tuple[Any, float]]:
    total = float(sum(counts.values()))
    if total <= 0:
        return []
    cume = 0.0
    items = []
    for k, v in counts.items():
        cume += float(v) / total
        items.append((k, cume))
    return items

def _sample_from_cdf(cdf: List[Tuple[Any, float]]) -> Any:
    if not cdf: return None
    r = random.random()
    for k, c in cdf:
        if r <= c:
            return k
    return cdf[-1][0]

def _build_conditional_cdfs(ng_counts: Dict[Any, float | int], order: int) -> Dict[Tuple[Any, ...], List[Tuple[Any, float]]]:
    from collections import defaultdict
    cond = defaultdict(lambda: {})
    for ng, v in ng_counts.items():
        if not isinstance(ng, tuple) or len(ng) != order:
            continue
        ctx = ng[:-1]
        nx  = ng[-1]
        cond[ctx][nx] = cond[ctx].get(nx, 0) + v
    return {ctx: _cdf_from_counts(nx_counts) for ctx, nx_counts in cond.items()}

def _choose_seed_context(cdfs: Dict[Tuple[Any, ...], List[Tuple[Any, float]]]) -> Tuple[Any, ...]:
    if not cdfs: return tuple()
    import random
    return random.choice(list(cdfs.keys()))

def generate_blended_word(author_a_root: str, author_b_root: str, model_root: str,
                          order: int, n_sentences: int, anchors: Optional[List[str]] = None) -> str:
    """
    Blend two authors’ word models by summing counts, then generate sentence-aware text.
    """
    assert order in (2, 3)

    a_w = load_json(os.path.join(model_root, author_a_root, "stats", "word_ngrams.json"))
    b_w = load_json(os.path.join(model_root, author_b_root, "stats", "word_ngrams.json"))
    a_s = load_json(os.path.join(model_root, author_a_root, "stats", "sentence_stats.json"))
    b_s = load_json(os.path.join(model_root, author_b_root, "stats", "sentence_stats.json"))

    a_u = _flatten_prob_json(a_w["unigram"]["counts"])
    a_bi = _flatten_prob_json(a_w["bigram"]["counts"])
    a_tri = _flatten_prob_json(a_w["trigram"]["counts"])
    b_u = _flatten_prob_json(b_w["unigram"]["counts"])
    b_bi = _flatten_prob_json(b_w["bigram"]["counts"])
    b_tri = _flatten_prob_json(b_w["trigram"]["counts"])

    uni = _sum_dicts(a_u, b_u)
    bi  = _sum_dicts(a_bi, b_bi)
    tri = _sum_dicts(a_tri, b_tri)

    # Build CDFs
    unigram_cdf = _cdf_from_counts(uni)
    cond2 = _build_conditional_cdfs(bi, 2)
    cond3 = _build_conditional_cdfs(tri, 3)

    # blended sentence lengths: concat both distributions
    lengths = (a_s.get("lengths", []) or []) + (b_s.get("lengths", []) or [])
    if not lengths: lengths = [12]

    def sample_len() -> int:
        import random
        L = random.choice(lengths)
        return max(1, int(L))

    def sample_word_from(cdf): return _sample_from_cdf(cdf)

    def generate_sentence():
        import random
        if order == 2:
            ctx = _choose_seed_context(cond2)
            if not ctx:
                # fallback to unigram sentence
                L = sample_len()
                return " ".join(sample_word_from(unigram_cdf) for _ in range(L)).capitalize() + "."
            words = [ctx[0], _sample_from_cdf(cond2.get(ctx) or unigram_cdf)]
        else:
            ctx = _choose_seed_context(cond3)
            if not ctx:
                L = sample_len()
                return " ".join(sample_word_from(unigram_cdf) for _ in range(L)).capitalize() + "."
            words = [ctx[0], ctx[1], _sample_from_cdf(cond3.get(ctx) or cond2.get((ctx[1],), None) or unigram_cdf)]

        L = sample_len()
        while len(words) < L:
            if order == 2:
                ctx = (words[-1],)
                cdf = cond2.get(ctx) or unigram_cdf
            else:
                ctx = (words[-2], words[-1])
                cdf = cond3.get(ctx) or cond2.get((words[-1],), None) or unigram_cdf
            words.append(_sample_from_cdf(cdf))
        words[0] = words[0].capitalize()
        # Simple punctuation
        end = "." if random.random() < 0.85 else ("!" if random.random() < 0.5 else "?")
        return " ".join(words) + end

    anchors = [a.strip().lower() for a in (anchors or []) if a.strip()]
    # multiple attempts to include anchors naturally
    for _ in range(20 if anchors else 1):
        sents = [generate_sentence() for _ in range(n_sentences)]
        text_words_lower = " ".join(sents).lower().split()
        if not anchors or all(a in text_words_lower for a in anchors):
            return " ".join(sents)

    # force-insert missing anchors
    sents = []
    sentences_words = []
    for _ in range(n_sentences):
        s = generate_sentence()
        sents.append(s)
        sentences_words.append(s[:-1].split())  # strip last punct
    have = set(w.lower() for sent in sentences_words for w in sent)
    need = [a for a in anchors if a not in have]
    import random
    for a in need:
        si = random.randrange(len(sentences_words))
        if sentences_words[si]:
            wi = random.randrange(len(sentences_words[si]))
            sentences_words[si][wi] = a
    puncted = []
    for words in sentences_words:
        if words:
            words[0] = words[0].capitalize()
        puncted.append(" ".join(words) + ".")
    return " ".join(puncted)

# CLI logic 

def cmd_analyze(args):
    # Build frequency tables for one author
    if not args.author or not args.file:
        die("Use: shannon_gen.py analyze --author <austen|twain|doyle> --file <path>")
    analyze_author(args.author, args.file, out_root=args.out)

def cmd_generate(args):
    kind, order = parse_level(args.level)
    tg = TextGenerator(args.model_root, args.author)

    if kind == "char":
        if order == 0:
            text = tg.generate_zero_char(args.length)
        elif order == 1:
            text = tg.generate_char1_english(args.length)
        elif order in (2, 3):
            text = tg.generate_char_markov(args.length, order=order)
        else:
            die("char level must be 0–3")
    else:
        if order not in (1, 2, 3):
            die("word level must be 1–3")
        anchors = [a.strip() for a in (args.anchors or "").split(",") if a.strip()]
        text = tg.generate_sentences_word_markov(args.sentences, order=order, anchors=anchors or None)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] wrote → {args.out}")
    else:
        print(text)

def cmd_compare(args):
    """
    Compare multiple levels for the same author.
    """
    tg = TextGenerator(args.model_root, args.author)
    print("# char-2")
    print(tg.generate_char_markov(120, order=2)); print()
    print("# char-3")
    print(tg.generate_char_markov(120, order=3)); print()
    print("# word-1")
    print(tg.generate_sentences_word_markov(1, order=1)); print()
    print("# word-2")
    print(tg.generate_sentences_word_markov(1, order=2)); print()
    print("# word-3")
    print(tg.generate_sentences_word_markov(1, order=3)); print()

def cmd_blend(args):
    """
    BONUS: Blend two authors’ word models (order 2 or 3).
    """
    kind, order = parse_level(args.level)
    if kind != "word" or order not in (2, 3):
        die("blend supports word-2 or word-3 only")
    authors = [a.strip().lower() for a in args.authors.split(",") if a.strip()]
    if len(authors) != 2:
        die("Use exactly two authors in --authors, e.g. 'austen,twain'")
    anchors = [a.strip() for a in (args.anchors or "").split(",") if a.strip()]
    text = generate_blended_word(authors[0], authors[1], args.model_root, order, args.sentences, anchors or None)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] wrote → {args.out}")
    else:
        print(text)

def main():
    ap = argparse.ArgumentParser(description="Shannon-style analysis & generation CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # analyze
    ap_a = sub.add_parser("analyze", help="build frequency tables for an author")
    ap_a.add_argument("--author", required=True, help="austen | twain | doyle")
    ap_a.add_argument("--file", required=True, help="path to .txt")
    ap_a.add_argument("--out", default="output", help="output root (default: output)")
    ap_a.set_defaults(func=cmd_analyze)

    # generate
    ap_g = sub.add_parser("generate", help="generate text at a level")
    ap_g.add_argument("--author", required=True, help="austen | twain | doyle")
    ap_g.add_argument("--level", required=True, help="char-0..char-3 or word-1..word-3 (accepts 'char -2' formats)")
    ap_g.add_argument("--length", type=int, default=400, help="character count for char levels")
    ap_g.add_argument("--sentences", type=int, default=5, help="sentence count for word levels")
    ap_g.add_argument("--anchors", type=str, default="", help="comma-separated anchor words")
    ap_g.add_argument("--model_root", default="output", help="root where <author>/stats lives")
    ap_g.add_argument("--out", default="", help="optional file to write")
    ap_g.set_defaults(func=cmd_generate)

    # compare
    ap_c = sub.add_parser("compare", help="show samples across levels for an author")
    ap_c.add_argument("--author", required=True, help="austen | twain | doyle")
    ap_c.add_argument("--model_root", default="output", help="root where <author>/stats lives")
    ap_c.set_defaults(func=cmd_compare)

    # blend (bonus)
    ap_b = sub.add_parser("blend", help="BONUS: blend two authors’ word models")
    ap_b.add_argument("--authors", required=True, help='e.g. "austen,twain"')
    ap_b.add_argument("--level", required=True, help="word-2 or word-3")
    ap_b.add_argument("--sentences", type=int, default=3)
    ap_b.add_argument("--anchors", type=str, default="")
    ap_b.add_argument("--model_root", default="output", help="root where <author>/stats lives")
    ap_b.add_argument("--out", default="")
    ap_b.set_defaults(func=cmd_blend)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

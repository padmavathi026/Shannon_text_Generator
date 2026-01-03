# Shannon Text Generator – Statistical Language Style Imitation

## Overview
This project implements a **Shannon-style probabilistic text generator** designed to model and reproduce the writing styles of **Jane Austen**, **Mark Twain**, and **Arthur Conan Doyle**.  
The system learns from character-level and word-level statistics to generate new text that mimics each author’s stylistic patterns using classical information-theoretic principles.

The full pipeline has been implemented, validated, and successfully executed across all components.

---

## Project Structure
The project is organized into four major components:

- **Part 1 – Preprocessing:** Text cleaning and normalization  
- **Part 2 – Statistical Analysis:** Character, word, and sentence-level frequency analysis  
- **Part 3 – Text Generation Engine:** Core probabilistic generator (`generator.py`)  
- **Part 4 – Command-Line Interface:** User-facing CLI (`shannon_gen.py`)  

---

## Setup Instructions

### Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows
Install Dependencies
pip install -r requirements.txt
```

### How to Run the Project
Build Statistical Models

Analyze each author’s text and construct frequency tables:
```
python shannon_gen.py analyze --author austen --file austen_pride_prejudice.txt
python shannon_gen.py analyze --author twain --file twain_tom_sawyer.txt
python shannon_gen.py analyze --author doyle --file doyle_sherlock_holmes.txt
```
### Text Generation
Character-Level Generation
python shannon_gen.py generate --author austen --level "char-2" --length 100

### Word-Level Generation
python shannon_gen.py generate --author twain --level "word-3" --sentences 5

### Generation with Anchor Words

python shannon_gen.py generate \
  --author doyle \
  --level "word-2" \
  --sentences 3 \
  --anchors "elementary,Watson,deduce"

### Model Comparison

Compare approximation quality across different n-gram levels:

python shannon_gen.py compare --author austen

Author Blending (Bonus Feature)

Generate hybrid styles by blending multiple authors:

python shannon_gen.py blend --authors "austen,twain" --level word-2 --sentences 3


To generate samples for all approximation levels:

python make_samples.py

Reports and Visualizations (Bonus)
Generate Analysis Reports
python build_reports.py


### This produces:

shannon_analysis_report.json – Author-level statistical summaries

model_comparison.csv – Entropy and perplexity comparisons across levels

generated_samples.txt – Representative generated text samples

style_examples.json – Short stylistic examples by author

Creative Blended Samples
python shannon_gen.py blend \
  --authors "austen,twain" \
  --level word-2 \
  --sentences 3 \
  --out samples/blend_austen_twain_word2.txt

python shannon_gen.py blend \
  --authors "austen,doyle" \
  --level word-3 \
  --sentences 3 \
  --out samples/blend_austen_doyle_word3.txt


Generated outputs are saved in the samples/ directory.

Visual Analysis
Zipf’s Law Distributions
python plot_zipf.py


Outputs:

images/zipf_austen_word1.png

images/zipf_twain_word1.png

images/zipf_doyle_word1.png

Entropy Comparison
python plot_entropy.py


### Output:

images/entropy_comparison.png

### Directory Structure
```text
├── analyze.py
├── build_reports.py
├── generator.py
├── info_theory.py
├── make_samples.py
├── plot_zipf.py
├── plot_entropy.py
├── shannon_gen.py
├── starter_preprocess.py
├── requirements.txt
├── README.md
│
├── austen_pride_prejudice.txt
├── twain_tom_sawyer.txt
├── doyle_sherlock_holmes.txt
│
│
├── output/
│   ├── austen/
│   ├── doyle/
│   ├── twain/
│
├── samples/
│
├── images/
│   ├── zipf_austen_word1.png
│   ├── zipf_twain_word1.png
│   ├── zipf_doyle_word1.png
│   ├── entropy_comparison.png
│
├── generated_samples.txt
├── shannon_analysis_report.json
├── model_comparison.csv
├── style_examples.json

```

This project demonstrates how classical Shannon-style probabilistic language models can effectively capture and reproduce stylistic patterns in natural language. By combining statistical analysis, entropy-based evaluation, and creative text synthesis, the system highlights the power and limitations of non-neural language modeling techniques.

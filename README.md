# Meridian: ESL-Aware AI Text Detection

**Meridian** is a dual-stream perplexity architecture for equitable AI text detection that reduces false positive rates on ESL student writing by 15.7× compared to standard detectors, while maintaining 98.3% detection of AI-mimicked ESL text.

> Associated paper: *"Meridian: A Bivariate Perplexity Architecture for Trustworthy and Equitable AI Text Detection"* — accepted to the 5th Workshop on NLP for Positive Impact (NLP4PI) @ EMNLP 2026.

---

## The Problem

Standard AI text detectors flag up to 72% of authentic ESL student essays as AI-generated. This is a population-scale fairness failure: the same perplexity axis that identifies AI text also penalizes learner writing patterns.

## How Meridian Works

Each essay is mapped to a point in a 2D space:
- **x-axis (P_esl):** How natural does this look to a character-level LSTM trained on ESL learner writing? (lower = more ESL-like)
- **y-axis (P_native):** How surprising does this look to DistilGPT-2? (higher = more idiosyncratic)

Human ESL writers occupy the upper-left; AI mimics cluster lower-left. A logistic classifier separates them.

## Results

| Method | ESL FPR | AI-ESL TPR | AUC |
|---|---|---|---|
| Fast-DetectGPT | 85.0% | 94.0% | 0.313 |
| Binoculars (ICML 2024) | 1.7% | 89.3% | — |
| **Meridian (ours)** | **8.3%** | **>99%** | **0.962** |
| **Meridian zero-shot TOEFL11** | **3.9%** | **98.3%** | — |

Low-proficiency writers: **0.0% FPR** (95% CI: [0.0%, 3.3%], n=110).

---

## Quickstart

```bash
pip install -r requirements.txt

# Score a single essay
python inference.py --text "I think that students would benefit from learn at home because..."

# Score from a file
python inference.py --file my_essay.txt

# Use a conservative threshold (lower FPR)
python inference.py --file essay.txt --threshold 0.80
```

**Output** (real output for an ESL-style sample paragraph):
```
Scoring essay (769 characters)...

=============================================
  Meridian Result
=============================================
  P_esl (ESL-Expert LSTM):     2.2
  P_native (DistilGPT-2):      45.351
  P(AI-generated):             0.8%
  Prediction:                  Human (ESL)
=============================================

Note: Meridian is a screening tool, not a verdict system.
All flagged essays should receive human review.
```

Essays must be longer than 60 characters: the ESL-Expert LSTM needs a
60-character context window before it can make its first prediction.

---

## Repository Contents

| File | Description |
|---|---|
| `inference.py` | Standalone inference script |
| `requirements.txt` | Python dependencies |
| `SUPER_esl_lstm_weights.pth` | Trained ESL-Expert LSTM weights (104-char vocab) |
| `notebooks/LSTM_ELS.ipynb` | Original Colab training/evaluation notebook: ESL-Expert LSTM training, DistilGPT-2 scoring, the Phase 2 logistic classifier, and the zero-shot TOEFL11 generalization run |
| `data/master_final_scores.csv` | Per-essay perplexity scores (P_esl, P_native) and group labels (human_esl / esl_ai / native_ai) used in the paper's Phase 2 evaluation |

### Reproducibility

`inference.py` reproduces the per-essay scores stored in
`data/master_final_scores.csv` to within ~2×10⁻⁵ relative error (float32
nondeterminism across hardware), so the released weights, the character
vocabulary, and the frozen classifier coefficients in `inference.py` are all
consistent with the numbers reported in the paper.

Two details matter if you reimplement scoring yourself:

- **P_esl** is computed with a **stride-1 sliding window**: every character
  after the first 60 is predicted from exactly the preceding 60 characters,
  and P_esl is the exponentiated mean negative log-likelihood over those
  predictions. Averaging over non-overlapping chunks instead gives values on
  a different scale, which the frozen classifier coefficients will misread.
- **The 104-character vocabulary** is `sorted(set(text))` over the ELLIPSE
  training corpus, so its ordering is fixed by the trained `fc` layer and
  cannot be regenerated from a generic character set. It is hardcoded in
  `inference.py`; see the comment there for provenance.

---

## Responsible Use

Meridian is a **screening tool, not a verdict system**. No automated detection output should initiate disciplinary proceedings without review by a qualified educator familiar with the student's language background.

See the paper's Responsible Deployment Framework (Appendix B) for full guidelines.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@inproceedings{raichura2026meridian,
  title     = {Meridian: A Bivariate Perplexity Architecture for
               Trustworthy and Equitable AI Text Detection},
  author    = {Raichura, Pranil},
  booktitle = {Proceedings of the 5th Workshop on NLP for Positive Impact (NLP4PI)},
  year      = {2026},
  note      = {EMNLP 2026 Workshop}
}
```

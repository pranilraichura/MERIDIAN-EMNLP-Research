"""
Meridian: ESL-Aware AI Text Detector
Scores an essay and returns a probability that it is AI-generated.

Usage:
    python inference.py --text "Your essay text here"
    python inference.py --file essay.txt

Requirements:
    pip install -r requirements.txt
"""

import argparse
import os

# Meridian only ever uses the PyTorch backend. Without this, `transformers`
# imports TensorFlow whenever it happens to be installed, which is slow and
# can deadlock in TensorFlow's native extension loader on some macOS setups.
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEQUENCE_LENGTH = 60


# ── Model Architecture (must match training) ─────────────────────────────────

class CharRNN(nn.Module):
    """Character-level LSTM. Predicts the next character from a fixed-length
    context window, so forward() returns logits for the final timestep only."""

    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


# ── Character vocabulary (must match training) ────────────────────────────────
#
# Recovered verbatim from the printed `char_to_index` output of the original
# training run. Source: the `LSTM_ELS.ipynb` training notebook, cell 7, whose
# stdout begins "104\nChar to Index Dictionary\n{'\n': 0, '\r': 1, ...}".
# That notebook had been deleted from this repo but was restored from git
# history (commits a686822 / 0739b87, deleted by c2ff50f); it is now committed
# back as `notebooks/LSTM_ELS.ipynb`. Four independently recovered revisions of
# the notebook print a byte-identical dictionary.
#
# The notebook builds it as `sorted(list(set(all_text)))` over the concatenated
# ELLIPSE corpus, so index order is codepoint-sorted and must not be changed --
# the `fc` layer's 104 output units are bound to exactly this ordering.
#
# Two quirks are genuine properties of the training corpus, not typos:
#   * '^' (0x5E) never appears in the corpus, so it is absent from the vocab.
#   * The final eight entries (0x80, 0x82, 0x9E, 0xBA, 0xC2, 0xC3, 0xCA, 0xE2)
#     are mojibake: the ELLIPSE text contained UTF-8 bytes that had already
#     been decoded as Latin-1, so e.g. a curly apostrophe U+2019 appears as the
#     three characters 'â', 0x80, 0x99. Unknown characters map to index 0 at
#     inference time, matching the training-time `char_to_index.get(ch, 0)`.

VOCAB = (
    "\n\r"
    " !\"#$%&'()*+,-./"
    "0123456789"
    ":;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~"
    "\x80\x82\x9e\xba\xc2\xc3\xca\xe2"
)

assert len(VOCAB) == 104, f"vocab must be 104 chars, got {len(VOCAB)}"
assert list(VOCAB) == sorted(VOCAB), "vocab must stay in codepoint-sorted order"


def build_vocab():
    """Return (char_to_index, vocab_size) for the trained model."""
    char_to_index = {c: i for i, c in enumerate(VOCAB)}
    return char_to_index, len(VOCAB)


# ── Perplexity functions ──────────────────────────────────────────────────────

def compute_esl_perplexity(text, model, char_to_index, vocab_size,
                           seq_len=SEQUENCE_LENGTH, device="cpu",
                           batch_size=256):
    """Character-level LSTM perplexity (P_esl).

    Mirrors `compute_complexity` from the training notebook: a stride-1 sliding
    window predicts each character from the preceding `seq_len` characters, and
    the mean negative log-likelihood is exponentiated. Windows are independent
    (no hidden state is carried across them), so they are batched here purely
    for speed; the result is identical to scoring them one at a time.
    """
    model.eval()
    indices = [char_to_index.get(ch, 0) for ch in text]

    if len(indices) <= seq_len:
        return float("inf")

    starts = list(range(seq_len, len(indices)))
    total_nll = 0.0

    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch = starts[i: i + batch_size]
            x = np.zeros((len(batch), seq_len, vocab_size), dtype=np.float32)
            for b, t in enumerate(batch):
                x[b, np.arange(seq_len), indices[t - seq_len: t]] = 1.0
            x_tensor = torch.from_numpy(x).to(device)
            targets = torch.tensor([indices[t] for t in batch],
                                   dtype=torch.long, device=device)

            logits = model(x_tensor)
            log_probs = F.log_softmax(logits, dim=1)
            total_nll += -log_probs.gather(1, targets[:, None]).sum().item()

    avg_nll = total_nll / len(starts)
    return float(np.exp(avg_nll))


def compute_native_perplexity(text, gpt_model, tokenizer, device="cpu"):
    """DistilGPT-2 sub-word perplexity (P_native)."""
    gpt_model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    if enc["input_ids"].shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        loss = gpt_model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss).item())


# ── Logistic classifier (Phase 2 frozen weights) ─────────────────────────────
# Verified by refitting `LogisticRegression(random_state=42)` on the
# human_esl vs. esl_ai rows of data/master_final_scores.csv, exactly as the
# training notebook does; that refit reproduces these values to 4 decimals
# (coef -1.81664516, -0.28311278; intercept 12.05685854).

W_ESL = -1.8166
W_NATIVE = -0.2831
INTERCEPT = 12.0569


def classify(p_esl, p_native, threshold=0.50):
    """
    Returns probability of AI-generated and binary prediction.
    Score > threshold → AI-generated.
    """
    logit = W_ESL * p_esl + W_NATIVE * p_native + INTERCEPT
    prob_ai = 1.0 / (1.0 + np.exp(-logit))
    return prob_ai, prob_ai > threshold


# ── Main ──────────────────────────────────────────────────────────────────────

def load_models(lstm_weights_path="SUPER_esl_lstm_weights.pth", device="cpu"):
    """Load the ESL-Expert LSTM and DistilGPT-2."""
    char_to_index, vocab_size = build_vocab()

    lstm = CharRNN(vocab_size).to(device)
    state = torch.load(lstm_weights_path, map_location=device, weights_only=True)
    lstm.load_state_dict(state)

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    gpt_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    return lstm, char_to_index, vocab_size, gpt_model, tokenizer


def score_essay(text, lstm_weights_path="SUPER_esl_lstm_weights.pth",
                threshold=0.50, device=None, models=None):
    """
    Score a single essay text.

    Returns:
        dict with keys: p_esl, p_native, prob_ai, prediction
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if models is None:
        models = load_models(lstm_weights_path, device)
    lstm, char_to_index, vocab_size, gpt_model, tokenizer = models

    p_esl = compute_esl_perplexity(text, lstm, char_to_index, vocab_size,
                                   device=device)
    p_native = compute_native_perplexity(text, gpt_model, tokenizer,
                                         device=device)

    if not np.isfinite(p_esl) or not np.isfinite(p_native):
        return {
            "p_esl": p_esl,
            "p_native": p_native,
            "prob_ai": float("nan"),
            "prediction": "Unscorable (text too short)",
        }

    prob_ai, is_ai = classify(p_esl, p_native, threshold)

    return {
        "p_esl": round(p_esl, 3),
        "p_native": round(p_native, 3),
        "prob_ai": round(float(prob_ai), 4),
        "prediction": "AI-generated" if is_ai else "Human (ESL)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meridian ESL-aware AI text detector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Essay text as a string")
    group.add_argument("--file", type=str, help="Path to a .txt file")
    parser.add_argument("--weights", default="SUPER_esl_lstm_weights.pth",
                        help="Path to LSTM weights file")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Classification threshold (default 0.50; use 0.80 for lower FPR)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = args.text

    print(f"\nScoring essay ({len(text)} characters)...")

    if len(text) <= SEQUENCE_LENGTH:
        print(f"\nError: text must be longer than {SEQUENCE_LENGTH} characters "
              f"to score (the LSTM needs a {SEQUENCE_LENGTH}-character context "
              f"window before its first prediction).")
        raise SystemExit(1)

    result = score_essay(text, lstm_weights_path=args.weights,
                         threshold=args.threshold)

    print(f"\n{'='*45}")
    print(f"  Meridian Result")
    print(f"{'='*45}")
    print(f"  P_esl (ESL-Expert LSTM):     {result['p_esl']}")
    print(f"  P_native (DistilGPT-2):      {result['p_native']}")
    print(f"  P(AI-generated):             {result['prob_ai']:.1%}")
    print(f"  Prediction:                  {result['prediction']}")
    print(f"{'='*45}")
    print(f"\nNote: Meridian is a screening tool, not a verdict system.")
    print(f"All flagged essays should receive human review.")

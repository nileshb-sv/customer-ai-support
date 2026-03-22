"""
pipeline/predict.py
Classifies a complaint → (category: str, confidence: float)

Flow:
  1. Keyword scoring  (fast, always runs)
  2. T5 fallback      (only when keyword score == 0 AND model trained)

All T5 prompt engineering decisions documented below.
"""
from pipeline.model_loader import load_model

# ── T5 LABEL MAP ──────────────────────────────────────────────────────
# Training data uses lowercase+underscore: fraud, account, payment,
# loan, credit_card, other
# We expose Title Case to the rest of the app: Fraud, Account, etc.
# This map converts T5 raw output → app-standard label
T5_LABEL_MAP = {
    "fraud":       "Fraud",
    "account":     "Account",
    "payment":     "Payment",
    "credit_card": "Credit Card",
    "creditcard":  "Credit Card",   # handle no-underscore variant
    "credit card": "Credit Card",   # handle space variant
    "loan":        "Loan",
    "other":       "Other",
}

VALID_CATEGORIES = ["Fraud", "Account", "Payment", "Credit Card", "Loan", "Other"]

# ── KEYWORDS ──────────────────────────────────────────────────────────
KEYWORDS = {
    "Fraud": [
        "fraud", "hacked", "unauthorized", "suspicious", "fraudulent",
        "unknown deduction", "account hacked", "phishing", "scam",
        "identity theft", "stolen", "compromised", "didn't do this",
        "not done by me", "someone accessed",
    ],
    "Account": [
        "account", "login", "password", "locked", "unable to login",
        "account locked", "mobile number", "username", "sign in",
        "otp", "verification", "access denied", "blocked", "register",
        "profile update", "mobile number update",
    ],
    "Payment": [
        "payment", "money deducted", "upi", "transaction", "payment failed",
        "double payment", "refund", "transfer", "neft", "rtgs", "imps",
        "debited", "charged", "amount not received", "pending", "stuck",
        "not reflected", "transaction pending",
    ],
    "Credit Card": [
        "credit card", "card declined", "credit limit", "card blocked",
        "card stolen", "cvv", "card not working", "statement", "billing",
        "international transaction", "credit card charges",
    ],
    "Loan": [
        "loan", "emi", "disbursement", "interest rate", "loan approval",
        "repayment", "mortgage", "personal loan", "home loan", "overdue",
        "loan disbursement", "emi deduction",
    ],
    "Other": [
        "service quality", "technical", "app crash", "support",
        "feedback", "not happy", "worst", "mobile app crashing",
        "customer support not responding",
    ],
}


# ── KEYWORD CLASSIFIER ────────────────────────────────────────────────
def keyword_classifier(ticket: str) -> tuple[str, float]:
    t      = ticket.lower()
    scores = {
        cat: sum(1 for w in words if w in t)
        for cat, words in KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Other", 0.0
    total = sum(scores.values())
    conf  = round(min(scores[best] / total, 1.0), 2)
    return best, conf


# ── T5 CLASSIFIER ─────────────────────────────────────────────────────
def t5_classifier(ticket: str) -> tuple[str, float] | None:
    """
    Uses fine-tuned T5-small for classification.

    PROMPT ENGINEERING DECISIONS:
    ─────────────────────────────
    1. FORMAT MATCHES TRAINING DATA EXACTLY
       Training format: "Classify complaint: <complaint text>"
       We use the same prefix — T5 learned to respond to THIS exact trigger.
       Using a different prompt (e.g. long instruction with category list) 
       causes the model to see an out-of-distribution input → garbage output.

    2. NO CATEGORY LIST IN PROMPT
       Training data never showed category names to T5 — it learned to map 
       complaint text directly to label tokens (fraud, account, etc.).
       Adding "Categories: Fraud, Account..." confuses the model.

    3. SHORT INPUT = BETTER ACCURACY
       T5-small has a 512 token context. Training inputs avg 231 chars (~58 tokens).
       Keeping inference input short and clean = model stays in-distribution.
       We pass the raw ticket (truncated at 300 chars to match training distribution).

    4. GREEDY DECODING (num_beams=1, do_sample=False)
       Classification has exactly one correct answer.
       Beam search adds latency with zero accuracy benefit for single-token outputs.
       Greedy is faster and equally accurate here.

    5. max_new_tokens=15
       Longest label is "credit_card" = 11 chars ≈ 4 tokens.
       15 tokens gives safe headroom without wasting compute.

    6. OUTPUT NORMALIZATION
       T5 outputs the exact label token it was trained on: "fraud", "credit_card" etc.
       We normalize: lowercase → strip → replace underscore → map to Title Case.
       This handles all variants: "Credit Card", "credit_card", "creditcard".
    """
    tokenizer, model, device, use_trained_model = load_model()
    if not use_trained_model or model is None:
        return None

    try:
        import torch

        # ── PROMPT: exactly matches training data format ──────────────
        prompt = f"Classify complaint: {ticket[:300]}"

        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=128,       # matches training tokenization max_length
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=15,    # safe for all label lengths
                num_beams=1,          # greedy — classification, not generation
                do_sample=False,      # deterministic output
            )

        raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # ── OUTPUT NORMALIZATION ──────────────────────────────────────
        # Normalize: lowercase, strip whitespace, collapse underscore/space
        normalized = raw.lower().strip().replace("_", " ")

        # Direct map lookup (handles credit_card → Credit Card etc.)
        raw_key = raw.lower().strip()
        if raw_key in T5_LABEL_MAP:
            return T5_LABEL_MAP[raw_key], 0.82

        # Normalized map lookup
        norm_key = normalized.replace(" ", "")
        for k, v in T5_LABEL_MAP.items():
            if k.replace("_","").replace(" ","") == norm_key:
                return v, 0.78

        # Partial match fallback
        for k, v in T5_LABEL_MAP.items():
            if k in raw.lower():
                return v, 0.70

    except Exception as e:
        print(f"T5 classify error: {e}")

    return None


# ── MAIN CLASSIFIER ───────────────────────────────────────────────────
def classify_ticket(ticket: str) -> tuple[str, float]:
    """
    Always returns (category: str, confidence: float 0-1).

    Strategy:
    - Keyword classifier runs first (fast, ~0ms)
    - T5 only called when keywords find nothing (conf == 0.0)
    - This keeps latency low for common complaints
    """
    cat, conf = keyword_classifier(ticket)

    if conf == 0.0:
        result = t5_classifier(ticket)
        if result:
            cat, conf = result

    return cat, conf


def predict(complaint: str) -> tuple[str, float]:
    return classify_ticket(complaint)

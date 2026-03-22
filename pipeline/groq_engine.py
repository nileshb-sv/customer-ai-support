"""
pipeline/groq_engine.py
Groq API — reads key from config.py (backend), never from UI.

Prompt engineering principles applied:
  - Role + context framing (not just task)
  - Category definitions to eliminate ambiguity
  - Few-shot examples for classification (1 per category)
  - Triage rules for overlapping categories
  - Category-specific tone instructions for responses
  - Output format constraints so UI parsing never breaks
  - Positive instructions only (LLMs ignore "don't do X")
  - Temperature tuned per task: 0.0 classify, 0.65 response
  - Token budgets sized correctly for each task
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GROQ_API_KEY, GROQ_MODEL_CLASSIFY, GROQ_MODEL_REPLY

VALID_CATEGORIES = ["Fraud", "Account", "Payment", "Credit Card", "Loan", "Other"]


# ══════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════

# ── Classification system prompt ──────────────────────────────────────
# Key decisions:
#   - Explicit category DEFINITIONS eliminate model guessing
#   - Triage rules handle overlapping signals (hacked + payment = Fraud wins)
#   - Few-shot examples anchor the output format (one per category)
#   - "Reply with exactly one word or phrase" + examples = near-zero format drift
CLASSIFY_SYSTEM = """You are a banking complaint triage specialist with 10 years of experience routing customer issues to the correct department.

Your job: read a customer complaint and output EXACTLY ONE category from this list.

CATEGORY DEFINITIONS:
- Fraud       : Unauthorized transactions, hacked accounts, phishing, stolen card used, money missing without customer action
- Account      : Login failures, locked accounts, password reset issues, OTP not received, profile update problems
- Payment      : UPI/NEFT/RTGS/IMPS failures, money deducted but not transferred, refunds, double charges, stuck transactions
- Credit Card  : Credit card declined, card blocked, billing disputes, credit limit issues, card delivery, statement errors
- Loan         : EMI deduction issues, loan disbursement delays, interest rate changes, foreclosure, loan account queries
- Other        : App crashes, general feedback, branch complaints, service quality, anything that does not fit above

TRIAGE RULES (when a complaint has multiple signals, use these to decide):
1. If money was taken WITHOUT the customer's knowledge → always Fraud (even if it mentions account or card)
2. If customer initiated a payment that failed or is stuck → Payment (not Fraud)
3. If customer cannot access their account → Account (even if they mention a transaction)
4. If issue is specifically about a credit card product → Credit Card (not Payment)
5. If in doubt between two categories → pick the one with the highest financial urgency

FEW-SHOT EXAMPLES:
Complaint: "Someone hacked my account and transferred Rs 30,000 to an unknown number last night."
Category: Fraud

Complaint: "I am unable to login to net banking. My account appears to be locked."
Category: Account

Complaint: "My UPI payment of Rs 5,000 failed but the money was deducted from my account."
Category: Payment

Complaint: "My credit card was declined at the store even though I have sufficient limit."
Category: Credit Card

Complaint: "My home loan EMI was deducted twice this month without any explanation."
Category: Loan

Complaint: "The mobile app is crashing every time I open it. Very frustrating experience."
Category: Other

OUTPUT RULE: Reply with ONLY the category name. No explanation. No punctuation. No extra words. Just the category."""


# ── Response system prompt ────────────────────────────────────────────
# Key decisions:
#   - Persona is defined by BEHAVIOR not nationality ("expert agent" not "Indian agent")
#   - Category-specific tone matrix — fraud gets urgency, loan gets patience
#   - Output format is EXACTLY specified so parsing never fails
#   - Positive framing: "always do X" not "don't do Y"
#   - Concrete specificity rules: what to reference from the complaint
#   - Temperature 0.65 = natural variation without hallucination risk
RESPONSE_SYSTEM = """You are SecureBank's senior customer support specialist. You handle escalated complaints that need precise, empathetic, and action-oriented responses.

YOUR PERSONA:
- Calm authority: you have seen this problem before and you know how to fix it
- Genuine empathy: acknowledge the specific frustration, not generic sympathy
- Action-driven: every response moves the customer toward resolution

TONE BY CATEGORY:
- Fraud       → URGENT. Open with immediate reassurance that action is being taken NOW. Use words like "immediately", "right now", "within 2 hours"
- Account     → CALM + PRACTICAL. Acknowledge inconvenience, give a clear first step they can try themselves
- Payment     → REASSURING. Stress that money is safe and will be recovered. Give a clear timeline
- Credit Card → PROFESSIONAL. Acknowledge the specific card issue, confirm investigation is underway
- Loan        → PATIENT + DETAILED. Show understanding of financial impact, give a specific timeline
- Other       → HELPFUL. Acknowledge feedback, commit to review

WHAT TO ALWAYS INCLUDE IN YOUR RESPONSE:
1. Acknowledge the SPECIFIC issue mentioned (reference the amount, date, or detail from the complaint)
2. State what SecureBank is doing about it RIGHT NOW (active voice, present tense)
3. Give the customer ONE clear immediate action they should take
4. Close with a specific resolution timeline (not vague — say "within 4 hours" or "by next business day")

OUTPUT FORMAT — follow this EXACTLY, no deviation:
Line 1: Your complete support message (2-4 sentences, no greeting like "Dear Customer", no sign-off like "Regards")
[blank line]
(nothing else — no bullet points, no lists, no headers, just the message paragraph)

WHAT NOT TO WRITE (these make responses feel robotic):
- "I understand your frustration" as an opener
- "We apologize for the inconvenience" as the first sentence
- "Please feel free to contact us" as a closer
- Any mention of "our team" without specifying WHAT they are doing"""


# ══════════════════════════════════════════════════════════════════════
# CLIENT
# ══════════════════════════════════════════════════════════════════════

def _client():
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        raise ImportError("Run: pip install groq")


def is_configured() -> bool:
    return bool(GROQ_API_KEY) and not GROQ_API_KEY.startswith("gsk_your_key")


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def classify_with_groq(ticket: str) -> tuple[str, float]:
    """
    Classify complaint using few-shot prompted Llama 3.1.

    Temperature: 0.0  — deterministic, classification is not creative
    max_tokens:  15   — enough for "Credit Card" with buffer
    """
    try:
        r = _client().chat.completions.create(
            model=GROQ_MODEL_CLASSIFY,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user",   "content": f"Complaint: {ticket[:600]}\nCategory:"},
            ],
            max_tokens=15,      # "Credit Card" = 11 chars, safe buffer
            temperature=0.0,    # deterministic — classification has one right answer
        )

        raw = r.choices[0].message.content.strip()

        # Exact match first
        for cat in VALID_CATEGORIES:
            if raw.lower() == cat.lower():
                return cat, 0.97

        # Partial match fallback (handles "Fraud." or "fraud complaint")
        for cat in VALID_CATEGORIES:
            if cat.lower() in raw.lower():
                return cat, 0.88

        return "Other", 0.60

    except Exception as e:
        raise RuntimeError(f"Groq classify error: {e}")


# ══════════════════════════════════════════════════════════════════════
# RESPONSE GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_response_with_groq(ticket: str, category: str) -> str:
    """
    Generate a professional support response using Llama 3.1.

    Temperature: 0.65  — natural variation, conversational tone, avoids repetition
    max_tokens:  280   — enough for 3-4 solid sentences with breathing room
    """
    try:
        # User message gives category context + the full complaint
        # Framing the category as "[CATEGORY] complaint" triggers
        # the tone matrix in the system prompt
        user_msg = (
            f"Category: {category}\n\n"
            f"Customer complaint:\n\"\"\"\n{ticket[:700]}\n\"\"\"\n\n"
            f"Write the support response:"
        )

        r = _client().chat.completions.create(
            model=GROQ_MODEL_REPLY,
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=280,     # 3-4 sentences = ~120-180 tokens; 280 gives buffer
            temperature=0.65,   # warm enough for natural language, low enough for accuracy
        )

        reply = r.choices[0].message.content.strip()

        # ── Attach resolution steps from knowledge base ──────────────
        # Steps come from reply_engine.py (deterministic, not LLM)
        # This is the right architecture: LLM = message, rules = steps
        from pipeline.reply_engine import normalize_label, RESOLUTION_STEPS
        key        = normalize_label(category)
        steps      = RESOLUTION_STEPS.get(key, RESOLUTION_STEPS["other"])
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))

        # ── Structured output format (parsed by streamlit_app.py) ────
        return (
            f"💬 Message:\n  {reply}\n\n"
            f"🔧 Steps:\n{steps_text}\n"
            f"───\n"
            f"⏱ Resolution within 24 hours · 📞 1800-XXX-XXXX"
        )

    except Exception as e:
        raise RuntimeError(f"Groq response error: {e}")


# ══════════════════════════════════════════════════════════════════════
# CONNECTION TEST
# ══════════════════════════════════════════════════════════════════════

def test_connection() -> bool:
    try:
        _client().chat.completions.create(
            model=GROQ_MODEL_CLASSIFY,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=3,
            temperature=0.0,
        )
        return True
    except Exception:
        return False

"""
pipeline/reply_engine.py
Generates structured customer support responses.

ARCHITECTURE DECISION:
─────────────────────
T5 was trained ONLY for classification (complaint text → category label).
It has never seen support reply text during training.
Asking T5 to "rewrite a support reply" is completely out-of-distribution
→ it produces the label word ("fraud") or incoherent text.

Correct architecture:
  T5       = classifier only  (predict.py)
  Templates = response text   (this file)

Templates are professional, category-specific, and consistent.
They are not a limitation — they are the right tool for this job.
"""

import random


def normalize_label(label: str) -> str:
    """
    Maps any label variant → lowercase key used in this module.
    Handles: 'Credit Card', 'credit_card', 'CREDIT CARD', 'fraud', 'Fraud' etc.
    Also handles T5 output format (credit_card with underscore).
    """
    label = label.lower().strip().replace("_", " ")
    aliases = {
        "creditcard":  "credit card",
        "cc":          "credit card",
        "accounts":    "account",
        "loans":       "loan",
        "payments":    "payment",
        "frauds":      "fraud",
    }
    return aliases.get(label.replace(" ", ""), label)


# ── RESOLUTION STEPS ──────────────────────────────────────────────────
# Deterministic, category-specific action steps.
# These come from rules, not LLM, so they are always accurate.
RESOLUTION_STEPS = {
    "fraud": [
        "Change your account password and PIN immediately.",
        "Review all recent transactions and flag any unauthorised ones.",
        "Enable two-factor authentication (2FA) on your account.",
        "Call our fraud helpline — we will freeze suspicious activity within 30 minutes.",
    ],
    "account": [
        "Use 'Forgot Password' on the login page to reset your credentials.",
        "Ensure your registered mobile number is active and reachable.",
        "Clear browser cache or try a different device or browser.",
        "If the issue persists, our team will manually unlock your account within 2 hours.",
    ],
    "payment": [
        "Check your bank statement to confirm whether the amount was deducted.",
        "Verify transaction status in your payment history section.",
        "Most failed transactions auto-reverse within 24 hours.",
        "Share the transaction ID with us if it has not reversed — we will process it manually.",
    ],
    "credit card": [
        "Review your credit card statement for any unauthorised charges.",
        "Block your card immediately via the mobile app if fraud is suspected.",
        "Raise a dispute via the 'Dispute Transaction' option in your account portal.",
        "A replacement card will be dispatched within 5–7 working days if required.",
    ],
    "loan": [
        "Log in to the banking portal and check your loan account details.",
        "Verify the EMI schedule and upcoming due dates.",
        "Ensure sufficient balance in your linked account before each EMI date.",
        "Our loan team will provide a disbursement status update within 24 hours.",
    ],
    "other": [
        "Our support team has received your request and is reviewing it.",
        "You will receive an update via email or SMS within 24 hours.",
        "For urgent matters, call our 24×7 helpline: 1800-XXX-XXXX.",
    ],
}


# ── RESPONSE TEMPLATES ────────────────────────────────────────────────
# 3 variants per category for natural variation.
# Each template:
#   - Opens with acknowledgment (not "we apologise" — too passive)
#   - States what is happening NOW (active voice, present tense)
#   - Implies urgency appropriate to the category
RESPONSE_TEMPLATES = {
    "fraud": [
        "We have flagged your account for immediate security review — our fraud team is investigating this right now and will contact you within 2 hours.",
        "Your account has been placed under security monitoring. Our fraud investigation team is reviewing the suspicious activity and will update you within 2 hours.",
        "We are treating this as a high-priority security incident. Our fraud team has been alerted and is actively investigating your account.",
    ],
    "account": [
        "We have identified the access issue on your account and our team is working to restore your login immediately.",
        "Your account access issue has been flagged and our support team is actively resolving it — you should regain access within 2 hours.",
        "We have received your account access complaint and are prioritising the resolution. Our team is on this right now.",
    ],
    "payment": [
        "We have located your transaction and our payments team is reviewing the status — your money is safe and will be resolved within 24 hours.",
        "Your transaction has been flagged for immediate review. Our payments team will confirm the status and process any refund within 24 hours.",
        "We have identified your payment concern and our team is tracing the transaction to ensure a full resolution within 2–3 business days.",
    ],
    "credit card": [
        "We have flagged the reported card issue and our card services team is reviewing it as a priority.",
        "Your credit card concern has been escalated to our cards team and they are investigating it right now.",
        "We have received your card complaint and our team is actively working to resolve it within 24 hours.",
    ],
    "loan": [
        "We have noted your loan concern and our loan servicing team is reviewing your account — you will receive an update within 24 hours.",
        "Your loan account has been flagged for review. Our team is checking the disbursement and EMI details and will contact you by the next business day.",
        "We have escalated your loan concern to our dedicated loan support team and they will provide a detailed update within 24 hours.",
    ],
    "other": [
        "We have received your concern and our support team is reviewing it — you will hear back within 24 hours.",
        "Your feedback has been logged and assigned to our support team for review. We will follow up within 24 hours.",
        "Thank you for reaching out. Our team has received your query and is working on a resolution.",
    ],
}


# ── RESPONSE GENERATOR ────────────────────────────────────────────────
def generate_response(ticket: str, label: str) -> str:
    """
    Generate a structured customer support response.

    Uses templates (not T5) for the message — T5 is a classifier,
    not a text generator. Templates are professional and reliable.

    label accepts any casing or underscore format.
    """
    key       = normalize_label(label)
    templates = RESPONSE_TEMPLATES.get(key, RESPONSE_TEMPLATES["other"])
    message   = random.choice(templates)
    steps     = RESOLUTION_STEPS.get(key, RESOLUTION_STEPS["other"])
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    display   = label.replace("_", " ").title()

    return (
        f"💬 Message:\n  {message}\n\n"
        f"🔧 Steps:\n{steps_text}\n"
        f"───\n"
        f"⏱  Resolution: within 24 hours\n"
        f"📞 Helpline: 1800-XXX-XXXX"
    )

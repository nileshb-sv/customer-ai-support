"""
app/streamlit_app.py
Run: python -m streamlit run app/streamlit_app.py
"""

import sys, os, random
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "enterprise_complaints.csv"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.predict      import classify_ticket
from pipeline.reply_engine import generate_response
from pipeline.groq_engine  import (
    classify_with_groq,
    generate_response_with_groq,
    is_configured,
    test_connection,
)

EXAMPLES = [
    ("Fraud",       "My account was hacked and Rs 42,000 was transferred to an unknown number without my knowledge or consent."),
    ("Account",     "I tried logging into net banking but it says my account is locked. I have not entered the wrong password."),
    ("Payment",     "My UPI payment of Rs 8,500 to Swiggy failed but the money got deducted from my account immediately."),
    ("Credit Card", "My credit card was declined at the supermarket even though I have Rs 75,000 available credit limit."),
    ("Loan",        "My home loan EMI of Rs 23,000 was deducted twice this month. I need an immediate refund of the extra amount."),
    ("Fraud",       "Someone made three unauthorized transactions on my debit card totaling Rs 15,000 yesterday evening."),
    ("Account",     "I cannot reset my password. The OTP is not arriving on my registered mobile number since two days."),
    ("Payment",     "My NEFT transfer of Rs 50,000 is stuck for 2 days. Money left my account but recipient has not received it."),
    ("Loan",        "My personal loan application was approved 2 weeks ago but the disbursement amount has not arrived yet."),
    ("Other",       "The mobile banking app crashes every time I try to open it. I have tried reinstalling but same issue."),
    ("Loan",        "Interest rate on my personal loan was raised from 12% to 15% without any prior notice or communication."),
    ("Credit Card", "I was charged a lifetime annual fee of Rs 2,000 on my credit card which was promised to be free forever."),
]

st.set_page_config(
    page_title="SupportAI · Banking",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1rem 5rem !important; max-width: 820px; }

/* ═══ HERO ═══════════════════════════════════════════════ */
.hero {
    background: #0A0F1E;
    border-radius: 24px;
    padding: 44px 40px 38px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.2);
}
.hero-glow-1 {
    position:absolute; top:-80px; right:-80px; width:320px; height:320px;
    background:radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 65%);
    pointer-events:none;
}
.hero-glow-2 {
    position:absolute; bottom:-60px; left:40px; width:240px; height:240px;
    background:radial-gradient(circle, rgba(168,85,247,0.2) 0%, transparent 65%);
    pointer-events:none;
}
.hero-grid {
    position:absolute; inset:0;
    background-image: linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events:none;
}
.hero-inner { position:relative; z-index:1; }
.hero-live {
    display:inline-flex; align-items:center; gap:7px;
    background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
    color:#34D399 !important; font-size:0.7rem; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase;
    padding:5px 14px; border-radius:30px; margin-bottom:18px;
}
.hero-live-dot {
    width:6px; height:6px; background:#10B981; border-radius:50%;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.hero-h1 {
    font-size:2.4rem; font-weight:700; color:#F1F5F9 !important;
    margin:0 0 12px; line-height:1.1;
    letter-spacing:-0.02em;
}
.hero-h1 em { color:#818CF8 !important; font-style:normal; }
.hero-desc { color:#94A3B8 !important; font-size:0.95rem; font-weight:400; margin:0 0 30px; line-height:1.6; }
.hero-pills { display:flex; gap:8px; flex-wrap:wrap; }
.hero-pill {
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
    color:#CBD5E1 !important; font-size:0.75rem; font-weight:500;
    padding:5px 13px; border-radius:20px;
}
.hero-pill span { color:#C7D2FE !important; font-weight:600; }

/* ═══ TABS ════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #F8FAFC; border-radius: 14px;
    padding: 5px; gap: 4px; border: none;
    margin-bottom: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 10px 24px;
    font-weight: 600; font-size: 0.88rem;
    color: #94A3B8; background: transparent; border: none;
    transition: all 0.18s;
}
.stTabs [aria-selected="true"] {
    background: white !important; color: #1E293B !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.09) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"]  { padding-top: 24px; }

/* ═══ SECTION CARDS ═══════════════════════════════════════ */
.section-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 24px 26px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.72rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: ''; display: inline-block; width: 16px; height: 2px;
    background: #6366F1; border-radius: 2px;
}

/* Randomize card — distinct teal accent */
.rand-card {
    background: linear-gradient(135deg, #F0FDF9 0%, #F8FFFE 100%);
    border: 1px solid #CCFBF1;
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 16px;
}
.rand-card .section-title::before { background: #14B8A6; }

/* ═══ TEXTAREA ════════════════════════════════════════════ */
.stTextArea textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.93rem !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    color: #0F172A !important;
    background: #FAFBFF !important;
    resize: none !important;
    line-height: 1.7 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    background: white !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.08) !important;
    outline: none !important;
}
.stTextArea label { display: none !important; }

/* ═══ BUTTONS ═════════════════════════════════════════════ */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 10px !important;
    padding: 11px 22px !important;
    border: none !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}

/* Analyze button — indigo */
.analyze-btn .stButton > button {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(99,102,241,0.32) !important;
}
.analyze-btn .stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5, #4338CA) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(99,102,241,0.38) !important;
}

/* Randomize button — teal, completely different from Analyze */
.rand-btn .stButton > button {
    background: linear-gradient(135deg, #0D9488, #0F766E) !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(13,148,136,0.28) !important;
}
.rand-btn .stButton > button:hover {
    background: linear-gradient(135deg, #0F766E, #115E59) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(13,148,136,0.36) !important;
}

/* ═══ STATUS DOT ══════════════════════════════════════════ */
.status-row { display:flex; align-items:center; gap:8px; margin-bottom:18px; }
.sdot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.sdot.on  { background:#10B981; box-shadow:0 0 0 0 rgba(16,185,129,0.4);
            animation:pulse-g 2s ease-in-out infinite; }
.sdot.off { background:#CBD5E1; }
@keyframes pulse-g {
    0%   { box-shadow:0 0 0 0 rgba(16,185,129,0.4); }
    70%  { box-shadow:0 0 0 7px rgba(16,185,129,0); }
    100% { box-shadow:0 0 0 0 rgba(16,185,129,0); }
}
.stext { font-size:0.82rem; color:#64748B; font-weight:500; }
.stext strong { color:#0F172A; }

/* ═══ RANDOM PREVIEW ══════════════════════════════════════ */
.rand-preview {
    background: white;
    border: 1px solid #99F6E4;
    border-left: 3px solid #14B8A6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 0.9rem; color: #134E4A;
    line-height: 1.7; margin: 12px 0 0;
}
.rand-cat-tag {
    display: inline-block;
    background: #CCFBF1; color: #0F766E;
    font-size: 0.7rem; font-weight: 700;
    padding: 2px 9px; border-radius: 10px;
    margin-bottom: 8px; letter-spacing: 0.04em;
}

/* ═══ RESULT CARD ═════════════════════════════════════════ */
@keyframes slide-up {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-outer { animation: slide-up 0.38s ease; margin-top: 20px; }
.rc {
    border-radius: 18px; overflow: hidden;
    box-shadow: 0 6px 32px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F0;
}
.rc-head {
    padding: 20px 26px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid rgba(0,0,0,0.06);
}
.rc-head.fraud   { background: linear-gradient(120deg,#FEF2F2,#FFF7F7); }
.rc-head.account { background: linear-gradient(120deg,#FFFBEB,#FFFEF7); }
.rc-head.payment { background: linear-gradient(120deg,#EFF6FF,#F5FAFF); }
.rc-head.credit  { background: linear-gradient(120deg,#F5F3FF,#FAF8FF); }
.rc-head.loan    { background: linear-gradient(120deg,#F0FDF4,#F5FFF7); }
.rc-head.other   { background: linear-gradient(120deg,#F8FAFC,#FAFBFC); }
.cat-icon  { font-size:1.4rem; margin-right:10px; }
.cat-name  { font-size:1.05rem; font-weight:700; color:#0F172A; }
.cat-sub   { font-size:0.73rem; color:#94A3B8; margin-top:1px; }
.mpill {
    font-size: 0.7rem; font-weight: 700;
    padding: 4px 12px; border-radius: 20px;
    letter-spacing: 0.02em;
}
.mpill.groq  { background:#EEF2FF; color:#4338CA; }
.mpill.local { background:#F0FDF4; color:#166534; }
.rc-body { padding: 24px 26px; background: white; }
.rlabel {
    font-size: 0.7rem; font-weight: 800; color: #CBD5E1;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}
.rmsg {
    font-size: 0.92rem; color: #1E293B; line-height: 1.78;
    padding: 16px 20px;
    background: #FAFBFF;
    border-left: 3px solid #6366F1;
    border-radius: 0 12px 12px 0;
    margin-bottom: 22px;
}
.steps-grid { display: flex; flex-direction: column; gap: 8px; }
.step-row  { display: flex; gap: 14px; align-items: flex-start; }
.snum {
    min-width: 26px; height: 26px;
    background: linear-gradient(135deg,#6366F1,#4F46E5);
    color: white; font-size: 0.7rem; font-weight: 700;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
    box-shadow: 0 2px 6px rgba(99,102,241,0.3);
}
.stxt { font-size: 0.875rem; color: #334155; line-height: 1.65; }
.rc-foot {
    padding: 14px 26px;
    background: #F8FAFC;
    border-top: 1px solid #F1F5F9;
    display: flex; align-items: center; justify-content: space-between;
}
.foot-item { font-size: 0.78rem; color: #94A3B8; }
.foot-item strong { color: #475569; }

/* ═══ CONFIG BOX ══════════════════════════════════════════ */
.cfg-box {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;
}
.cfg-title { font-size:0.9rem; font-weight:700; color:#92400E; margin-bottom:8px; }
.cfg-body  { font-size:0.85rem; color:#78350F; line-height:1.65; }
.cfg-code  {
    font-family:'JetBrains Mono',monospace;
    background:#FEF3C7; border-radius:8px;
    padding:10px 14px; font-size:0.82rem;
    color:#92400E; margin-top:10px; display:block;
}

/* ═══ MISC ════════════════════════════════════════════════ */
.stSpinner > div { color:#6366F1 !important; }
.stAlert { border-radius:12px !important; font-size:0.88rem !important; }
div[data-testid="stExpander"] {
    border:1px solid #E2E8F0 !important; border-radius:12px !important;
    overflow:hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Category metadata ─────────────────────────────────────────────────
CAT_META = {
    "Fraud":       {"icon":"🔴","sub":"Unauthorized activity","cls":"fraud"},
    "Account":     {"icon":"🟡","sub":"Login & access issues","cls":"account"},
    "Payment":     {"icon":"🔵","sub":"Transaction problems","cls":"payment"},
    "Credit Card": {"icon":"🟣","sub":"Card issues & disputes","cls":"credit"},
    "Loan":        {"icon":"🟢","sub":"EMI & disbursement","cls":"loan"},
    "Other":       {"icon":"⚪","sub":"General complaint","cls":"other"},
}


def parse_and_render(category: str, resp: str, is_groq: bool):
    lines   = resp.strip().split("\n")
    message, steps, mode = "", [], None
    SKIP    = {"───","⏱","📞","📋","🔧","💬"}

    for line in lines:
        s = line.strip()
        if not s: continue
        if s.startswith("💬"):
            mode   = "msg"
            inline = s.replace("💬 Message:","").replace("💬","").strip()
            if inline: message = inline
        elif s.startswith("🔧"):
            mode = "steps"
        elif any(s.startswith(c) for c in SKIP):
            continue
        elif mode == "msg":
            message = (message + " " + s).strip()
        elif mode == "steps":
            step = s.lstrip("0123456789. ").strip()
            if step: steps.append(step)

    meta   = CAT_META.get(category, CAT_META["Other"])
    pc     = "mpill groq" if is_groq else "mpill local"
    tag    = "Groq · Llama 3.1" if is_groq else "T5 · Local"

    steps_html = "".join(
        f'<div class="step-row"><div class="snum">{i}</div>'
        f'<div class="stxt">{s}</div></div>'
        for i, s in enumerate(steps, 1)
    )

    st.markdown(f"""
<div class="result-outer"><div class="rc">
  <div class="rc-head {meta['cls']}">
    <div style="display:flex;align-items:center">
      <span class="cat-icon">{meta['icon']}</span>
      <div>
        <div class="cat-name">{category}</div>
        <div class="cat-sub">{meta['sub']}</div>
      </div>
    </div>
    <span class="{pc}">{tag}</span>
  </div>
  <div class="rc-body">
    <div class="rlabel">Support Response</div>
    <div class="rmsg">{message}</div>
    {'<div class="rlabel">Recommended Steps</div><div class="steps-grid">' + steps_html + '</div>' if steps else ''}
  </div>
  <div class="rc-foot">
    <span class="foot-item">⏱ Resolution within <strong>24 hours</strong></span>
    <span class="foot-item">📞 <strong>1800-XXX-XXXX</strong></span>
  </div>
</div></div>
""", unsafe_allow_html=True)


def pick_random(exclude=""):
    pool = [e for e in EXAMPLES if e[1] != exclude]
    return random.choice(pool)


# ── HERO ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-glow-1"></div>
  <div class="hero-glow-2"></div>
  <div class="hero-grid"></div>
  <div class="hero-inner">
    <div class="hero-live"><div class="hero-live-dot"></div>Live System</div>
    <h1 class="hero-h1">Banking Support<br><em>AI Assistant</em></h1>
    <p class="hero-desc">Instantly classify customer complaints and generate<br>professional, empathetic responses in under 2 seconds.</p>
    <div class="hero-pills">
      <span class="hero-pill"><span>6</span> complaint categories</span>
      <span class="hero-pill"><span>2,400</span> training samples</span>
      <span class="hero-pill"><span>Llama 3.1</span> via Groq</span>
      <span class="hero-pill"><span>T5</span> local model</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────
tab_groq, tab_t5 = st.tabs(["⚡  Groq — Llama 3.1", "🧠  T5 — Local Model"])


# ══════════════════════════════════════════════════════════
# TAB 1 — GROQ
# ══════════════════════════════════════════════════════════
with tab_groq:

    groq_ready = is_configured()

    if groq_ready:
        if "groq_conn" not in st.session_state:
            with st.spinner("Connecting to Groq..."):
                st.session_state["groq_conn"] = test_connection()
        if st.session_state["groq_conn"]:
            st.markdown('<div class="status-row"><div class="sdot on"></div>'
                        '<span class="stext"><strong>Groq connected</strong> — Llama 3.1 8B ready</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-row"><div class="sdot off"></div>'
                        '<span class="stext">Cannot reach Groq — check API key in config.py</span></div>',
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cfg-box">
          <div class="cfg-title">⚙️ One-time setup — add your Groq API key</div>
          <div class="cfg-body">Open <code>config.py</code> in the project root and paste your key.
          Free key at <strong>console.groq.com</strong> — 14,400 req/day, no card needed.</div>
          <code class="cfg-code">GROQ_API_KEY = "gsk_your_actual_key_here"</code>
        </div>""", unsafe_allow_html=True)

    # ── ANALYZE SECTION ───────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title">Analyze a Complaint</div>
    </div>
    """, unsafe_allow_html=True)

    if "g_ver" not in st.session_state:
        st.session_state["g_ver"]  = 0
        st.session_state["g_text"] = ""
        st.session_state["g_cat"]  = ""

    g_input = st.text_area(
        "groq_input",
        value=st.session_state["g_text"],
        height=130,
        placeholder="Describe the customer complaint in detail...\ne.g. My account was hacked and Rs 42,000 was transferred without my knowledge.",
        key=f"g_ta_{st.session_state['g_ver']}",
    )

    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    g_analyze = st.button("🔍  Analyze Complaint", key="g_go", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if g_analyze:
        text = g_input.strip()
        if not text:
            st.warning("Please type a complaint or use the Randomizer below.")
        elif not groq_ready:
            st.warning("Add your Groq API key to config.py first.")
        elif not st.session_state.get("groq_conn", False):
            st.warning("Cannot connect to Groq — check config.py.")
        else:
            with st.spinner("Llama 3.1 is reading your complaint..."):
                try:
                    cat, _ = classify_with_groq(text)
                    resp   = generate_response_with_groq(text, cat)
                    parse_and_render(cat, resp, is_groq=True)
                except RuntimeError as e:
                    st.error(str(e))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── RANDOMIZER SECTION — visually separate ────────────
    st.markdown("""
    <div class="rand-card">
      <div class="section-title" style="color:#0F766E">Try a Random Complaint</div>
      <p style="font-size:0.85rem;color:#134E4A;margin:0 0 14px;line-height:1.6;">
        Not sure what to test? Click below to load a real example complaint from 6 categories.
        Each click picks a different one.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rand-btn">', unsafe_allow_html=True)
    g_rand = st.button("🎲  Load Random Complaint", key="g_rand", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if g_rand:
        cat_hint, picked = pick_random(st.session_state["g_text"])
        st.session_state["g_text"] = picked
        st.session_state["g_cat"]  = cat_hint
        st.session_state["g_ver"] += 1
        st.rerun()

    if st.session_state["g_text"] and st.session_state.get("g_cat"):
        st.markdown(
            f'<div class="rand-preview">'
            f'<div class="rand-cat-tag">{st.session_state["g_cat"]}</div><br>'
            f'{st.session_state["g_text"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption("👆 This complaint is loaded in the Analyze box above — click Analyze to process it.")


# ══════════════════════════════════════════════════════════
# TAB 2 — T5 / KEYWORD
# ══════════════════════════════════════════════════════════
with tab_t5:

    from pipeline.model_loader import load_model as _lm
    _, _, _, t5_on = _lm()

    if t5_on:
        st.markdown('<div class="status-row"><div class="sdot on"></div>'
                    '<span class="stext"><strong>T5 model loaded</strong> — fine-tuned on 2,400 complaints</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-row"><div class="sdot off"></div>'
                    '<span class="stext"><strong>Keyword mode</strong> — fast rule-based classifier active</span></div>',
                    unsafe_allow_html=True)

    # ── ANALYZE SECTION ───────────────────────────────────
    st.markdown("""
    <div class="section-card">
      <div class="section-title">Analyze a Complaint</div>
    </div>
    """, unsafe_allow_html=True)

    if "t_ver" not in st.session_state:
        st.session_state["t_ver"]  = 0
        st.session_state["t_text"] = ""
        st.session_state["t_cat"]  = ""

    t_input = st.text_area(
        "t5_input",
        value=st.session_state["t_text"],
        height=130,
        placeholder="Describe the customer complaint in detail...\ne.g. My UPI payment failed but the money was deducted.",
        key=f"t_ta_{st.session_state['t_ver']}",
    )

    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    t_analyze = st.button("🔍  Analyze Complaint", key="t_go", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if t_analyze:
        text = t_input.strip()
        if not text:
            st.warning("Please type a complaint or use the Randomizer below.")
        else:
            with st.spinner("Analyzing complaint..."):
                cat, _ = classify_ticket(text)
                resp   = generate_response(text, cat)
            parse_and_render(cat, resp, is_groq=False)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── RANDOMIZER SECTION ────────────────────────────────
    st.markdown("""
    <div class="rand-card">
      <div class="section-title" style="color:#0F766E">Try a Random Complaint</div>
      <p style="font-size:0.85rem;color:#134E4A;margin:0 0 14px;line-height:1.6;">
        Load a real example from our dataset — covers all 6 complaint categories.
        Each click picks a different one.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rand-btn">', unsafe_allow_html=True)
    t_rand = st.button("🎲  Load Random Complaint", key="t_rand", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if t_rand:
        cat_hint, picked = pick_random(st.session_state["t_text"])
        st.session_state["t_text"] = picked
        st.session_state["t_cat"]  = cat_hint
        st.session_state["t_ver"] += 1
        st.rerun()

    if st.session_state["t_text"] and st.session_state.get("t_cat"):
        st.markdown(
            f'<div class="rand-preview">'
            f'<div class="rand-cat-tag">{st.session_state["t_cat"]}</div><br>'
            f'{st.session_state["t_text"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption("👆 This complaint is loaded in the Analyze box above — click Analyze to process it.")

    if not t5_on:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        with st.expander("🚀 Upgrade to T5 model for better accuracy"):
            st.code("python pipeline/train.py\n# ~10 min GPU  ·  ~1-2 hrs CPU\n# Restart Streamlit after training",
                    language="bash")

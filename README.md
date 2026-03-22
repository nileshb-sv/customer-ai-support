# 🏦 AI Customer Support Agent

End-to-end banking complaint classifier + automated response generator.  
Two modes — Groq LLM (real AI) and T5 local model.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

---

## 📁 Structure

```
customer_ai/
├── app/
│   └── streamlit_app.py        # UI — 2 tabs: Groq + T5
├── data/
│   ├── enterprise_complaints.csv
│   └── training_dataset.csv
├── models/
│   └── support_llm/            # Created after training
├── pipeline/
│   ├── groq_engine.py          # Groq API (Llama 3.1) ← real LLM
│   ├── model_loader.py         # Shared T5 loader (loads once)
│   ├── predict.py              # Keyword + T5 classifier
│   ├── preprocess.py           # Data utilities
│   ├── reply_engine.py         # Template + T5 responses
│   └── train.py                # T5 fine-tuning
├── main.py                     # CLI
├── check_accuracy.py           # Evaluation
├── requirements.txt
└── README.md
```

---

## ⚡ Tab 1 — Groq API (Real LLM)

Get a **free** API key at https://console.groq.com (14,400 req/day).

```bash
# Test from CLI
python main.py --groq YOUR_API_KEY --samples 5
```

What Groq does:
- Classifies the complaint using **Llama 3.1 8B**
- Generates a **specific, contextual reply** that references the actual complaint
- No keyword matching, no templates

---

## 🧠 Tab 2 — T5 Local Model

Works immediately without training (keyword mode).  
Train for better edge-case accuracy:

```bash
python pipeline/train.py
python pipeline/train.py --epochs 5
```

```bash
# Evaluate accuracy
python check_accuracy.py --samples 200
```

---

## 🔑 Key Design Points

- `classify_ticket()` always returns `(category: str, confidence: float)` — consistent everywhere
- T5 loaded once via `model_loader.py` — no double memory usage
- `normalize_label()` handles `credit_card` / `Credit Card` / `CREDIT CARD` correctly
- API key entered in UI — never hardcoded in code
- Absolute paths via `Path(__file__).resolve()` — no path bugs

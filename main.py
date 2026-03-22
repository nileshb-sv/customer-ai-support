"""
main.py — CLI runner

Usage:
    python main.py
    python main.py --samples 20
    python main.py --groq YOUR_API_KEY --samples 5
"""
import argparse, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocess  import load_labeled_data
from pipeline.predict     import classify_ticket
from pipeline.reply_engine import generate_response


def run_t5(df, n):
    print(f"\n{'='*70}\nT5 / KEYWORD MODE — {n} samples\n{'='*70}")
    correct = 0
    for i, (_, row) in enumerate(df.sample(n).iterrows()):
        ticket, actual = row["ticket"], row["label"]
        cat, conf = classify_ticket(ticket)
        match = actual.lower().replace("_"," ") == cat.lower()
        if match: correct += 1
        print(f"\n[{i+1}] {ticket[:120]}...")
        print(f"  Predicted : {cat} ({conf:.0%})  |  Actual: {actual}  {'✅' if match else '❌'}")
        print(f"  Response  :\n{generate_response(ticket, cat)}")
    print(f"\nAccuracy: {correct}/{n} = {correct/n*100:.1f}%")


def run_groq(df, n, api_key):
    from pipeline.groq_engine import classify_with_groq, generate_response_with_groq
    print(f"\n{'='*70}\nGROQ / LLAMA 3.1 MODE — {n} samples\n{'='*70}")
    correct = 0
    for i, (_, row) in enumerate(df.sample(n).iterrows()):
        ticket, actual = row["ticket"], row["label"]
        cat, _ = classify_with_groq(ticket, api_key)
        match  = actual.lower().replace("_"," ") == cat.lower()
        if match: correct += 1
        print(f"\n[{i+1}] {ticket[:120]}...")
        print(f"  Predicted: {cat}  |  Actual: {actual}  {'✅' if match else '❌'}")
        print(f"  Response :\n{generate_response_with_groq(ticket, cat, api_key)}")
    print(f"\nAccuracy: {correct}/{n} = {correct/n*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--groq",    type=str, default=None, help="Groq API key")
    a  = ap.parse_args()

    df = load_labeled_data("./data/enterprise_complaints.csv")
    if a.groq:
        run_groq(df, a.samples, a.groq)
    else:
        run_t5(df, a.samples)

#!/usr/bin/env python3
"""check_accuracy.py — Evaluate keyword/T5 classification accuracy"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pipeline.predict    import classify_ticket
from pipeline.preprocess import load_labeled_data


def evaluate(n):
    df = load_labeled_data("./data/enterprise_complaints.csv")
    n  = min(n, len(df))
    sample = df.sample(n, random_state=42).reset_index(drop=True)

    print(f"\n{'='*90}\nACCURACY EVALUATION — {n} samples\n{'='*90}")
    correct, results = 0, []

    for i, row in sample.iterrows():
        text, actual = row["ticket"], row["label"]
        pred, conf   = classify_ticket(text)
        match        = actual.lower().replace("_"," ") == pred.lower()
        if match: correct += 1
        results.append({"actual":actual,"predicted":pred,"conf":conf,"match":match,"text":text[:80]})
        mark = "✅" if match else "❌"
        print(f"{i+1:<4} {actual:<18} {pred:<18} {conf:.2f}  {mark}  {text[:45]}...")

    acc = correct / n * 100
    print(f"\n{'─'*90}\nAccuracy: {correct}/{n} = {acc:.1f}%\n")

    print("CATEGORY BREAKDOWN")
    for cat in sorted(set(r["actual"] for r in results)):
        sub  = [r for r in results if r["actual"].lower()==cat.lower()]
        ok   = sum(1 for r in sub if r["match"])
        print(f"  {cat:<20} {ok}/{len(sub)}  ({ok/len(sub)*100:.0f}%)")

    bad = [r for r in results if not r["match"]]
    if bad:
        print(f"\nMISCLASSIFICATIONS ({len(bad)} total, showing 5)")
        for r in bad[:5]:
            print(f"  actual={r['actual']} predicted={r['predicted']} | {r['text']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    evaluate(ap.parse_args().samples)

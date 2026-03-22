@echo off
echo Removing API key from config.py and git history...

:: Overwrite config.py with NO hardcoded key
(
echo """
echo config.py - API key loaded from environment or Streamlit secrets only.
echo Never hardcode your key here.
echo """
echo.
echo import os
echo.
echo try:
echo     import streamlit as st
echo     GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
echo except Exception:
echo     GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
echo.
echo GROQ_MODEL_CLASSIFY = "llama-3.1-8b-instant"
echo GROQ_MODEL_REPLY    = "llama-3.1-8b-instant"
echo.
echo DATA_PATH     = "./data/enterprise_complaints.csv"
echo TRAINING_PATH = "./data/training_dataset.csv"
echo MODEL_PATH    = "./models/support_llm"
) > config.py

:: Remove config.py from git tracking entirely
git rm --cached config.py

:: Make sure config.py is in .gitignore
echo config.py >> .gitignore

:: Stage the changes
git add .gitignore
git commit -m "Remove API key - config.py excluded from repo"

:: Force push (rewrites history to remove the secret)
git push -u origin main --force

echo.
echo ============================================================
echo Done! config.py is now excluded from GitHub.
echo Your API key will be added via Streamlit Cloud Secrets only.
echo ============================================================
pause

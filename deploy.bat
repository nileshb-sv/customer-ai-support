@echo off
echo Fixing Git setup...

:: Set your identity
git config --global user.email "nileshb@example.com"
git config --global user.name "nileshb-sv"

:: Remove old remote and re-add
git remote remove origin
git remote add origin https://github.com/nileshb-sv/customer-ai-support.git

:: Stage and commit everything
git add .
git commit -m "AI Customer Support — Groq + T5 Banking Assistant"

:: Push
git branch -M main
git push -u origin main

echo.
echo Done! Check https://github.com/nileshb-sv/customer-ai-support
pause
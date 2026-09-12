@echo off
title Task 2 - Interactive Sentiment Classifier
cd /d "%~dp0"
echo ======================================================================
echo       TASK 2: INTERACTIVE SENTIMENT TESTER (Type your sentences)
echo ======================================================================
".venv\Scripts\python.exe" main.py --interactive
echo.
pause

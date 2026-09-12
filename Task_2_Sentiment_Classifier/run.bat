@echo off
title Task 2 - Sentiment Classifier Pipeline
cd /d "%~dp0"
echo ======================================================================
echo           RUNNING TASK 2: SENTIMENT CLASSIFIER PIPELINE
echo ======================================================================
".venv\Scripts\python.exe" main.py
echo.
pause

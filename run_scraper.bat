@echo off
setlocal

REM --- Optional: create venv ---
REM python -m venv .venv
REM call .venv\Scripts\activate

REM --- Install dependencies (uncomment if you want auto-install) ---
REM python -m pip install -r requirements.txt
REM python -m playwright install chromium

REM --- Defaults ---
set INPUT=ENRICHED DATA FOR B2B OUTREACH.xlsx
set OUTPUT=ENRICHED DATA FOR B2B OUTREACH_scraped.csv

python scraper_magical.py --input "%INPUT%" --output "%OUTPUT%" --write-xlsx

echo.
echo Done. Output: %OUTPUT%
pause
endlocal

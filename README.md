# Magical Website Scraper (Requests-first + Playwright fallback)

This scraper enriches your Excel list of websites by crawling **top internal pages** and extracting **full text** + a **short preview** per site.
It is resumable, logs errors clearly, and writes detailed status columns.

## What it does
- Reads an Excel file that contains a `website` column
- For each site:
  - Fetches pages using **requests** (fast)
  - Automatically falls back to **Playwright** when pages are JS-rendered or blocked
  - Crawls **top internal pages** (About/Contact/Services/Products/etc.) + (optional) Sitemap hints
  - Extracts clean readable text and combines it across pages
- Outputs:
  - CSV (always) + optional XLSX
  - Full text + preview
  - Diagnostics: status codes, methods used, pages visited, errors, timing, truncation flags

## Setup (Windows)
1) Install Python 3.10+.
2) Open Command Prompt in this folder and run:
   - `python -m pip install -r requirements.txt`
   - `python -m playwright install chromium`

## Run
- Double-click `run_scraper.bat` (uses defaults)
OR run manually:
- `python scraper_magical.py --input "ENRICHED DATA FOR B2B OUTREACH.xlsx" --output "output.csv" --write-xlsx`

## Resume behavior
- If you re-run with the same output path, rows with `scrape_status == "done"` are skipped automatically.

## Notes
- "Complete website" is interpreted as: homepage + a safe set of high-signal internal pages and (optionally) a small number of sitemap-picked URLs.
- You can tune crawl limits:
  - `--max-pages 15` (default 12)
  - `--use-sitemap` (off by default)
  - `--max-chars-full 500000` (default 250000)

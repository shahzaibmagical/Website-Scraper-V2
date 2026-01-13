#!/usr/bin/env python3
"""Entry-point for the GitHub Actions scraper automation.

Responsibilities
- Accept a Google Drive link (file or folder)
- Download the input file(s) from Drive
- Run scraper_magical.py on each input file
- Generate outputs and upload-ready artifacts (CSV/XLSX + logs + summary)

Design goal
Match the operational standards of the existing "close existing deals" automation:
- strict/defensive input handling
- deterministic artifact folder
- human-readable run summary for Actions UI
- failures leave behind helpful artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# When executed as `python scripts/run_actions_pipeline.py`, the scripts/ folder is
# on sys.path, so we import sibling module directly.
from gdrive_fetch import download_drive_inputs


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")



def write_progress_snapshot(path: Path, snapshot: Dict[str, Any]) -> None:
    """Write a small progress file for near-real-time visibility in logs/artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def progress_md(snapshot: Dict[str, Any]) -> str:
    return (
        f"### Live progress\n"
        f"- Files: {snapshot.get('files_completed')}/{snapshot.get('files_total')}\n"
        f"- Rows: {snapshot.get('rows_done')}/{snapshot.get('rows_total')} done\n"
        f"- Pending: {snapshot.get('rows_pending')}\n"
        f"- Errors: {snapshot.get('rows_error')}\n"
        f"- Skipped: {snapshot.get('rows_skipped')}\n"
        f"- Updated (UTC): {snapshot.get('utc_now')}\n"
    )
def safe_run_scraper(
    repo_root: Path,
    input_path: Path,
    output_csv: Path,
    checkpoint_path: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run the scraper for a single input file and return a summary dict."""

    t0 = time.time()

    cmd = [
        sys.executable,
        str(repo_root / "scraper_magical.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_csv),
        "--checkpoint",
        str(checkpoint_path),
        "--log",
        str(log_path),
        "--max-pages",
        str(args.max_pages),
        "--timeout",
        str(args.timeout),
        "--playwright-timeout-ms",
        str(args.playwright_timeout_ms),
        "--min-text",
        str(args.min_text),
        "--max-chars-full",
        str(args.max_chars_full),
        "--preview-chars",
        str(args.preview_chars),
        "--per-page-delay",
        str(args.per_page_delay),
        "--save-every",
        str(args.save_every),
        "--site-workers",
        str(args.site_workers),
        "--playwright-workers",
        str(args.playwright_workers),
        "--site-time-budget-sec",
        str(args.site_time_budget_sec),
    ]

    if args.use_sitemap:
        cmd.append("--use-sitemap")
    if args.write_xlsx:
        cmd.append("--write-xlsx")
    if args.headless:
        cmd.append("--headless")
    else:
        cmd.append("--no-headless")

    # Capture stdout/stderr into the per-file log (the scraper already logs to file,
    # but this catches Python-level tracebacks in CI as well).
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # Stream output in real time to BOTH GitHub Actions logs and the per-file log.
    # This enables "kinda realtime" progress stats (the scraper emits progress lines).
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n\n--- subprocess live output ---\n")
        if proc.stdout is not None:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                print(line.rstrip("\n"), flush=True)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"scraper_magical.py exited with code {proc.returncode}. See log: {log_path.name}")

    # Summarize output
    df = pd.read_csv(output_csv, low_memory=False)
    status = df.get("scrape_status")
    counts = status.value_counts(dropna=False).to_dict() if status is not None else {}

    done = int(counts.get("done", 0))
    error = int(counts.get("error", 0))
    skipped = int(counts.get("skipped", 0))

    duration = round(time.time() - t0, 3)
    return {
        "input_file": input_path.name,
        "output_csv": output_csv.name,
        "output_xlsx": (output_csv.with_suffix(".xlsx").name if args.write_xlsx else ""),
        "rows_total": int(len(df)),
        "rows_done": done,
        "rows_error": error,
        "rows_skipped": skipped,
        "duration_sec": duration,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-link", required=True, help="Google Drive link to a file or folder.")

    # Output
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts output directory.")
    ap.add_argument("--write-xlsx", action="store_true", help="Also emit XLSX next to the output CSV.")

    # Scraper config passthrough
    ap.add_argument("--max-pages", type=int, default=12)
    ap.add_argument("--use-sitemap", action="store_true")
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument("--playwright-timeout-ms", type=int, default=25000)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--no-headless", action="store_false", dest="headless")
    ap.add_argument("--min-text", type=int, default=900)
    ap.add_argument("--max-chars-full", type=int, default=250000)
    ap.add_argument("--preview-chars", type=int, default=8000)
    ap.add_argument("--per-page-delay", type=float, default=0.6)
    ap.add_argument("--save-every", type=int, default=10)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifacts = Path(args.artifacts_dir)
    inputs_dir = artifacts / "inputs"
    outputs_dir = artifacts / "outputs"
    logs_dir = artifacts / "logs"
    checkpoints_dir = artifacts / "checkpoints"
    for d in [inputs_dir, outputs_dir, logs_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    started = utc_now_iso()

    # Download inputs
    downloaded = download_drive_inputs(args.drive_link, dest_dir=inputs_dir, allowed_exts=(".xlsx", ".csv"))

    # Initial progress snapshot
    write_progress_snapshot(
        artifacts / "progress.json",
        {
            "utc_now": utc_now_iso(),
            "files_completed": 0,
            "files_total": len(downloaded),
            "rows_total": 0,
            "rows_done": 0,
            "rows_error": 0,
            "rows_skipped": 0,
            "rows_pending": 0,
        },
    )
    write_text(artifacts / "progress.md", progress_md(json.loads((artifacts / "progress.json").read_text())))

    # Process each file
    per_file: List[Dict[str, Any]] = []
    failures: List[str] = []

    for i, input_path in enumerate(downloaded, start=1):
        stem = input_path.stem
        out_csv = outputs_dir / f"{stem}__scraped.csv"
        ck = checkpoints_dir / f"{stem}__checkpoint.json"
        log = logs_dir / f"{stem}__scraper.log"

        try:
            info = safe_run_scraper(
                repo_root=repo_root,
                input_path=input_path,
                output_csv=out_csv,
                checkpoint_path=ck,
                log_path=log,
                args=args,
            )
            per_file.append(info)
            # Update aggregated progress snapshot (useful for logs/artifacts during long runs)
            agg_total = sum(int(x.get("rows_total", 0)) for x in per_file)
            agg_done = sum(int(x.get("rows_done", 0)) for x in per_file)
            agg_err = sum(int(x.get("rows_error", 0)) for x in per_file)
            agg_skip = sum(int(x.get("rows_skipped", 0)) for x in per_file)
            agg_pending = max(0, agg_total - (agg_done + agg_err + agg_skip))

            snapshot = {
                "utc_now": utc_now_iso(),
                "files_completed": len(per_file),
                "files_total": len(downloaded),
                "rows_total": agg_total,
                "rows_done": agg_done,
                "rows_error": agg_err,
                "rows_skipped": agg_skip,
                "rows_pending": agg_pending,
            }

            write_progress_snapshot(artifacts / "progress.json", snapshot)
            write_text(artifacts / "progress.md", progress_md(snapshot))

            # Nice UI signal while the job is still running
            print(
                f"::notice title=Pipeline Progress::files={snapshot['files_completed']}/{snapshot['files_total']} "
                f"rows_done={snapshot['rows_done']}/{snapshot['rows_total']} pending={snapshot['rows_pending']} "
                f"errors={snapshot['rows_error']} skipped={snapshot['rows_skipped']}",
                flush=True,
            )

        except Exception as e:
            failures.append(f"{input_path.name}: {e}")
            # keep going; we still want artifacts for other files

    finished = utc_now_iso()

    # If everything failed, treat as pipeline failure (infra / auth / etc.)
    if not per_file:
        err = "All input files failed to process.\n" + "\n".join(failures)
        write_text(artifacts / "runner_error.txt", err)
        raise SystemExit(2)

    # Build summary
    total_rows = sum(int(x.get("rows_total", 0)) for x in per_file)
    total_done = sum(int(x.get("rows_done", 0)) for x in per_file)
    total_error = sum(int(x.get("rows_error", 0)) for x in per_file)
    total_skipped = sum(int(x.get("rows_skipped", 0)) for x in per_file)
    total_pending = max(0, total_rows - (total_done + total_error + total_skipped))

    summary: Dict[str, Any] = {
        "utc_started": started,
        "utc_finished": finished,
        "drive_link": args.drive_link,
        "inputs_downloaded": [p.name for p in downloaded],
        "results": per_file,
        "totals": {
            "rows_total": total_rows,
            "rows_done": total_done,
            "rows_error": total_error,
            "rows_skipped": total_skipped,
        },
        "pipeline_failures": failures,
        "config": {
            "max_pages": args.max_pages,
            "use_sitemap": bool(args.use_sitemap),
            "timeout": args.timeout,
            "playwright_timeout_ms": args.playwright_timeout_ms,
            "headless": bool(args.headless),
            "min_text": args.min_text,
            "max_chars_full": args.max_chars_full,
            "preview_chars": args.preview_chars,
            "per_page_delay": args.per_page_delay,
            "save_every": args.save_every,
            "write_xlsx": bool(args.write_xlsx),
            "has_service_account": bool(os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON") or os.environ.get("GDRIVE_SERVICE_ACCOUNT_B64")),
        },
    }

    (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Human-readable markdown summary for Actions UI
    md = []
    md.append(f"**UTC started:** {started}")
    md.append(f"**UTC finished:** {finished}")
    md.append("")
    md.append("### Totals")
    md.append(f"- Rows total: **{total_rows}**")
    md.append(f"- Done: **{total_done}**")
    md.append(f"- Error: **{total_error}**")
    md.append(f"- Skipped: **{total_skipped}**")
    md.append("")
    md.append("### Per input file")
    for r in per_file:
        md.append(
            f"- **{r['input_file']}** → {r['rows_done']}/{r['rows_total']} done, {r['rows_error']} error, {r['rows_skipped']} skipped "
            f"(output: `{r['output_csv']}`{', ' + '`' + r['output_xlsx'] + '`' if r.get('output_xlsx') else ''})"
        )
    if failures:
        md.append("")
        md.append("### Pipeline-level warnings")
        for f in failures[:25]:
            md.append(f"- {f}")
        if len(failures) > 25:
            md.append(f"- ... and {len(failures) - 25} more")

    (artifacts / "run_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Magical Website Scraper

Features:
- Saves full text + short preview
- Requests-first + Playwright fallback
- Crawls top internal pages safely (+ optional sitemap hints)
- Detailed status/error columns
- Resumable via output file (skips done rows) + checkpoint JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional (installed via requirements.txt)
try:
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

try:
    from readability import Document
except Exception:  # pragma: no cover
    Document = None


KEYWORD_PATH_HINTS = [
    "about", "contact", "services", "service", "products", "product", "solutions",
    "pricing", "plans", "menu", "locations", "location", "stores", "store",
    "team", "people", "leadership", "company", "who-we-are", "our-story",
    "careers", "jobs", "faq", "support", "help", "shipping", "returns",
]

COMMON_PATHS = [
    "/about", "/about-us", "/company", "/who-we-are", "/our-story",
    "/contact", "/contact-us", "/locations", "/location",
    "/services", "/products", "/solutions", "/pricing", "/plans",
    "/menu", "/shop", "/collections", "/catalog",
    "/faq", "/support", "/help",
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}



def emit_progress(message: str) -> None:
    """Emit progress in a GitHub Actions-friendly way (and still safe locally)."""
    # Plain line for logs
    try:
        print(message, flush=True)
    except Exception:
        pass

    # GitHub Actions annotation (shows nicely in UI)
    if os.getenv("GITHUB_ACTIONS") == "true":
        # Limit message length for annotations
        msg = message
        if len(msg) > 900:
            msg = msg[:900] + "…"
        print(f"::notice title=Scraper Progress::{msg}", flush=True)

@dataclass
class PageResult:
    url: str
    ok: bool
    status_code: Optional[int]
    method: str  # requests|playwright
    bytes: int
    text_len: int
    error: str = ""


@dataclass
class SiteResult:
    input_url: str
    normalized_url: str
    final_url: str = ""
    scrape_status: str = "error"  # done|error|skipped
    error_reason: str = ""
    pages_attempted: int = 0
    pages_scraped: int = 0
    methods_used: str = ""  # requests/playwright/both
    http_statuses: str = ""  # e.g., 200;200;403
    total_bytes: int = 0
    total_text_len: int = 0
    truncated_full: bool = False
    truncated_preview: bool = False
    duration_sec: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    visited_urls: str = ""  # semicolon-separated

    # Extra enrichment (doesn't break existing automation; adds more usefulness)
    site_title: str = ""
    site_meta_description: str = ""
    site_headings: str = ""  # JSON array string
    main_text_len: int = 0


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("scraper_magical")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def build_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update(DEFAULT_HEADERS)
    return sess


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parsed = urlparse(url)
    # drop fragments
    parsed = parsed._replace(fragment="")
    # normalize path
    path = parsed.path or "/"
    parsed = parsed._replace(path=path)
    return urlunparse(parsed)


def domain_key(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return urlparse(url).netloc.lower()
    return f"{ext.domain}.{ext.suffix}".lower()


def same_domain(url_a: str, url_b: str) -> bool:
    try:
        return domain_key(url_a) == domain_key(url_b)
    except Exception:
        return False


def is_probably_html(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return ("text/html" in ctype) or ("application/xhtml+xml" in ctype) or (ctype == "")


def looks_like_js_shell(html: str) -> bool:
    # heuristic: huge script tags + tiny visible text container
    if not html:
        return True
    # common app shells
    patterns = [
        r'id="__next"',
        r'id="root"',
        r'data-reactroot',
        r'window\.__INITIAL_STATE__',
        r'__NUXT__',
    ]
    if any(re.search(p, html, re.I) for p in patterns):
        # if body text is very small, likely needs JS
        soup = BeautifulSoup(html, "lxml")
        txt = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return len(txt) < 800
    return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"[\u00A0\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    # Remove repeated nav-like crumbs
    return text.strip()


def extract_readable_text_from_html(html: str, url: str = "") -> str:
    if not html:
        return ""
    # 1) trafilatura if available (best for messy pages)
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html, url=url, include_comments=False, include_tables=True, favor_recall=True
            )
            if extracted and len(extracted.strip()) > 200:
                return clean_text(extracted)
        except Exception:
            pass


def extract_texts_and_meta_from_html(html: str, url: str = "") -> Dict[str, str]:
    """Return multiple views of content for better completeness.

    - main_text: readability/trafilatura extracted (high-signal)
    - full_text: raw-ish text from the whole document
    - title: <title> or first h1
    - meta_description: <meta name="description">
    - headings: JSON string of top headings (h1-h3), de-duped, up to 50
    """
    html = html or ""
    main_text = extract_readable_text_from_html(html, url=url)

    # full text (keep it simple + fast)
    full_text = ""
    title = ""
    meta_desc = ""
    headings: List[str] = []
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            title = (soup.title.string or "").strip()
        md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if md and md.get("content"):
            meta_desc = str(md.get("content")).strip()

        # headings
        seen = set()
        for tag in soup.find_all(["h1", "h2", "h3"]):
            t = " ".join(tag.get_text(" ", strip=True).split())
            if t and t.lower() not in seen:
                seen.add(t.lower())
                headings.append(t)
            if len(headings) >= 50:
                break

        # full text
        full_text = soup.get_text(" ", strip=True)
        full_text = " ".join(full_text.split())
    except Exception:
        pass

    if not title and headings:
        title = headings[0]

    return {
        "main_text": main_text or "",
        "full_text": full_text or (main_text or ""),
        "title": title,
        "meta_description": meta_desc,
        "headings": json.dumps(headings, ensure_ascii=False),
    }
    # 2) readability-lxml (good main content)
    if Document is not None:
        try:
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            soup = BeautifulSoup(summary_html, "lxml")
            txt = soup.get_text("\n", strip=True)
            if txt and len(txt) > 200:
                return clean_text(txt)
        except Exception:
            pass

    # 3) fallback: BeautifulSoup with gentle tag removal
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # Remove common clutter blocks by role/class when present
    for bad in soup.select(
        "nav, header, footer, aside, iframe, form, svg, canvas, "
        "[role='navigation'], [role='banner'], [role='contentinfo']"
    ):
        # keep if it contains lots of useful text (rare)
        if len(bad.get_text(" ", strip=True)) < 400:
            bad.decompose()

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return clean_text(text)


def fetch_with_requests(session: requests.Session, url: str, timeout: float) -> Tuple[Optional[requests.Response], str]:
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        return resp, ""
    except requests.RequestException as e:
        return None, str(e)


def fetch_with_playwright(
    url: str,
    timeout_ms: int,
    headless: bool,
    user_agent: str,
    semaphore: Optional["threading.BoundedSemaphore"] = None,
) -> Tuple[str, str]:
    """Returns (html, error).

    Upgrades:
    - Optional concurrency limit via semaphore (prevents spawning too many Chromium instances)
    - Blocks heavy resources (images/fonts/media/stylesheets) to speed up
    - Lightweight scroll + common "load more / show more" clicks (bounded) for better coverage
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return "", f"Playwright not available: {e}"

    import threading  # local import to avoid adding global dependency order issues

    sem = semaphore
    if sem is None:
        class _Noop:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        sem_ctx = _Noop()
    else:
        sem_ctx = sem

    try:
        with sem_ctx:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=headless)
                context = browser.new_context(
                    user_agent=user_agent,
                    viewport={"width": 1280, "height": 800},
                )

                # Speed: abort non-essential resources
                def _route(route, request):
                    try:
                        if request.resource_type in {"image", "media", "font", "stylesheet"}:
                            return route.abort()
                    except Exception:
                        pass
                    return route.continue_()

                try:
                    context.route("**/*", _route)
                except Exception:
                    pass

                page = context.new_page()
                page.set_default_navigation_timeout(timeout_ms)
                page.set_default_timeout(timeout_ms)

                page.goto(url, wait_until="domcontentloaded")

                # Small settle for async content
                try:
                    page.wait_for_timeout(800)
                except Exception:
                    pass

                # Bounded scroll
                try:
                    last_h = 0
                    for _ in range(3):
                        h = page.evaluate("() => document.body.scrollHeight || 0")
                        if h and h > last_h:
                            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                            page.wait_for_timeout(700)
                            last_h = h
                        else:
                            break
                except Exception:
                    pass

                # Bounded "load more/show more"
                try:
                    selectors = [
                        "text=/load more/i",
                        "text=/show more/i",
                        "text=/read more/i",
                        "text=/more/i",
                        "button:has-text('Load more')",
                        "button:has-text('Show more')",
                    ]
                    clicks = 0
                    for sel in selectors:
                        if clicks >= 3:
                            break
                        try:
                            el = page.query_selector(sel)
                            if el:
                                el.click(timeout=1500)
                                page.wait_for_timeout(600)
                                clicks += 1
                        except Exception:
                            continue
                except Exception:
                    pass

                html = ""
                try:
                    html = page.content() or ""
                except Exception:
                    html = ""

                try:
                    context.close()
                except Exception:
                    pass
                try:
                    browser.close()
                except Exception:
                    pass

                if not html:
                    return "", "Playwright returned empty HTML"
                return html, ""
    except Exception as e:
        return "", f"Playwright fetch failed: {e}"
def pick_candidate_internal_links(base_url: str, html: str, max_links: int = 20) -> List[str]:
    soup = BeautifulSoup(html or "", "lxml")
    links: List[str] = []
    seen: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)
        if not abs_url or not same_domain(base_url, abs_url):
            continue

        # prefer keyword-y urls
        lower = abs_url.lower()
        score = 0
        for kw in KEYWORD_PATH_HINTS:
            if f"/{kw}" in lower or kw in lower:
                score += 2
        # shorter is usually better
        score += max(0, 3 - lower.count("/"))

        if abs_url not in seen:
            seen.add(abs_url)
            links.append((score, abs_url))

    # Add common paths
    for p in COMMON_PATHS:
        abs_url = normalize_url(urljoin(base_url, p))
        if abs_url and same_domain(base_url, abs_url) and abs_url not in seen:
            seen.add(abs_url)
            links.append((3, abs_url))

    links.sort(key=lambda x: x[0], reverse=True)
    out = [u for _, u in links][:max_links]
    # Ensure homepage first
    home = normalize_url(base_url)
    if home and home not in out:
        out.insert(0, home)
    else:
        # move to front
        out = [home] + [u for u in out if u != home]
    # Deduplicate while keeping order
    final: List[str] = []
    s2: Set[str] = set()
    for u in out:
        if u not in s2:
            s2.add(u)
            final.append(u)
    return final[:max_links]


def sitemap_hints(session: requests.Session, base_url: str, timeout: float, limit: int = 10) -> List[str]:
    """Lightweight sitemap support: grabs a few URLs with relevant keywords."""
    candidates = []
    for path in ["/sitemap.xml", "/sitemap_index.xml"]:
        url = normalize_url(urljoin(base_url, path))
        resp, err = fetch_with_requests(session, url, timeout=timeout)
        if err or resp is None:
            continue
        if resp.status_code >= 400:
            continue
        xml = resp.text or ""
        # extract <loc>...</loc>
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml, flags=re.I)
        for loc in locs:
            loc = normalize_url(loc)
            if loc and same_domain(base_url, loc):
                candidates.append(loc)
        if candidates:
            break

    if not candidates:
        return []

    def score(u: str) -> int:
        ul = u.lower()
        s = 0
        for kw in KEYWORD_PATH_HINTS:
            if kw in ul:
                s += 3
        # prefer not-too-deep urls
        s += max(0, 4 - ul.count("/"))
        return s

    uniq = []
    seen = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    uniq.sort(key=score, reverse=True)
    return uniq[:limit]


def combine_texts(pages: List[Tuple[str, str]]) -> str:
    """pages: list of (url, text)"""
    parts = []
    for url, text in pages:
        if not text:
            continue
        parts.append(f"URL: {url}\n{text}")
    return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(parts) if parts else ""


def scrape_site(
    session: requests.Session,
    input_url: str,
    max_pages: int,
    timeout: float,
    playwright_timeout_ms: int,
    headless: bool,
    min_text_for_requests_ok: int,
    use_sitemap: bool,
    max_chars_full: int,
    preview_chars: int,
    per_page_delay: float,
    logger: logging.Logger,
    playwright_semaphore: Optional["threading.BoundedSemaphore"] = None,
    site_time_budget_sec: int = 45,
) -> Tuple[SiteResult, str, str, List[PageResult], str]:
    started = time.time()
    sr = SiteResult(input_url=input_url, normalized_url="")
    sr.started_at = datetime.utcnow().isoformat() + "Z"

    norm = normalize_url(input_url)
    sr.normalized_url = norm
    if not norm:
        sr.scrape_status = "skipped"
        sr.error_reason = "empty_url"
        sr.finished_at = datetime.utcnow().isoformat() + "Z"
        return sr, "", "", [], ""

    visited: List[str] = []
    page_results: List[PageResult] = []
    extracted_pages: List[Tuple[str, str]] = []
    methods: Set[str] = set()
    statuses: List[str] = []

    # Step 1: fetch homepage with requests
    resp, err = fetch_with_requests(session, norm, timeout=timeout)
    homepage_html = ""
    final_url = norm
    if resp is not None:
        final_url = resp.url or norm
        sr.final_url = final_url
        statuses.append(str(resp.status_code))
        if resp.status_code < 400 and is_probably_html(resp):
            homepage_html = resp.text or ""
        else:
            err = err or f"HTTP {resp.status_code}"
    else:
        sr.final_url = norm

    # If homepage is blocked/JS shell/too short -> Playwright
    need_playwright_home = False
    if not homepage_html:
        need_playwright_home = True
    elif looks_like_js_shell(homepage_html):
        need_playwright_home = True
    else:
        txt0 = extract_readable_text_from_html(homepage_html, url=final_url)
        if len(txt0) < min_text_for_requests_ok:
            need_playwright_home = True

    if need_playwright_home:
        html_pw, err_pw = fetch_with_playwright(
            final_url, timeout_ms=playwright_timeout_ms, headless=headless, user_agent=DEFAULT_HEADERS["User-Agent"], semaphore=playwright_semaphore
        )
        if html_pw:
            homepage_html = html_pw
            methods.add("playwright")
        else:
            # if requests had something, keep it; otherwise fail
            if not homepage_html:
                sr.scrape_status = "error"
                sr.error_reason = f"homepage_failed: {err or ''} {err_pw}".strip()
                sr.duration_sec = round(time.time() - started, 3)
                sr.finished_at = datetime.utcnow().isoformat() + "Z"
                return sr, "", "", [], ""

    else:
        methods.add("requests")

    # homepage text
    meta0 = extract_texts_and_meta_from_html(homepage_html, url=final_url)
    homepage_text = meta0.get('main_text','')
    homepage_full_text = meta0.get('full_text','')
    sr.site_title = meta0.get('title','')
    sr.site_meta_description = meta0.get('meta_description','')
    sr.site_headings = meta0.get('headings','')
    
    extracted_pages.append((final_url, homepage_text))
    visited.append(final_url)
    page_results.append(PageResult(
        url=final_url,
        ok=bool(homepage_text),
        status_code=(resp.status_code if resp is not None else None),
        method=("playwright" if "playwright" in methods and "requests" not in methods else "requests"),
        bytes=len(homepage_html.encode("utf-8", errors="ignore")),
        text_len=len(homepage_text),
        error="" if homepage_text else "empty_text",
    ))
    sr.total_bytes += page_results[-1].bytes
    sr.total_text_len += page_results[-1].text_len

    # Step 2: pick internal links
    candidates = pick_candidate_internal_links(final_url, homepage_html, max_links=max_pages)
    if use_sitemap:
        hints = sitemap_hints(session, final_url, timeout=timeout, limit=max(0, max_pages - len(candidates)))
        for h in hints:
            if h not in candidates and len(candidates) < max_pages:
                candidates.append(h)

    # Step 3: crawl
    queued = [u for u in candidates if u not in set(visited)]
    for page_url in queued:
        if len(visited) >= max_pages:
            break
        # hard time budget per site
        if site_time_budget_sec and (time.time() - started) > float(site_time_budget_sec):
            logger.info("Time budget reached for %s; stopping crawl early", final_url)
            break

        # polite delay
        if per_page_delay > 0:
            time.sleep(per_page_delay)

        pr = PageResult(url=page_url, ok=False, status_code=None, method="requests", bytes=0, text_len=0, error="")
        html = ""
        resp2, err2 = fetch_with_requests(session, page_url, timeout=timeout)
        if resp2 is not None:
            pr.status_code = resp2.status_code
            statuses.append(str(resp2.status_code))
            if resp2.status_code < 400 and is_probably_html(resp2):
                html = resp2.text or ""
            else:
                err2 = err2 or f"HTTP {resp2.status_code}"
        else:
            pr.error = err2

        use_pw = False
        if not html:
            use_pw = True
        elif looks_like_js_shell(html):
            use_pw = True
        else:
            txt_tmp = extract_readable_text_from_html(html, url=page_url)
            if len(txt_tmp) < min_text_for_requests_ok:
                use_pw = True

        if use_pw:
            html_pw, err_pw = fetch_with_playwright(
                page_url, timeout_ms=playwright_timeout_ms, headless=headless, user_agent=DEFAULT_HEADERS["User-Agent"], semaphore=playwright_semaphore
            )
            if html_pw:
                html = html_pw
                pr.method = "playwright"
                methods.add("playwright")
            else:
                pr.method = "requests"
                methods.add("requests")
                pr.error = (pr.error + " " + err_pw).strip()
        else:
            pr.method = "requests"
            methods.add("requests")

        pr.bytes = len(html.encode("utf-8", errors="ignore")) if html else 0
        text = extract_readable_text_from_html(html, url=page_url) if html else ""
        pr.text_len = len(text)
        pr.ok = bool(text and pr.text_len > 100)
        if not pr.ok and not pr.error:
            pr.error = "empty_or_too_short"

        visited.append(page_url)
        page_results.append(pr)

        sr.total_bytes += pr.bytes
        sr.total_text_len += pr.text_len

        if pr.ok:
            extracted_pages.append((page_url, text))

    # Step 4: combine + truncation policy
    sr.pages_attempted = len(page_results)
    sr.pages_scraped = sum(1 for p in page_results if p.ok)
    sr.methods_used = "both" if len(methods) > 1 else (next(iter(methods)) if methods else "")
    sr.http_statuses = ";".join(statuses[:50])  # cap to keep file small
    sr.visited_urls = ";".join(visited[:max_pages])

    combined = combine_texts(extracted_pages)
    if not combined:
        sr.scrape_status = "error"
        sr.error_reason = "no_text_extracted"
    else:
        sr.scrape_status = "done"

    full_text = combined
    preview = combined[:preview_chars]

    if len(full_text) > max_chars_full:
        full_text = full_text[:max_chars_full]
        sr.truncated_full = True
    if len(preview) >= preview_chars and len(combined) > preview_chars:
        sr.truncated_preview = True

    sr.duration_sec = round(time.time() - started, 3)
    sr.finished_at = datetime.utcnow().isoformat() + "Z"
    return sr, full_text, preview, page_results, main_text


def stable_row_id(value: str, fallback_index: int) -> str:
    base = (value or "").strip().lower()
    if not base:
        return f"row_{fallback_index}"
    h = hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"site_{h}"


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "row_id",
        "scrape_status",
        "error_reason",
        "normalized_url",
        "final_url",
        "pages_attempted",
        "pages_scraped",
        "methods_used",
        "http_statuses",
        "total_bytes",
        "total_text_len",
        "truncated_full",
        "truncated_preview",
        "duration_sec",
        "started_at",
        "finished_at",
        "visited_urls",
        "website_text_full",
        "website_text_preview",
        "website_text_main",
        "site_title",
        "site_meta_description",
        "site_headings",
        "main_text_len",
        "pages_planned",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c.endswith(("url", "status", "reason", "methods_used", "http_statuses", "visited_urls", "started_at", "finished_at")) else None
    return df


def load_existing_output(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        if path.lower().endswith(".xlsx"):
            return pd.read_excel(path)
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def save_outputs(df: pd.DataFrame, output_csv: str, write_xlsx: bool) -> None:
    df.to_csv(output_csv, index=False)
    if write_xlsx:
        xlsx_path = os.path.splitext(output_csv)[0] + ".xlsx"
        # Excel cell limit ~32,767 chars; full text may exceed. We'll still write, but Excel will truncate display.
        df.to_excel(xlsx_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="Input file with a 'website' column (.xlsx or .csv).",
    )
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--checkpoint", default="scraper_checkpoint.json", help="Checkpoint file (JSON).")
    ap.add_argument("--log", default="scraper.log", help="Log file path.")
    ap.add_argument("--max-pages", type=int, default=12, help="Max pages to crawl per website (default 12).")
    ap.add_argument("--use-sitemap", action="store_true", help="Use sitemap hints to add a few pages (optional).")
    ap.add_argument("--timeout", type=float, default=12.0, help="Requests timeout seconds (default 12).")
    ap.add_argument("--playwright-timeout-ms", type=int, default=25000, help="Playwright timeout in ms (default 25000).")
    ap.add_argument("--headless", action="store_true", default=True, help="Run browser headless (default True).")
    ap.add_argument("--no-headless", action="store_false", dest="headless", help="Run browser with UI.")
    ap.add_argument("--min-text", type=int, default=900, help="If extracted text is smaller, use Playwright fallback.")
    ap.add_argument("--max-chars-full", type=int, default=250000, help="Max chars stored in full text column.")
    ap.add_argument("--preview-chars", type=int, default=8000, help="Chars stored in preview column.")
    ap.add_argument("--per-page-delay", type=float, default=0.6, help="Delay between page fetches (seconds).")
    ap.add_argument("--save-every", type=int, default=10, help="Save output every N processed rows.")
    ap.add_argument("--write-xlsx", action="store_true", help="Also write an XLSX copy next to the CSV.")
    args = ap.parse_args()

    logger = setup_logger(args.log)
    logger.info("Starting scraper_magical")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    # Support XLSX and CSV (Actions inputs may come from Google Drive exports)
    in_lower = args.input.lower()
    if in_lower.endswith((".xlsx", ".xls")):
        df_in = pd.read_excel(args.input)
    elif in_lower.endswith(".csv"):
        df_in = pd.read_csv(args.input, low_memory=False)
    else:
        # Best effort: try Excel first, then CSV
        try:
            df_in = pd.read_excel(args.input)
        except Exception:
            df_in = pd.read_csv(args.input, low_memory=False)
    if "website" not in df_in.columns:
        raise ValueError("Input file must contain a 'website' column.")

    # Attach stable row_id
    if "row_id" not in df_in.columns:
        df_in.insert(0, "row_id", [stable_row_id(str(u) if pd.notna(u) else "", i) for i, u in enumerate(df_in["website"].tolist())])

    # Load existing output to resume
    df_out_existing = load_existing_output(args.output)
    if df_out_existing is not None and "row_id" in df_out_existing.columns:
        # merge: keep latest scraped columns from existing output
        df = df_in.merge(df_out_existing.drop_duplicates("row_id"), on="row_id", how="left", suffixes=("", "_old"))
        # prefer existing scraped cols if present
        for col in df_out_existing.columns:
            if col in df.columns and col not in df_in.columns:
                pass
        # If merge created duplicates, clean them (simple strategy)
        for col in list(df.columns):
            if col.endswith("_old"):
                base = col[:-4]
                if base in df.columns:
                    # prefer non-null base; fill with _old
                    df[base] = df[base].where(df[base].notna(), df[col])
                    df.drop(columns=[col], inplace=True)
    else:
        df = df_in.copy()

    df = ensure_columns(df)

    # Determine pending rows
    pending_mask = df["scrape_status"].fillna("") != "done"
    pending_indices = df.index[pending_mask].tolist()
    logger.info("Total rows: %d | Pending: %d", len(df), len(pending_indices))

    session = build_session()

    # load checkpoint (optional)
    done_row_ids: Set[str] = set()
    if os.path.exists(args.checkpoint):
        try:
            with open(args.checkpoint, "r", encoding="utf-8") as f:
                ck = json.load(f)
            done_row_ids = set(ck.get("done_row_ids", []))
        except Exception:
            done_row_ids = set()

    processed = 0
    total_rows = len(df)
    errors = 0
    skipped = 0
    done = 0
    emit_progress(f"START file={os.path.basename(args.input)} total_rows={total_rows}")
    last_save = time.time()

    import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Concurrency controls
site_workers = max(1, int(getattr(args, "site_workers", 16) or 16))
playwright_workers = max(1, int(getattr(args, "playwright_workers", 2) or 2))
site_time_budget_sec = int(getattr(args, "site_time_budget_sec", 45) or 45)

pw_semaphore: "threading.BoundedSemaphore" = threading.BoundedSemaphore(playwright_workers)

def _scrape_one(idx: int) -> Tuple[int, SiteResult, str, str, str, List[PageResult]]:
    # Each thread uses its own session (safe and faster with keep-alives).
    session = build_session()
    url = df.at[idx, "website"]
    sr, full_text, preview, page_results, main_text = scrape_site(
        session=session,
        input_url=url,
        max_pages=int(args.max_pages),
        timeout=float(args.timeout),
        playwright_timeout_ms=int(args.playwright_timeout_ms),
        headless=bool(args.headless),
        min_text_for_requests_ok=int(args.min_text),
        use_sitemap=bool(args.use_sitemap),
        max_chars_full=int(args.max_chars_full),
        preview_chars=int(args.preview_chars),
        per_page_delay=float(args.per_page_delay),
        logger=logger,
        playwright_semaphore=pw_semaphore,
        site_time_budget_sec=site_time_budget_sec,
    )
    return idx, sr, full_text, preview, main_text, page_results

# Submit pending rows
futures = []
with ThreadPoolExecutor(max_workers=site_workers) as ex:
    for idx in pending_indices:
        row_id = str(df.at[idx, "row_id"])
        if row_id in done_row_ids and str(df.at[idx, "scrape_status"]) == "done":
            continue
        futures.append(ex.submit(_scrape_one, idx))

    # Consume results as they complete (real-time progress)
    for fut in as_completed(futures):
        try:
            idx, sr, full_text, preview, main_text, page_results = fut.result()
            row_id = str(df.at[idx, "row_id"])

            df.at[idx, "scrape_status"] = sr.scrape_status
            df.at[idx, "error_reason"] = sr.error_reason
            df.at[idx, "normalized_url"] = sr.normalized_url
            df.at[idx, "final_url"] = sr.final_url
            df.at[idx, "pages_attempted"] = sr.pages_attempted
            df.at[idx, "pages_scraped"] = sr.pages_scraped
            df.at[idx, "pages_planned"] = max(sr.pages_attempted, sr.pages_scraped)
            df.at[idx, "methods_used"] = sr.methods_used
            df.at[idx, "http_statuses"] = sr.http_statuses
            df.at[idx, "total_bytes"] = sr.total_bytes
            df.at[idx, "total_text_len"] = sr.total_text_len
            df.at[idx, "truncated_full"] = sr.truncated_full
            df.at[idx, "truncated_preview"] = sr.truncated_preview
            df.at[idx, "duration_sec"] = sr.duration_sec
            df.at[idx, "started_at"] = sr.started_at
            df.at[idx, "finished_at"] = sr.finished_at
            df.at[idx, "visited_urls"] = sr.visited_urls

            # Content columns (upgrade)
            df.at[idx, "website_text_full"] = full_text
            df.at[idx, "website_text_preview"] = preview
            df.at[idx, "website_text_main"] = main_text
            df.at[idx, "site_title"] = sr.site_title
            df.at[idx, "site_meta_description"] = sr.site_meta_description
            df.at[idx, "site_headings"] = sr.site_headings
            df.at[idx, "main_text_len"] = sr.main_text_len

            if sr.scrape_status == "done":
                done_row_ids.add(row_id)

        except Exception as e:
            # This is a thread-level failure, not a scrape error captured by sr.
            # Mark as error and continue.
            df.at[idx, "scrape_status"] = "error"
            df.at[idx, "error_reason"] = f"thread_exception: {e}"

        # Update counters from current row status
        processed += 1
        st = str(df.at[idx, "scrape_status"])
        if st == "done":
            done += 1
        elif st == "error":
            errors += 1
        elif st == "skipped":
            skipped += 1

        pending = max(0, total_rows - (done + errors + skipped))

        # Emit progress often (kinda realtime)
        if processed % max(1, args.save_every) == 0 or processed == total_rows:
            emit_progress(
                f"PROGRESS file={os.path.basename(args.input)} processed={processed}/{total_rows} "
                f"done={done} error={errors} skipped={skipped} pending={pending} "
                f"site_workers={site_workers} pw_workers={playwright_workers}"
            )

        # Save periodically (atomic-ish)
        if processed % max(1, args.save_every) == 0 or (time.time() - last_save) > 60:
            save_outputs(df, args.output, args.write_xlsx)
            with open(args.checkpoint, "w", encoding="utf-8") as f:
                json.dump({"done_row_ids": sorted(done_row_ids), "updated_at": utc_now()}, f)
            logger.info("Saved progress to %s (processed=%d)", args.output, processed)
            last_save = time.time()
    emit_progress(
        f"FINISH file={os.path.basename(args.input)} processed={processed}/{total_rows} "
        f"done={done} error={errors} skipped={skipped} pending={max(0, total_rows-(done+errors+skipped))}"
    )

    # final save
    save_outputs(df, args.output, args.write_xlsx)
    with open(args.checkpoint, "w", encoding="utf-8") as f:
        json.dump({"done_row_ids": sorted(done_row_ids), "updated_at": utc_now()}, f)

    logger.info("Complete. Output: %s", args.output)


if __name__ == "__main__":

    main()

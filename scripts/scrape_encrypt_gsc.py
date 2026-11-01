from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


def _encrypt(data: bytes, key: str) -> bytes:
    from cryptography.fernet import Fernet
    return Fernet(key.encode()).encrypt(data)


_BASE_URL        = "https://www.goldsilvercentral.com.sg"
_GOLD_BASE_URL   = f"{_BASE_URL}/gold/"
_SILVER_BASE_URL = f"{_BASE_URL}/silver-coins/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-SG,en-GB;q=0.9,en;q=0.8",
    "Referer": "https://www.goldsilvercentral.com.sg/",
    "Upgrade-Insecure-Requests": "1",
}

GRAMS_PER_TROY_OUNCE = 31.1034768
GRAMS_PER_KG = 1000.0

_WEIGHT_RE = re.compile(
    r"(\d+(?:[._]\d+)?(?:/\d+)?)\s*(gram|g|kg|troy\s*oz|oz)\b", re.IGNORECASE
)
_MULTI_WEIGHT_RE = re.compile(
    r"(\d+)\s*[xX×]\s*(\d+(?:[._]\d+)?(?:/\d+)?)\s*(gram|g|kg|troy\s*oz|oz)\b",
    re.IGNORECASE,
)


def _parse_single_weight(raw: str, unit: str) -> float:
    unit = unit.lower()
    if "/" in raw:
        num, den = raw.split("/", 1)
        val = float(num) / float(den)
    elif "_" in raw:
        num, den = raw.split("_", 1)
        val = float(num) / float(den)
    else:
        val = float(raw)
    if "kg" in unit:
        return val * GRAMS_PER_KG
    if "oz" in unit:
        return val * GRAMS_PER_TROY_OUNCE
    return val


def _parse_weight_grams(name: str) -> Optional[float]:
    m = _MULTI_WEIGHT_RE.search(name)
    if m:
        return int(m.group(1)) * _parse_single_weight(m.group(2), m.group(3))
    for m in _WEIGHT_RE.finditer(name):
        return _parse_single_weight(m.group(1), m.group(2))
    return None


def _parse_sgd(text: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9.]", "", text.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


@dataclass
class GscProduct:
    product_id: str
    name: str
    metal: str
    weight_grams: Optional[float]
    buy_price: float
    buyback_price: Optional[float]
    url: str


_LI_PRODUCT_RE = re.compile(r'<li\s[^>]*\bpost-(\d+)\b[^>]*\bproduct\b[^>]*>')
_OOS_RE        = re.compile(r'\boutofstock\b')
_PID_RE        = re.compile(
    r'data-product_id="(\d+)"[^>]*data-product_price="([0-9.]+)"[^>]*data-fb-name="([^"]+)"'
)
_BB_RE         = re.compile(r"BuyBack Price: SGD ([0-9.,]+)")
_WC_COUNT_RE   = re.compile(
    r'woocommerce-result-count[^>]*>[^<]*?(\d+)[^<]*?(\d+)[^<]*?of\s+(\d+)\s+results?',
    re.IGNORECASE | re.DOTALL,
)


def _parse_page(html: str, metal: str) -> List[GscProduct]:
    li_matches = list(_LI_PRODUCT_RE.finditer(html))
    products, seen = [], set()
    for i, li_m in enumerate(li_matches):
        block_start = li_m.start()
        block_end = li_matches[i+1].start() if i+1 < len(li_matches) else len(html)
        block = html[block_start:block_end]
        li_end = block.find(">")
        if li_end != -1 and _OOS_RE.search(block[:li_end+1]):
            continue
        pid_m = _PID_RE.search(block)
        if not pid_m:
            continue
        pid, price_str, name = pid_m.group(1), pid_m.group(2), pid_m.group(3)
        if pid in seen:
            continue
        seen.add(pid)
        sell_price = _parse_sgd(price_str)
        if sell_price is None:
            continue
        bb_m = _BB_RE.search(block)
        products.append(GscProduct(
            product_id=pid, name=name, metal=metal,
            weight_grams=_parse_weight_grams(name),
            buy_price=sell_price,
            buyback_price=_parse_sgd(bb_m.group(1)) if bb_m else None,
            url=f"{_BASE_URL}/?p={pid}",
        ))
    return products


def _discover_page_count(html: str) -> Optional[int]:
    import math
    m = _WC_COUNT_RE.search(html)
    if m:
        per_page = int(m.group(2))
        total = int(m.group(3))
        if per_page > 0:
            return math.ceil(total / per_page)
    return None


def _fetch_page_html(scraper, url: str, timeout_s: float) -> Optional[str]:
    try:
        resp = scraper.get(url, timeout=timeout_s)
        return resp.text if resp.status_code == 200 else None
    except Exception:
        return None


def _curl_cffi_get(url: str, timeout_s: float) -> Optional[str]:
    try:
        from curl_cffi.requests import Session as CurlSession
    except ImportError:
        return None
    for target in ("chrome124", "chrome110", "chrome99", "safari17_0"):
        try:
            with CurlSession(impersonate=target) as s:
                r = s.get(url, timeout=timeout_s)
                if r.status_code == 200 and "Just a moment" not in r.text:
                    return r.text
        except Exception:
            pass
    return None


def _scraperapi_get(url: str, timeout_s: float):
    key = os.environ.get("SCRAPERAPI_KEY", "").strip()
    if not key:
        return None, {}
    import requests as _req
    params = {"api_key": key, "url": url, "render": "true"}
    try:
        r = _req.get("https://api.scraperapi.com/", params=params, timeout=timeout_s + 30)
        if r.status_code == 200 and "Just a moment" not in r.text:
            return r.text, dict(r.cookies)
    except Exception:
        pass
    return None, {}


def _playwright_bootstrap(url: str, timeout_s: float):
    import requests as _req
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=_HEADERS["User-Agent"], locale="en-SG")
        pg = ctx.new_page()
        pg.goto(url, wait_until="domcontentloaded", timeout=int(timeout_s * 1000))
        try:
            pg.wait_for_function("() => !document.title.includes('Just a moment')", timeout=30000)
        except Exception:
            pass
        html = pg.content()
        pw_cookies = ctx.cookies()
        browser.close()
    s = _req.Session()
    s.headers.update(_HEADERS)
    for c in pw_cookies:
        s.cookies.set(c["name"], c["value"], domain=c.get("domain", ""), path=c.get("path", "/"))
    return html, s


def _fetch_all_pages(base_url: str, metal: str, timeout_s: float, scraper, page1_html: str) -> List[GscProduct]:
    products = _parse_page(page1_html, metal)
    if not products:
        return []
    total_pages = _discover_page_count(page1_html)
    if total_pages is not None and total_pages <= 1:
        return products
    pages = range(2, (total_pages or 20) + 1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_fetch_page_html, scraper, f"{base_url}page/{p}/", timeout_s) for p in pages]
        for fut in futs:
            html = fut.result()
            if html:
                products.extend(_parse_page(html, metal))
    return products


def fetch_products(timeout_s: float = 60.0) -> List[GscProduct]:
    gold_p1_html = None
    scraper = None

    gold_p1_html, sa_cookies = _scraperapi_get(_GOLD_BASE_URL, timeout_s)
    if gold_p1_html:
        import requests as _req
        scraper = _req.Session()
        scraper.headers.update(_HEADERS)
        scraper.cookies.update(sa_cookies)

    if not gold_p1_html or "Just a moment" in gold_p1_html:
        gold_p1_html = _curl_cffi_get(_GOLD_BASE_URL, timeout_s)
        if gold_p1_html:
            from curl_cffi.requests import Session as _CS
            scraper = _CS(impersonate="chrome124")

    if not gold_p1_html or "Just a moment" in gold_p1_html:
        gold_p1_html, scraper = _playwright_bootstrap(_GOLD_BASE_URL, timeout_s)

    if not gold_p1_html or "Just a moment" in gold_p1_html:
        raise RuntimeError("blocked")

    silver_p1_html = _fetch_page_html(scraper, _SILVER_BASE_URL, timeout_s) or ""

    with ThreadPoolExecutor(max_workers=2) as pool:
        gf = pool.submit(_fetch_all_pages, _GOLD_BASE_URL, "gold", timeout_s, scraper, gold_p1_html)
        sf = pool.submit(_fetch_all_pages, _SILVER_BASE_URL, "silver", timeout_s, scraper, silver_p1_html)
        products = gf.result() + sf.result()

    return products


def main() -> None:
    key = os.environ.get("GSC_FERNET_KEY", "").strip()
    if not key:
        sys.exit(1)

    out_path = Path(os.environ.get("OUT_PATH", "data/gsc.enc"))
    products = fetch_products(timeout_s=120.0)
    if not products:
        sys.exit(1)

    payload = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "products": [asdict(p) for p in products],
    }
    encrypted = _encrypt(json.dumps(payload).encode("utf-8"), key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(encrypted)


if __name__ == "__main__":
    main()

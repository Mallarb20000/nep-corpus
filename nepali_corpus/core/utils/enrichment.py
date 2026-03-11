from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib3
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote

from nepali_corpus.core.services.scrapers.pdf.utils import HAS_PYMUPDF, _extract_text_from_pdf
from nepali_corpus.core.utils.content_types import identify_content_type

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

def _cache_path(cache_dir: str, url: str, ext: str = ".html") -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{h}{ext}")


def fetch_content(url: str, cache_dir: str, timeout: int = 30, delay: float = 0.5) -> Tuple[Optional[bytes], str]:
    """Fetches url content and returns (bytes, content_type). Downloads PDFs and HTML."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check cache first for html or pdf
    html_path = _cache_path(cache_dir, url, ".html")
    pdf_path = _cache_path(cache_dir, url, ".pdf")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            return f.read(), "text/html"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            return f.read(), "application/pdf"

    time.sleep(delay)
    # Unquote URL to avoid double-encoding by requests
    fetch_url = unquote(url)
    
    try:
        r = requests.get(
            fetch_url, 
            timeout=timeout, 
            headers={"User-Agent": "NepaliCorpusBot/1.0 (+https://himalaya.ai)"}, 
            stream=True, 
            verify=False
        )
        if r.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {r.status_code}")
            return None, ""
        
        content_type = r.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            c_type = "application/pdf"
            path = pdf_path
            # limit pdf to 50MB
            # we just stream and read up to 50MB
            data = r.raw.read(50 * 1024 * 1024)
        else:
            c_type = "text/html"
            path = html_path
            data = r.content
            
        with open(path, "wb") as f:
            f.write(data)
        return data, c_type
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None, ""


def extract_text(data: bytes, content_type: str, url: Optional[str] = None, use_trafilatura: bool = True) -> str:
    if not data:
        return ""
        
    # Refine content type if needed using URL and sniff
    refined_type = identify_content_type(url or "", data)
    
    # If the provided content_type header is robust (like pdf), keep it, otherwise use refined
    if "pdf" in content_type.lower():
        ctype = "pdf"
    elif "json" in content_type.lower():
        ctype = "json"
    elif "xml" in content_type.lower():
        ctype = "xml"
    elif "csv" in content_type.lower():
        ctype = "csv"
    else:
        ctype = refined_type

    if ctype == "pdf":
        if not HAS_PYMUPDF:
            logger.warning(f"Skipping PDF {url} because PyMuPDF is not installed")
            return ""
        try:
            return _extract_text_from_pdf(data).strip()
        except Exception as e:
            logger.warning(f"Failed to extract PDF {url}: {e}")
            return ""

    if ctype == "json":
        try:
            import json
            obj = json.loads(data)
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            pass
            
    if ctype == "csv":
        try:
            import csv
            import io
            text_data = data.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(text_data))
            rows = [" ".join(row) for row in reader]
            return "\n".join(rows)
        except Exception:
            pass

    if ctype == "xml":
        try:
            soup = BeautifulSoup(data, "xml")
            return soup.get_text("\n")
        except Exception:
            pass

    # Default: Treat as HTML
    try:
        html = data.decode("utf-8")
    except UnicodeDecodeError:
        html = data.decode("utf-8", errors="ignore")

    extracted_text = ""
    if use_trafilatura:
        try:
            import trafilatura
            # silence trafilatura logs
            logging.getLogger("trafilatura").setLevel(logging.ERROR)

            extracted_text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        except Exception:
            pass

    if not extracted_text:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        
        # Remove common boilerplate tags
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "button"]):
            tag.extract()

        # Specifically remove elements that are often sidebars, ads, or navigation in Nepal govt sites
        selectors = [
            "#sidebar", ".sidebar", "#footer", ".footer", ".nav", ".menu", ".ads", 
            ".top-bar", ".top-header", ".language-switcher", ".social-links", 
            ".breadcrumb", ".breadcrumb-wrapper", ".search-form", ".print-btn",
            ".share-links", ".related-posts", "#comments", ".pagination"
        ]
        for selection in soup.select(", ".join(selectors)):
            selection.extract()
            
        # Try reaching for common content containers
        content_node = None
        # Prioritize more specific content containers found in parliament and govt sites
        content_selectors = [
            "article", "main", ".content-inner", ".post-content", 
            ".article-content", "#content", ".entry-content", 
            ".news-detail", ".notice-detail"
        ]
        for selector in content_selectors:
            found = soup.select_one(selector)
            if found and len(found.get_text()) > 200:
                content_node = found
                break
                
        if content_node:
            extracted_text = content_node.get_text("\n")
        else:
            extracted_text = soup.get_text("\n")
            
    if not extracted_text:
        return ""

    # Post-processing: Strip common navigation patterns that leak through
    lines = [ln.strip() for ln in extracted_text.split('\n')]
    lines = [ln for ln in lines if ln]
    
    top_menu_keywords = {
        "check mail", "english", "nepali", "toggle navigation", "home", 
        "about caan", "organization structure", "caan board", "management", 
        "about us", "safety policy statement", "skip to content", "search for:"
    }
    
    # Strip from the top of the document
    while lines:
        ln_lower = lines[0].lower()
        is_menu = False
        for p in top_menu_keywords:
            if ln_lower.startswith(p) or ln_lower == p:
                is_menu = True
                break
                
        if not is_menu and len(lines[0].split()) <= 2 and ln_lower in [
            "search", "menu", "close", "login", "register", "contact us", "faq", "language", "ne", "en"
        ]:
            is_menu = True
            
        if is_menu:
            lines.pop(0)
        else:
            break
            
    text = "\n".join(lines).strip()
    
    bad_patterns = [
        "Language : नेपाली ENGLISH",
        "Language : English नेपाली",
        "Skip to content",
        "Toggle navigation",
        "Search for:",
        "Print PDF",
        "Share this:",
    ]
    for p in bad_patterns:
        if text.startswith(p):
            text = text[len(p):].strip()
        text = text.replace(p, " ")
        
    import re
    text = re.sub(r' +', ' ', text)
    return text.strip()

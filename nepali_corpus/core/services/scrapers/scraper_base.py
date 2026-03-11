from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class RetryableHTTPError(Exception):
    """Raised for HTTP status codes that should be retried (429, 5xx)."""

    def __init__(self, status_code: int, url: str = ""):
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} for {url}")


class ScraperBase:
    """Shared base for HTTP scrapers (session, fetch helpers)."""

    def __init__(self, base_url: str, delay: float = 0.5, verify_ssl: bool = False) -> None:
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.delay = delay
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "NepaliCorpusBot/1.0 (+https://himalaya.ai)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,ne;q=0.8",
            }
        )
        self.session.verify = verify_ssl

    @retry(
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, RetryableHTTPError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def fetch_page(self, url: str, timeout: int = 30) -> Optional[BeautifulSoup]:
        if not url:
            return None
        if self.delay:
            time.sleep(self.delay)
        try:
            response = self.session.get(url, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            raise  # let tenacity retry these
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

        # Retryable HTTP errors
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableHTTPError(response.status_code, url)
        # Non-retryable errors — return None
        if response.status_code == 404:
            logger.debug("404 Not Found: %s", url)
            return None
        if response.status_code >= 400:
            logger.warning("HTTP %s for %s", response.status_code, url)
            return None

        return BeautifulSoup(response.text, "html.parser")

    def base_domain(self) -> str:
        if not self.base_url:
            return ""
        return urlparse(self.base_url).netloc.lower().lstrip("www.")


__all__ = ["ScraperBase", "RetryableHTTPError"]

from typing import Optional
import os
from urllib.parse import urlparse

def identify_content_type(url: str, content: Optional[bytes] = None) -> str:
    """
    Identifies the content type of a document based on its URL and optionally its content.
    Returns strings like 'html', 'pdf', 'json', 'csv', 'xlsx', 'xml', 'social'.
    """
    if not url:
        return "html"

    # Social media identification
    social_domains = ["twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com", "nitter.poast.org", "nitter.cz", "nitter.net"]
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    for sd in social_domains:
        if sd in domain:
            return "social"

    # 1. Check URL extension
    path = parsed_url.path.lower()
    ext = os.path.splitext(path)[1].lstrip('.')
    
    mapping = {
        'pdf': 'pdf',
        'json': 'json',
        'csv': 'csv',
        'xlsx': 'xlsx',
        'xls': 'xlsx',
        'xml': 'xml',
        'html': 'html',
        'htm': 'html',
        'php': 'html',
        'aspx': 'html',
        'jsp': 'html',
    }
    
    if ext in mapping:
        return mapping[ext]
    
    # 2. Check content signature (sniffing) if available
    if content:
        # Remove whitespace at start for sniffing
        start = content[:100].strip()
        if start.startswith(b'%PDF'):
            return 'pdf'
        if start.startswith(b'{') or start.startswith(b'['):
            return 'json'
        if start.startswith(b'<?xml'):
            return 'xml'
        if b'<html' in start.lower() or b'<!doctype html' in start.lower():
            return 'html'
            
    # Default fallback for web URLs
    return 'html'

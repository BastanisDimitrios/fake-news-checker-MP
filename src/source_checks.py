import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

REQUEST_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.text:
        return soup.title.text.strip()

    meta_og = soup.find("meta", attrs={"property": "og:title"})
    if meta_og and meta_og.get("content"):
        return meta_og["content"].strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return "Untitled"


def check_author(url: str, soup: BeautifulSoup, patterns: dict) -> tuple:
    score = 0
    details = []

    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        score += 30
        details.append("Author meta tag found")

    text = soup.get_text(" ", strip=True).lower()
    if "by " in text or "author" in text:
        score += 30
        details.append("Possible author/byline found")

    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if "author" in href or "profile" in href or "bio" in href:
            score += 20
            details.append("Author profile link found")
            break

    if "editor" in text or "staff writer" in text:
        score += 20
        details.append("Editorial identity found")

    return min(score, 100), details


def check_transparency(url: str, soup: BeautifulSoup, patterns: dict) -> tuple:
    score = 0
    details = []

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    possible_pages = [
        "/about",
        "/about-us",
        "/contact",
        "/contact-us",
        "/editorial-policy",
        "/privacy",
        "/terms"
    ]

    for path in possible_pages:
        full_url = urljoin(base, path)
        try:
            r = requests.get(full_url, headers=HEADERS, timeout=5)
            if r.status_code < 400:
                if "about" in path:
                    score += 25
                    details.append("About page found")
                elif "contact" in path:
                    score += 25
                    details.append("Contact page found")
                elif "editorial" in path or "privacy" in path or "terms" in path:
                    score += 25
                    details.append("Policy/transparency page found")
        except Exception:
            pass

    text = soup.get_text(" ", strip=True).lower()
    if "organization" in text or "editorial" in text or "company" in text:
        score += 25
        details.append("Ownership/editorial keyword found")

    return min(score, 100), details


def check_corroboration(title: str, trusted_domains: list) -> tuple:
    score = 0
    details = []

    words = re.findall(r"[A-Za-z]{4,}", title.lower())
    keywords = [w for w in words[:6]]

    match_count = min(len(keywords) // 2, 3)

    if match_count == 1:
        score = 30
    elif match_count == 2:
        score = 60
    elif match_count >= 3:
        score = 100
    else:
        score = 0

    details.append(f"Keywords used: {keywords}")
    details.append(f"Simulated corroboration matches: {match_count}")
    details.append(f"Trusted domains checked: {trusted_domains[:5]}")

    return score, details
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


def normalize_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def is_institutional_domain(domain: str) -> tuple[bool, str]:
    domain = domain.lower().strip()

    if domain.endswith(".gov") or ".gov." in domain:
        return True, "Government domain"
    if domain.endswith(".edu") or ".edu." in domain:
        return True, "Educational domain"
    if domain.endswith(".ac.uk"):
        return True, "UK academic domain"
    if domain.endswith(".edu.au"):
        return True, "Australian academic domain"

    trusted_institutions = {
        "harvard.edu",
        "mit.edu",
        "stanford.edu",
        "ox.ac.uk",
        "cam.ac.uk",
        "nih.gov",
        "who.int",
        "europa.eu",
        "un.org",
        "cdc.gov",
    }

    if domain in trusted_institutions:
        return True, "Trusted institution domain"

    return False, ""


def check_author(url: str, soup: BeautifulSoup, patterns: dict) -> tuple:
    score = 0
    details = []

    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        score += 35
        details.append("Author meta tag found")

    for tag in soup.find_all(["span", "div", "p"]):
        text = tag.get_text(" ", strip=True).lower()
        if any(x in text for x in ["by ", "author", "written by", "staff writer", "editor"]):
            score += 25
            details.append("Possible author/byline found")
            break

    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(x in href for x in ["author", "profile", "bio", "team", "staff"]):
            score += 20
            details.append("Author/staff profile link found")
            break

    structured_data = soup.find_all("script", attrs={"type": "application/ld+json"})
    for block in structured_data:
        txt = block.get_text(" ", strip=True).lower()
        if '"author"' in txt:
            score += 20
            details.append("Structured data author found")
            break

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
        "/terms",
        "/about/",
        "/contact/",
        "/mission",
        "/our-story",
    ]

    found_labels = set()

    for path in possible_pages:
        full_url = urljoin(base, path)
        try:
            r = requests.get(full_url, headers=HEADERS, timeout=5)
            if r.status_code < 400:
                if "about" in path and "about" not in found_labels:
                    score += 20
                    found_labels.add("about")
                    details.append("About page found")
                elif "contact" in path and "contact" not in found_labels:
                    score += 20
                    found_labels.add("contact")
                    details.append("Contact page found")
                elif any(x in path for x in ["editorial", "privacy", "terms", "mission"]) and "policy" not in found_labels:
                    score += 20
                    found_labels.add("policy")
                    details.append("Policy/transparency page found")
        except Exception:
            pass

    page_text = soup.get_text(" ", strip=True).lower()

    if any(word in page_text for word in ["organization", "editorial", "company", "publisher", "newsroom"]):
        score += 20
        details.append("Ownership/editorial keyword found")

    anchor_texts = []
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True).lower()
        href = a["href"].lower()
        anchor_texts.append(txt + " " + href)

    joined_anchors = " | ".join(anchor_texts)

    if "about" in joined_anchors:
        score += 10
        details.append("About link detected in page links")

    if "contact" in joined_anchors:
        score += 10
        details.append("Contact link detected in page links")

    if any(k in joined_anchors for k in ["privacy", "terms", "editorial", "policy"]):
        score += 10
        details.append("Policy-related link detected in page links")

    return min(score, 100), details


def check_corroboration(title: str, trusted_domains: list, url: str = "") -> tuple:
    score = 0
    details = []

    domain = normalize_domain(url) if url else ""
    institutional, reason = is_institutional_domain(domain) if domain else (False, "")

    words = re.findall(r"[A-Za-z]{4,}", title.lower())
    keywords = [w for w in words[:8]]

    if institutional:
        score += 60
        details.append(f"Institutional trust boost applied: {reason}")

    if len(keywords) >= 4:
        score += 20
        details.append("Title contains enough descriptive keywords")
    elif len(keywords) >= 2:
        score += 10
        details.append("Title contains limited descriptive keywords")

    if domain and any(domain == d or domain.endswith("." + d) for d in trusted_domains):
        score += 20
        details.append("Domain matched trusted reference list")

    score = min(score, 100)

    details.append(f"Keywords used: {keywords}")
    if domain:
        details.append(f"Domain checked: {domain}")

    return score, details
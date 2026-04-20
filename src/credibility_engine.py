from source_checks import (
    fetch_html,
    make_soup,
    extract_title,
    check_author,
    check_transparency,
    check_corroboration,
    normalize_domain,
    is_institutional_domain,
)
from updater import load_reference_lists


def classify_source(domain: str) -> str:
    domain = domain.lower().strip()

    if domain.endswith(".gov") or ".gov." in domain:
        return "government"

    if (
        domain.endswith(".edu")
        or ".edu." in domain
        or domain.endswith(".ac.uk")
        or domain.endswith(".edu.au")
    ):
        return "university"

    news_keywords = [
        "news", "times", "post", "bbc", "cnn", "reuters",
        "guardian", "apnews", "foxnews", "politico"
    ]
    if any(k in domain for k in news_keywords):
        return "news"

    return "other"


def evaluate_source(url: str) -> dict:
    refs = load_reference_lists()

    html = fetch_html(url)
    soup = make_soup(html)
    title = extract_title(soup)

    domain = normalize_domain(url)
    institutional, institutional_reason = is_institutional_domain(domain)
    source_type = classify_source(domain)

    author_score, _ = check_author(url, soup, {})
    transparency_score, _ = check_transparency(url, soup, {})
    corroboration_score, _ = check_corroboration(title, refs["domains"], url)

    if source_type == "government":
        w_author, w_transparency, w_corroboration = 0.20, 0.50, 0.30
    elif source_type == "university":
        w_author, w_transparency, w_corroboration = 0.30, 0.30, 0.40
    elif source_type == "news":
        w_author, w_transparency, w_corroboration = 0.40, 0.30, 0.30
    else:
        w_author, w_transparency, w_corroboration = 0.35, 0.35, 0.30

    final_score = round(
        w_author * author_score +
        w_transparency * transparency_score +
        w_corroboration * corroboration_score,
        2
    )

    if institutional:
        final_score = max(final_score, 75)
        if transparency_score >= 60 and corroboration_score >= 60:
            final_score = max(final_score, 85)

    final_score = min(100, final_score)

    if final_score >= 70:
        label = "More Credible"
    elif final_score >= 40:
        label = "Uncertain"
    else:
        label = "Low Credibility"

    return {
        "title": title,
        "label": label,
        "final_score": final_score,
        "author_score": author_score,
        "transparency_score": transparency_score,
        "corroboration_score": corroboration_score,
        "domain": domain,
        "source_type": source_type,
        "institutional_detected": institutional,
        "institutional_reason": institutional_reason if institutional else "",
    }
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


def evaluate_source(url: str) -> dict:
    refs = load_reference_lists()

    html = fetch_html(url)
    soup = make_soup(html)
    title = extract_title(soup)

    domain = normalize_domain(url)
    institutional, institutional_reason = is_institutional_domain(domain)

    author_score, _ = check_author(url, soup, {})
    transparency_score, _ = check_transparency(url, soup, {})
    corroboration_score, _ = check_corroboration(title, refs["domains"], url)

    final_score = round(
        0.35 * author_score +
        0.35 * transparency_score +
        0.30 * corroboration_score,
        2
    )

    if institutional:
        final_score = max(final_score, 75)
        if transparency_score >= 60 or corroboration_score >= 60:
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
        "institutional_detected": institutional,
        "institutional_reason": institutional_reason if institutional else "",
    }
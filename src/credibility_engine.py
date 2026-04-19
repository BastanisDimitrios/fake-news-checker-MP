from source_checks import fetch_html, make_soup, extract_title, check_author, check_transparency, check_corroboration
from updater import load_reference_lists


def evaluate_source(url: str) -> dict:
    refs = load_reference_lists()

    html = fetch_html(url)
    soup = make_soup(html)
    title = extract_title(soup)

    author_score, _ = check_author(url, soup, {})
    transparency_score, _ = check_transparency(url, soup, {})
    corroboration_score, _ = check_corroboration(title, refs["domains"])

    final_score = round(
        0.30 * author_score +
        0.30 * transparency_score +
        0.40 * corroboration_score, 2
    )

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
    }
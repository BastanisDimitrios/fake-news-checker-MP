import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
REF_DIR = BASE_DIR / "data" / "reference"

TRUSTED_FILE = REF_DIR / "trusted_domains.json"
META_FILE = REF_DIR / "_meta.json"


DEFAULT_DOMAINS = {
    "domains": [
        "bbc.com",
        "reuters.com",
        "apnews.com",
        "cnn.com",
        "nytimes.com",
        "theguardian.com"
    ]
}


def update_reference_lists():
    REF_DIR.mkdir(parents=True, exist_ok=True)

    if not TRUSTED_FILE.exists():
        with open(TRUSTED_FILE, "w") as f:
            json.dump(DEFAULT_DOMAINS, f, indent=2)

    with open(META_FILE, "w") as f:
        json.dump({"last_updated": datetime.utcnow().isoformat()}, f)

    return "updated"


def load_reference_lists():
    if not TRUSTED_FILE.exists():
        update_reference_lists()

    with open(TRUSTED_FILE) as f:
        return json.load(f)
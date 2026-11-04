import re
from typing import List, Dict, Any, Optional

NEGATIVE_RESPONSES = {
    "none", "no", "not visible", "not found", "cannot determine", "unanswerable",
    "not clearly visible", "cannot be seen", "no target", "nothing", "n/a", "not present"
}

def is_negative_or_unanswerable(text: str) -> bool:
    """Returns True if the response is a refusal, empty, or negative detection."""
    clean = re.sub(r"[^\w\s]", "", str(text).lower()).strip()
    if not clean or clean in NEGATIVE_RESPONSES:
        return True
    return any(neg in clean for neg in ["not clearly visible", "cannot see", "not visible", "no evidence"])



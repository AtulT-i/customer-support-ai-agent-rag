import re


HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
SPACE_RE = re.compile(r"\s+")
PLACEHOLDER_RE = re.compile(r"<(?:EMAIL|URL|HANDLE|NUMBER)>", re.I)


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = URL_RE.sub("<URL>", text)
    text = HANDLE_RE.sub("<HANDLE>", text)
    text = LONG_NUMBER_RE.sub("<NUMBER>", text)
    return SPACE_RE.sub(" ", text).strip()


def sanitize_reply(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"^<HANDLE>[,!.\s-]*", "", text)
    text = re.sub(r"^(hi|hey|hello)(\s+there)?[,!.\s-]*", "", text, flags=re.I)
    text = re.sub(r"^<HANDLE>[,!.\s-]*", "", text)
    text = re.sub(r"\s*/[A-Z]{2}\s*(<URL>)?\s*$", "", text)
    return text[:500].strip()


def normalize_for_matching(value: object) -> str:
    """Use identical, identifier-free text for training, retrieval and deduplication."""
    text = PLACEHOLDER_RE.sub(" ", normalize_text(value))
    return SPACE_RE.sub(" ", text).strip().casefold()


def safe_guidance(value: object) -> str:
    """Conservative heuristic filter, not a guarantee of factual correctness."""
    text = sanitize_reply(value)
    # A historical account action or redacted link is not reusable instructions.
    if PLACEHOLDER_RE.search(text):
        return ""
    blocked = (
        r"\b(?:refund\w*|charg\w*|payment\w*|billing|password\w*|credential\w*|"
        r"hack\w*|secur\w*|private|dm|direct message|email|legal|privacy)\b",
        r"\b(?:we|i)(?:['’]ve| have| already| just|['’]ll| will| can)\b",
        r"\b(?:your|the)\s+(?:account|subscription|plan)\s+(?:is|has|was)\b",
        r"\b(?:resolved|refunded|cancelled|canceled|recorded|escalated|fixed|"
        r"investigating|confirmed|guarantee\w*)\b",
        r"\b(?:ignore|disregard)\b.*\b(?:instructions|rules|previous)\b",
        r"\bsystem prompt\b|\b(?:assistant|system|developer)\s*:",
    )
    if any(re.search(pattern, text, re.I) for pattern in blocked):
        return ""
    # Only reuse a complete, modest troubleshooting suggestion, not status claims.
    suggestions = re.findall(r"[^.!?]+[.!?]|[^.!?]+$", text)
    allowed = []
    for sentence in suggestions:
        sentence = sentence.strip()
        if re.match(
            r"^(?:please\s+)?(?:try\b|restart\b|reinstall\b|update\b|check\b|"
            r"could you\s+(?:try|restart|check|update)\b|"
            r"can you\s+(?:try|restart|check|update)\b)", sentence, re.I
        ):
            allowed.append(sentence)
    return " ".join(allowed)[:500]

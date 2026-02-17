import re

def _extract_reprompt(text: str) -> str:
    """
    Robust extraction of the prompt from the model's output.
    Prioritizes 'Reprompt:' marker, then XML tags, then fallback cleanup.
    """
    # 1. Look for explicit Reprompt marker (Best for our new system prompt)
    # Allows for "Reprompt:", "Re-prompt:", etc.
    patterns = [
        r"(?:^|\n)\s*Reprompt\s*:\s*(.*)",
        r"(?:^|\n)\s*Re-prompt\s*:\s*(.*)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            tail = m.group(1)
            # Stop at standard stop tokens or new sections
            stop = re.search(
                r"(?:\n\s*(?:User|Raw|Prompt|CoT|<think>|</think>))",
                tail,
                flags=re.IGNORECASE,
            )
            if stop:
                tail = tail[:stop.start()]
            return tail.strip()

    # 2. Look for XML tags (Paper standard)
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. Fallback: Remove <think> blocks and return the rest
    # This happens if the model forgets "Reprompt:" but outputs text after thinking.
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return cleaned

def replace_single_quotes(text):
    """Normalize quotes."""
    return text.replace("’", "”").replace("‘", "“")

# Regex for stricter banning (Illustration Mode)
_BANNED_PATTERNS = [
    r"\bphoto(graphy)?\b", r"\bphotorealistic\b", r"\brealism\b", r"\brealistic\b",
    r"\bhyper[-\s]?realistic\b", r"\bhyper[-\s]?realism\b", r"\bultra[-\s]?realistic\b",
    r"\bcinematic\b", r"\bdslr\b", r"\b35mm\b", r"\blens\b", r"\bdof\b", 
    r"\bdepth of field\b", r"\bbokeh\b", r"\brender(ing)?\b", r"\b8k\b", 
    r"\boverall\b", r"\bpresentation\b", r"\bmasterpiece\b", r"\bbest quality\b",
    r"写实|摄影|照片|实写"
]
_BANNED_RE = re.compile("|".join(_BANNED_PATTERNS), flags=re.IGNORECASE)

def strip_unwanted_photo_style(reprompt: str, original_user_text: str) -> str:
    """
    Safeguard: Removes photo-specific terms from the generated prompt 
    UNLESS the user explicitly asked for them in the original text.
    """
    # 1. Check if user WANTS photo style
    user_text_lower = original_user_text.lower()
    user_wants_photo = any(x in user_text_lower for x in ["photo", "realistic", "dslr", "写实", "照片"])
    
    if user_wants_photo:
        return reprompt

    # 2. Process the reprompt
    # Split by commas (standard tag separation)
    chunks = re.split(r"\s*(?:,|，)\s*", reprompt)
    kept = []
    
    for c in chunks:
        c_clean = c.strip()
        if not c_clean:
            continue
        # Filter out banned words
        if _BANNED_RE.search(c_clean):
            continue
        kept.append(c_clean)

    return ", ".join(kept)

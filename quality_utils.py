"""Shared quality checks for prefixed-verb data and Anki cards."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

BAD_GLOSS_EXACT = {
    "transitiv :",
    "intransitiv :",
    "reflexiv :",
    "transitiv:",
    "intransitiv:",
    "reflexiv:",
}
PLACEHOLDER_PATTERNS = (
    "dieser abschnitt fehlt",
    "hilf mit",
    "wiktionary zu vervollständigen",
    "bedeutungen fehlen",
    "noch keine bedeutung",
)
METADATA_GLOSS_PATTERNS = (
    "anmerkung: duden online",
    "duden online",
    "referenzen und weiterführende informationen",
)
GENERIC_CONSTRUCTION_PATTERNS = (
    "subjekt + verb + objekt",
    "subject + verb + object",
    "subject + verb stem",
    "subjekt -",
)
BLOCKING_QUALITY_FLAGS = {
    "missing_gloss_de",
    "missing_gloss_es",
    "missing_example_de",
    "missing_example_de_with_blank",
    "missing_answer",
    "missing_construction",
    "missing_present_3sg",
    "missing_perfect_auxiliary",
    "missing_participle_ii",
    "placeholder_gloss",
    "placeholder_example",
    "suspected_participle",
    "short_or_metadata_gloss",
    "metadata_gloss",
    "invalid_context_answer",
    "invalid_perfect_auxiliary",
    "invalid_separable_present_3sg",
    "invalid_participle_ii",
    "generic_construction",
    "openai_marked_unsuitable",
}
AUX_CANONICAL = {
    "haben": "haben",
    "hat": "haben",
    "sein": "sein",
    "ist": "sein",
}


def parse_optional_bool(value: Any) -> Optional[bool]:
    """Parse bool-like values while preserving unknown/null."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "ja"}:
        return True
    if lowered in {"false", "0", "no", "nein"}:
        return False
    return None


def normalize_perfect_auxiliary(value: Any) -> str:
    """Return canonical haben/sein where possible."""
    return AUX_CANONICAL.get(str(value or "").strip().lower(), str(value or "").strip().lower())


def word_tokens(text: str) -> List[str]:
    """Return lowercase word tokens, preserving German letters."""
    return re.findall(r"[A-Za-zÄÖÜäöüß]+", text.lower())


def is_quality_ok(flags: Sequence[str]) -> bool:
    """Return whether a sense is safe for default Anki generation."""
    return not (set(flags) & BLOCKING_QUALITY_FLAGS)


def is_valid_verbal_answer(
    *,
    answer: str,
    derived: str,
    base: str,
    prefix: str,
    participle_ii: str = "",
    present_3sg: str = "",
) -> bool:
    """Return True only when a cloze answer looks verbal for this verb."""
    answer_l = str(answer or "").strip().lower()
    if not answer_l:
        return False

    prefix_plain = str(prefix or "").rstrip("-").lower()
    allowed_exact = {
        str(derived or "").lower(),
        prefix_plain,
        str(participle_ii or "").lower(),
    }
    allowed_exact |= set(word_tokens(str(present_3sg or "")))
    allowed_exact.discard("")
    if answer_l in allowed_exact:
        return True

    if prefix_plain and answer_l.startswith(prefix_plain) and (answer_l.endswith("en") or "ge" in answer_l):
        return True
    if base and answer_l.endswith("t") and str(base).lower()[:4] in answer_l:
        return True
    return False


def is_useful_construction_data(
    *,
    construction: str,
    derived: str = "",
    base: str = "",
    prefix: str = "",
) -> bool:
    """Return whether a construction is specific enough for a card."""
    construction_l = str(construction or "").strip().lower()
    if not construction_l:
        return False
    if any(pattern in construction_l for pattern in GENERIC_CONSTRUCTION_PATTERNS):
        return False
    if construction_l.startswith("subject +") or "verb stem" in construction_l:
        return False

    prefix_plain = str(prefix or "").rstrip("-").lower()
    if str(derived or "").lower() and str(derived).lower() in construction_l:
        return True
    if prefix_plain and str(base or "").lower() and f"{prefix_plain}{str(base).lower()}" in construction_l:
        return True
    if any(token in construction_l for token in ("etwas", "jemand", "jemandem", "sich ", "akkusativ", "dativ")):
        return True
    return False


def detect_quality_flags(row: Dict[str, Any]) -> List[str]:
    """Infer data-quality flags from a normalized flat sense row."""
    flags: List[str] = []
    gloss_de = str(row.get("gloss_de") or row.get("gloss_de_simple") or "").strip()
    gloss_es = str(row.get("gloss_es") or "").strip()
    gloss_en = str(row.get("gloss_en") or "").strip()
    example_de = str(row.get("example_de") or row.get("example") or "").strip()
    example_de_with_blank = str(row.get("example_de_with_blank") or "").strip()
    answer = str(row.get("answer") or "").strip()
    construction = str(row.get("construction") or "").strip()
    present_3sg = str(row.get("present_3sg") or "").strip()
    perfect_auxiliary = normalize_perfect_auxiliary(row.get("perfect_auxiliary"))
    participle_ii = str(row.get("participle_ii") or "").strip()
    derived = str(row.get("derived") or "").strip()
    base = str(row.get("base") or "").strip()
    prefix = str(row.get("prefix") or "").strip()
    separability = str(row.get("separability") or "").strip().lower()
    register = str(row.get("register") or "").strip().lower()
    prefix_plain = prefix.rstrip("-").lower()

    if not gloss_de:
        flags.append("missing_gloss_de")
    if not gloss_es:
        flags.append("missing_gloss_es")
    if not gloss_en:
        flags.append("missing_gloss_en")
    if not example_de:
        flags.append("missing_example_de")
    if not example_de_with_blank or "___" not in example_de_with_blank:
        flags.append("missing_example_de_with_blank")
    if not answer:
        flags.append("missing_answer")
    if not construction:
        flags.append("missing_construction")
    if not present_3sg:
        flags.append("missing_present_3sg")
    if not perfect_auxiliary:
        flags.append("missing_perfect_auxiliary")
    elif perfect_auxiliary not in {"haben", "sein"}:
        flags.append("invalid_perfect_auxiliary")
    if not participle_ii:
        flags.append("missing_participle_ii")

    lowered_gloss = gloss_de.lower()
    lowered_example = example_de.lower()
    if lowered_gloss in BAD_GLOSS_EXACT:
        flags.append("metadata_gloss")
    if any(pattern in lowered_gloss for pattern in PLACEHOLDER_PATTERNS):
        flags.append("placeholder_gloss")
    if any(pattern in lowered_example for pattern in PLACEHOLDER_PATTERNS):
        flags.append("placeholder_example")
    if prefix.lower() == "ge-" or (derived.lower().startswith("ge") and bool(base)):
        flags.append("suspected_participle")
    if gloss_de and (len(gloss_de) < 8 or any(pattern in lowered_gloss for pattern in METADATA_GLOSS_PATTERNS)):
        flags.append("short_or_metadata_gloss")
    if example_de and len(example_de.split()) > 25:
        flags.append("long_example")
    if register in {"rare", "archaic", "domain-specific"}:
        flags.append("rare_or_domain_specific")
    if construction and not is_useful_construction_data(
        construction=construction,
        derived=derived,
        base=base,
        prefix=prefix,
    ):
        flags.append("generic_construction")
    if answer and not is_valid_verbal_answer(
        answer=answer,
        derived=derived,
        base=base,
        prefix=prefix,
        participle_ii=participle_ii,
        present_3sg=present_3sg,
    ):
        flags.append("invalid_context_answer")

    present_tokens = word_tokens(present_3sg)
    if separability == "separable" and present_3sg and prefix_plain:
        if present_3sg.lower().startswith(prefix_plain):
            flags.append("invalid_separable_present_3sg")
        if prefix_plain not in present_tokens:
            flags.append("invalid_separable_present_3sg")
    if separability == "inseparable" and present_3sg and prefix_plain and prefix_plain in present_tokens:
        flags.append("invalid_inseparable_present_3sg")

    participle_l = participle_ii.lower()
    if separability == "separable" and participle_l and prefix_plain and not participle_l.startswith(prefix_plain):
        flags.append("invalid_participle_ii")
    if separability == "inseparable" and participle_l and prefix_plain and participle_l.startswith(prefix_plain + "ge"):
        flags.append("invalid_participle_ii")

    return sorted(dict.fromkeys(flags))

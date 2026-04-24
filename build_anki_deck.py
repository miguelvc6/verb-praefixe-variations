"""Build Anki cards for German prefixed verbs from CSV or JSON data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

@dataclass
class AnkiVerbSense:
    base: str
    derived: str
    sense_id: int
    prefix: str
    separability: str
    gloss_de: str = ""
    gloss_es: str = ""
    gloss_en: str = ""
    example_de: str = ""
    example_es: str = ""
    example_en: str = ""
    example_de_with_blank: str = ""
    answer: str = ""
    construction: str = ""
    construction_es: str = ""
    register: str = "unknown"
    difficulty: str = "unknown"
    frequency_bucket: str = "unknown"
    is_reflexive: bool = False
    takes_accusative: Optional[bool] = None
    takes_dative: Optional[bool] = None
    fixed_preposition: str = ""
    present_3sg: str = ""
    perfect_auxiliary: str = ""
    participle_ii: str = ""
    separable_sentence_pattern: str = ""
    quality_flags: List[str] = field(default_factory=list)
    is_quality_ok: bool = True
    anki_hint_es: str = ""
    wiktionary_url: str = ""
    source: str = "wiktionary"


@dataclass
class AnkiCard:
    cardtype: str
    key: str
    front: str
    back: str
    tags: List[str]


def stable_int_id(text: str, modulo: int = 10**10) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def stable_guid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_input(path: Path) -> List[AnkiVerbSense]:
    """Load CSV or JSON verb data and normalize to AnkiVerbSense rows."""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            return [normalize_flat_row(row) for row in csv.DictReader(csv_file)]

    with path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)
    if not isinstance(payload, list):
        raise ValueError("JSON input must be a list")

    senses: List[AnkiVerbSense] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        nested = item.get("senses")
        if isinstance(nested, list):
            for sense in nested:
                if isinstance(sense, dict):
                    senses.append(normalize_nested_sense(item, sense))
        else:
            senses.append(normalize_flat_row(item))
    return senses


def normalize_nested_sense(verb: Dict[str, Any], sense: Dict[str, Any]) -> AnkiVerbSense:
    row = {
        **sense,
        "base": verb.get("base", ""),
        "derived": verb.get("derived", ""),
        "prefix": verb.get("prefix", ""),
        "separability": verb.get("separability", ""),
        "pos": verb.get("pos", ""),
        "wiktionary_url": verb.get("wiktionary_url", ""),
        "source": verb.get("source", "wiktionary"),
    }
    return normalize_flat_row(row)


def normalize_flat_row(row: Dict[str, Any]) -> AnkiVerbSense:
    """Normalize old and new flat rows into one dataclass."""
    quality_flags = parse_quality_flags(row.get("quality_flags", []))
    if not quality_flags:
        quality_flags = detect_quality_flags(row)
    quality_ok_raw = row.get("is_quality_ok")
    quality_ok = parse_bool_default(quality_ok_raw, is_quality_ok(quality_flags))

    return AnkiVerbSense(
        base=str(row.get("base", "")),
        derived=str(row.get("derived", "")),
        sense_id=parse_int(row.get("sense_id"), 1),
        prefix=str(row.get("prefix", "")),
        separability=str(row.get("separability", "")),
        gloss_de=str(row.get("gloss_de") or row.get("gloss_de_simple") or ""),
        gloss_es=str(row.get("gloss_es", "")),
        gloss_en=str(row.get("gloss_en", "")),
        example_de=str(row.get("example_de") or row.get("example") or ""),
        example_es=str(row.get("example_es", "")),
        example_en=str(row.get("example_en", "")),
        example_de_with_blank=str(row.get("example_de_with_blank", "")),
        answer=str(row.get("answer") or row.get("derived", "")),
        construction=str(row.get("construction", "")),
        construction_es=str(row.get("construction_es", "")),
        register=str(row.get("register") or "unknown"),
        difficulty=str(row.get("difficulty") or "unknown"),
        frequency_bucket=str(row.get("frequency_bucket") or "unknown"),
        is_reflexive=parse_bool_default(row.get("is_reflexive"), False),
        takes_accusative=parse_optional_bool(row.get("takes_accusative")),
        takes_dative=parse_optional_bool(row.get("takes_dative")),
        fixed_preposition=str(row.get("fixed_preposition", "")),
        present_3sg=str(row.get("present_3sg", "")),
        perfect_auxiliary=str(row.get("perfect_auxiliary", "")),
        participle_ii=str(row.get("participle_ii", "")),
        separable_sentence_pattern=str(row.get("separable_sentence_pattern", "")),
        quality_flags=quality_flags,
        is_quality_ok=quality_ok,
        anki_hint_es=str(row.get("anki_hint_es") or row.get("gloss_es") or ""),
        wiktionary_url=str(row.get("wiktionary_url", "")),
        source=str(row.get("source") or "wiktionary"),
    )


def parse_quality_flags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return []


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_bool_default(value: Any, default: bool) -> bool:
    parsed = parse_optional_bool(value)
    return default if parsed is None else parsed


def parse_optional_bool(value: Any) -> Optional[bool]:
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


def detect_quality_flags(row: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    gloss_de = str(row.get("gloss_de") or "").strip()
    gloss_es = str(row.get("gloss_es") or "").strip()
    gloss_en = str(row.get("gloss_en") or "").strip()
    example_de = str(row.get("example_de") or row.get("example") or "").strip()
    prefix = str(row.get("prefix") or "").strip().lower()
    derived = str(row.get("derived") or "").strip().lower()
    base = str(row.get("base") or "").strip().lower()
    register = str(row.get("register") or "").strip().lower()
    placeholder_patterns = ("dieser abschnitt fehlt", "hilf mit", "wiktionary zu vervollständigen")

    if not gloss_de:
        flags.append("missing_gloss_de")
    if not gloss_es:
        flags.append("missing_gloss_es")
    if not gloss_en:
        flags.append("missing_gloss_en")
    if not example_de:
        flags.append("missing_example_de")
    if any(pattern in gloss_de.lower() for pattern in placeholder_patterns):
        flags.append("placeholder_gloss")
    if any(pattern in example_de.lower() for pattern in placeholder_patterns):
        flags.append("placeholder_example")
    if prefix == "ge-" or (derived.startswith("ge") and bool(base)):
        flags.append("suspected_participle")
    if gloss_de and (len(gloss_de) < 8 or "duden online" in gloss_de.lower()):
        flags.append("short_or_metadata_gloss")
    if example_de and len(example_de.split()) > 25:
        flags.append("long_example")
    if register in {"rare", "archaic", "domain-specific"}:
        flags.append("rare_or_domain_specific")
    return sorted(dict.fromkeys(flags))


def is_quality_ok(flags: Sequence[str]) -> bool:
    blocking = {
        "missing_gloss_de",
        "missing_gloss_es",
        "missing_example_de",
        "placeholder_gloss",
        "placeholder_example",
        "suspected_participle",
        "short_or_metadata_gloss",
    }
    return not (set(flags) & blocking)


def make_context_blank(sense: AnkiVerbSense) -> Optional[Tuple[str, str]]:
    """Return a safe blanked German example and answer, or None."""
    if sense.example_de_with_blank and "___" in sense.example_de_with_blank:
        return sense.example_de_with_blank, sense.answer or sense.derived
    if not sense.example_de:
        return None
    for answer in blanking_candidates(sense):
        answer = (answer or "").strip()
        if not answer:
            continue
        pattern = re.compile(rf"\b{re.escape(answer)}\b", flags=re.IGNORECASE)
        if pattern.search(sense.example_de):
            return pattern.sub("___", sense.example_de, count=1), answer
    return None


def make_prefix_blank(sense: AnkiVerbSense) -> Optional[Tuple[str, str]]:
    """Return a conservative prefix cloze for exact derived or participle forms."""
    prefix_plain = sense.prefix.rstrip("-")
    if not prefix_plain or not sense.example_de:
        return None
    for answer in blanking_candidates(sense):
        answer = (answer or "").strip()
        if not answer.lower().startswith(prefix_plain.lower()):
            continue
        blanked_answer = "___" + answer[len(prefix_plain) :]
        pattern = re.compile(rf"\b{re.escape(answer)}\b", flags=re.IGNORECASE)
        if pattern.search(sense.example_de):
            return pattern.sub(blanked_answer, sense.example_de, count=1), answer
    return None


def blanking_candidates(sense: AnkiVerbSense) -> List[str]:
    """Return conservative verb forms that may safely be blanked."""
    candidates = [sense.derived, sense.participle_ii]
    prefix_plain = sense.prefix.rstrip("-")
    if sense.separability == "separable" and prefix_plain and sense.base:
        candidates.append(f"{prefix_plain}ge{sense.base}")
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def build_tags(sense: AnkiVerbSense, cardtype: str) -> List[str]:
    """Build normalized Anki tags."""
    tags = [
        "verb_prefix",
        f"family::{sense.base}",
        f"prefix::{sense.prefix.rstrip('-') or 'unknown'}",
        f"separability::{sense.separability or 'unknown'}",
        f"difficulty::{sense.difficulty or 'unknown'}",
        f"register::{sense.register or 'unknown'}",
        f"source::{(sense.source or 'wiktionary').replace('+', '_')}",
        f"quality::{'ok' if sense.is_quality_ok else 'low'}",
        f"cardtype::{cardtype}",
    ]
    return [normalize_tag(tag) for tag in tags]


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_:\-äöüß]+", "_", tag)


def generate_cards(senses: Sequence[AnkiVerbSense], *, include_low_quality: bool = False) -> List[AnkiCard]:
    """Generate all default Anki cards."""
    usable = [sense for sense in senses if include_low_quality or sense.is_quality_ok]
    cards: List[AnkiCard] = []

    for sense in usable:
        cards.extend(generate_context_cards(sense))
        cards.extend(generate_prefix_cloze_cards(sense))
        cards.extend(generate_construction_cards(sense))

    cards.extend(generate_separability_cards(usable))
    cards.extend(generate_contrast_cards(usable))
    return dedupe_cards(cards)


def generate_context_cards(sense: AnkiVerbSense) -> List[AnkiCard]:
    blank = make_context_blank(sense)
    if not blank:
        return []
    front_example, answer = blank
    front = f"Completa la frase:<br><br>{escape(front_example)}<br><br>Pista: {escape(sense.anki_hint_es or sense.gloss_es)}"
    back = card_back(
        answer=answer,
        example_de=sense.example_de,
        example_es=sense.example_es,
        gloss_es=sense.gloss_es,
        gloss_en=sense.gloss_en,
        construction=sense.construction,
        separability=sense.separability,
        source_url=sense.wiktionary_url,
    )
    return [
        AnkiCard(
            cardtype="context",
            key=f"context:{sense.derived}:{sense.sense_id}",
            front=front,
            back=back,
            tags=build_tags(sense, "context"),
        )
    ]


def generate_prefix_cloze_cards(sense: AnkiVerbSense) -> List[AnkiCard]:
    blank = make_prefix_blank(sense)
    if not blank:
        return []
    front_example, answer = blank
    front = (
        f"Completa el prefijo:<br><br>{escape(front_example)}"
        f"<br><br>Pista: {escape(sense.gloss_es)}<br>Base: {escape(sense.base)}"
    )
    back = f"{escape(answer)}<br><br>{escape(sense.derived)} = {escape(sense.gloss_es)}<br>Separabilidad: {escape(sense.separability)}"
    return [
        AnkiCard(
            cardtype="prefix_cloze",
            key=f"prefix:{sense.derived}:{sense.sense_id}",
            front=front,
            back=back,
            tags=build_tags(sense, "prefix_cloze"),
        )
    ]


def generate_construction_cards(sense: AnkiVerbSense) -> List[AnkiCard]:
    if not sense.construction:
        return []
    prompt = sense.construction_es or sense.gloss_es
    front = f'¿Qué construcción alemana corresponde a "{escape(prompt)}"?<br><br>Verbo: {escape(sense.derived)}'
    back = f"{escape(sense.construction)}<br><br>Ejemplo:<br>{escape(sense.example_de)}"
    return [
        AnkiCard(
            cardtype="construction",
            key=f"construction:{sense.derived}:{sense.sense_id}",
            front=front,
            back=back,
            tags=build_tags(sense, "construction"),
        )
    ]


def generate_separability_cards(senses: Sequence[AnkiVerbSense]) -> List[AnkiCard]:
    cards: List[AnkiCard] = []
    by_derived: Dict[str, AnkiVerbSense] = {}
    for sense in senses:
        by_derived.setdefault(sense.derived, sense)

    for sense in by_derived.values():
        present = sense.separable_sentence_pattern or sense.present_3sg
        perfect = build_perfect_phrase(sense)
        if not present or not perfect or sense.separability not in {"separable", "inseparable"}:
            continue
        answer = "Sí." if sense.separability == "separable" else "No."
        front = f"¿Es separable?<br><br>{escape(sense.derived)}"
        back = f"{answer}<br><br>Presente:<br>{escape(present)}<br><br>Perfekt:<br>{escape(perfect)}"
        cards.append(
            AnkiCard(
                cardtype="separability",
                key=f"separability:{sense.derived}",
                front=front,
                back=back,
                tags=build_tags(sense, "separability"),
            )
        )
    return cards


def build_perfect_phrase(sense: AnkiVerbSense) -> str:
    if not sense.participle_ii:
        return ""
    auxiliary = sense.perfect_auxiliary or "hat"
    return f"Er {auxiliary} es {sense.participle_ii}."


def generate_contrast_cards(senses: Sequence[AnkiVerbSense]) -> List[AnkiCard]:
    cards: List[AnkiCard] = []
    grouped: Dict[str, List[AnkiVerbSense]] = {}
    for sense in senses:
        grouped.setdefault(sense.base, []).append(sense)

    for base, group in grouped.items():
        selected = select_contrast_senses(group)
        if len(selected) < 3:
            continue
        front_lines = [f"Familia: {escape(base)}", "", "Elige el verbo correcto:"]
        back_lines: List[str] = []
        for index, sense in enumerate(selected, start=1):
            front_lines.append(f"{index}. {escape(sense.gloss_es or sense.gloss_en or sense.gloss_de)}")
            answer = sense.construction or sense.derived
            back_lines.append(f"{index}. {escape(answer)}")
        exemplar = selected[0]
        cards.append(
            AnkiCard(
                cardtype="contrast",
                key=f"contrast:{base}:{','.join(s.derived for s in selected)}",
                front="<br>".join(front_lines),
                back="<br>".join(back_lines),
                tags=build_tags(exemplar, "contrast"),
            )
        )
    return cards


def select_contrast_senses(group: Sequence[AnkiVerbSense]) -> List[AnkiVerbSense]:
    seen: set[str] = set()
    ranked = sorted(group, key=contrast_rank)
    selected: List[AnkiVerbSense] = []
    for sense in ranked:
        if sense.derived in seen:
            continue
        if not (sense.gloss_es or sense.gloss_en or sense.gloss_de):
            continue
        selected.append(sense)
        seen.add(sense.derived)
        if len(selected) >= 7:
            break
    return selected


def contrast_rank(sense: AnkiVerbSense) -> Tuple[int, int, str]:
    frequency_rank = {"high": 0, "medium": 1, "low": 2, "rare": 3, "unknown": 2}
    completeness = 0 if sense.example_de and sense.gloss_es else 1
    return (frequency_rank.get(sense.frequency_bucket, 2), completeness, sense.derived)


def dedupe_cards(cards: Iterable[AnkiCard]) -> List[AnkiCard]:
    seen: set[str] = set()
    unique: List[AnkiCard] = []
    for card in cards:
        if card.key in seen:
            continue
        seen.add(card.key)
        unique.append(card)
    return unique


def card_back(
    *,
    answer: str,
    example_de: str,
    example_es: str,
    gloss_es: str,
    gloss_en: str,
    construction: str,
    separability: str,
    source_url: str,
) -> str:
    parts = [
        f'<div class="answer">{escape(answer)}</div>',
        f'<div class="example">{escape(example_de)}</div>',
    ]
    if example_es:
        parts.append(f'<div class="translation">{escape(example_es)}</div>')
    meta = [
        f"<p><b>Significado:</b> {escape(gloss_es)}</p>",
        f"<p><b>English:</b> {escape(gloss_en)}</p>" if gloss_en else "",
        f"<p><b>Construcción:</b> {escape(construction)}</p>" if construction else "",
        f"<p><b>Separabilidad:</b> {escape(separability)}</p>",
        f"<p><b>Fuente:</b> {escape(source_url)}</p>" if source_url else "",
    ]
    parts.append(f'<div class="meta">{"".join(item for item in meta if item)}</div>')
    return "\n".join(parts)


def escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=False)


def validate_cards(cards: Sequence[AnkiCard]) -> List[str]:
    errors: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for card in cards:
        if not card.front.strip():
            errors.append(f"{card.key}: empty front")
        if not card.back.strip():
            errors.append(f"{card.key}: empty back")
        if card.cardtype == "context" and "___" not in card.front:
            errors.append(f"{card.key}: context card has no blank")
        if not card.tags:
            errors.append(f"{card.key}: missing tags")
        dup_key = (card.key, card.cardtype)
        if dup_key in seen:
            errors.append(f"{card.key}: duplicate card")
        seen.add(dup_key)
    return errors


def write_csv_export(cards: Sequence[AnkiCard], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[AnkiCard]] = {}
    for card in cards:
        grouped.setdefault(card.cardtype, []).append(card)

    for cardtype, card_group in grouped.items():
        path = out_dir / f"anki_cards_{cardtype}.csv"
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["Front", "Back", "Tags"])
            writer.writeheader()
            for card in card_group:
                writer.writerow({"Front": card.front, "Back": card.back, "Tags": " ".join(card.tags)})


def write_apkg(cards: Sequence[AnkiCard], out_path: Path, deck_name: str) -> None:
    try:
        import genanki
    except ImportError as exc:
        raise RuntimeError("genanki is required for .apkg output. Install requirements.txt first.") from exc

    model = genanki.Model(
        stable_int_id(f"{deck_name}::BasicModel"),
        "German Prefix Verb Card",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": '<div class="front">{{Front}}</div>',
                "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
            }
        ],
        css=CARD_CSS,
    )
    deck = genanki.Deck(stable_int_id(deck_name), deck_name)
    for card in cards:
        note = genanki.Note(model=model, fields=[card.front, card.back], tags=card.tags)
        note.guid = stable_guid(card.key)
        deck.add_note(note)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(out_path))


CARD_CSS = """
.card {
  font-family: Arial, sans-serif;
  font-size: 20px;
  text-align: left;
  line-height: 1.45;
}
.answer {
  font-size: 28px;
  font-weight: bold;
  margin: 12px 0;
}
.example {
  margin-top: 12px;
}
.translation {
  color: #555;
  margin-top: 6px;
}
.meta {
  margin-top: 16px;
  font-size: 15px;
  color: #444;
}
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Anki deck for German prefixed verbs.")
    parser.add_argument("--input", type=Path, required=True, help="Input .csv or .json from derive_verbs.py.")
    parser.add_argument("--out", type=Path, default=Path("German_Prefix_Verbs.apkg"), help="Output .apkg path.")
    parser.add_argument("--out-dir", type=Path, default=Path("anki_export"), help="CSV export directory.")
    parser.add_argument("--deck-name", default="German Prefix Verbs")
    parser.add_argument("--format", choices=["apkg", "csv"], default="apkg")
    parser.add_argument("--include-low-quality", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        senses = load_input(args.input)
        cards = generate_cards(senses, include_low_quality=args.include_low_quality)
        errors = validate_cards(cards)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.validate_only:
        if errors:
            print("Validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print(f"Validation ok: {len(cards)} cards from {len(senses)} senses.")
        return 0

    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    try:
        if args.format == "csv":
            write_csv_export(cards, args.out_dir)
            print(f"Wrote {len(cards)} cards to {args.out_dir}.")
        else:
            write_apkg(cards, args.out, args.deck_name)
            print(f"Wrote {len(cards)} cards to {args.out}.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

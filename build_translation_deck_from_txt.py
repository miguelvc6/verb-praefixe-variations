#!/usr/bin/env python3
"""Build a two-direction Anki deck from a simple translations text file.

The input format is intentionally small:

    ablegen
    quitarse una prenda o accesorio
    depositar / dejar algo en un lugar

    anlegen
    ponerse ropa, joyas o un accesorio

Each block is separated by one or more blank lines. The first line of a block is
the German verb, and the remaining lines are Spanish translations.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


GERMAN_VERB_RE = re.compile(r"^[a-zäöüß]+$")


@dataclass(frozen=True)
class TranslationNote:
    german: str
    spanish: tuple[str, ...]
    base: str

    @property
    def tags(self) -> list[str]:
        return ["verb_translation", f"family::{self.base}", f"verb::{self.german}"]


def stable_int_id(text: str, modulo: int = 10**10) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def stable_guid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_base_verbs(path: Optional[Path]) -> list[str]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as input_file:
        bases = [line.strip().lower() for line in input_file if line.strip()]
    return sorted(dict.fromkeys(bases), key=len, reverse=True)


def find_base_verb(german: str, base_verbs: Sequence[str]) -> str:
    german = german.lower()
    for base in base_verbs:
        if german == base or german.endswith(base):
            return base
    return german


def parse_translation_blocks(text: str, base_verbs: Sequence[str]) -> list[TranslationNote]:
    notes_by_verb: dict[str, TranslationNote] = {}

    for raw_block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        german = lines[0].lower()
        if not GERMAN_VERB_RE.fullmatch(german):
            continue

        spanish = tuple(dict.fromkeys(lines[1:]))
        if not spanish:
            continue

        existing = notes_by_verb.get(german)
        if existing:
            merged = tuple(dict.fromkeys((*existing.spanish, *spanish)))
            notes_by_verb[german] = TranslationNote(
                german=german,
                spanish=merged,
                base=existing.base,
            )
            continue

        notes_by_verb[german] = TranslationNote(
            german=german,
            spanish=spanish,
            base=find_base_verb(german, base_verbs),
        )

    return list(notes_by_verb.values())


def load_translation_notes(input_path: Path, base_verbs_path: Optional[Path]) -> list[TranslationNote]:
    return parse_translation_blocks(
        input_path.read_text(encoding="utf-8"),
        load_base_verbs(base_verbs_path),
    )


def render_spanish(translations: Iterable[str]) -> str:
    items = [f"<li>{html.escape(item)}</li>" for item in translations]
    return "<ul>" + "".join(items) + "</ul>"


def write_apkg(notes: Sequence[TranslationNote], out_path: Path, deck_name: str) -> None:
    try:
        import genanki
    except ImportError as exc:
        raise RuntimeError("genanki is required for .apkg output. Install requirements.txt first.") from exc

    model = genanki.Model(
        stable_int_id(f"{deck_name}::GermanSpanishTranslationModel"),
        "German-Spanish Verb Translation",
        fields=[
            {"name": "German"},
            {"name": "Spanish"},
            {"name": "Base"},
        ],
        templates=[
            {
                "name": "German -> Spanish",
                "qfmt": '<div class="prompt">{{German}}</div>',
                "afmt": '{{FrontSide}}<hr id="answer"><div class="answer">{{Spanish}}</div>',
            },
            {
                "name": "Spanish -> German",
                "qfmt": '<div class="prompt">{{Spanish}}</div>',
                "afmt": '{{FrontSide}}<hr id="answer"><div class="answer">{{German}}</div>',
            },
        ],
        css=CARD_CSS,
    )

    deck = genanki.Deck(stable_int_id(deck_name), deck_name)
    for note_data in notes:
        note = genanki.Note(
            model=model,
            fields=[note_data.german, render_spanish(note_data.spanish), note_data.base],
            tags=note_data.tags,
        )
        note.guid = stable_guid(f"{deck_name}:{note_data.german}")
        deck.add_note(note)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(out_path))


CARD_CSS = """
.card {
  font-family: Arial, sans-serif;
  font-size: 22px;
  line-height: 1.45;
  text-align: left;
}
.prompt {
  font-size: 28px;
  font-weight: 700;
}
.answer {
  font-size: 24px;
}
ul {
  margin: 0;
  padding-left: 1.25em;
}
li {
  margin: 0.35em 0;
}
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Anki deck from translations.txt.")
    parser.add_argument("--input", type=Path, default=Path("translations.txt"), help="Translation text file.")
    parser.add_argument("--base-verbs", type=Path, default=Path("verbs.txt"), help="Base verb list for family tags.")
    parser.add_argument("--out", type=Path, default=Path("German_Spanish_Verb_Translations.apkg"))
    parser.add_argument("--deck-name", default="German Spanish Verb Translations")
    parser.add_argument("--validate-only", action="store_true", help="Parse input and report note count without writing.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        notes = load_translation_notes(args.input, args.base_verbs)
        if not notes:
            raise ValueError(f"No translation notes found in {args.input}.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.validate_only:
        family_count = len({note.base for note in notes})
        print(f"Validation ok: {len(notes)} notes across {family_count} verb families.")
        return 0

    try:
        write_apkg(notes, args.out, args.deck_name)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(notes)} notes / {len(notes) * 2} cards to {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

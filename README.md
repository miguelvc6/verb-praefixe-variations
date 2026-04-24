# German Prefix Verb Anki Decks

This project collects German prefixed verbs from German Wiktionary, optionally enriches weak entries with the OpenAI API, and builds Anki decks for Spanish-speaking learners.

The pipeline has two stages:

1. `derive_verbs.py` discovers and enriches verb data, writing `out.csv` and `out.json`.
2. `build_anki_deck.py` turns `out.json` or `out.csv` into `.apkg` decks or Anki-importable CSV files.

## Setup

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For OpenAI enrichment, put your API key in `.env`:

```text
OPENAI_API_KEY=...
```

## Generate Verb Data

Basic Wiktionary extraction:

```powershell
python derive_verbs.py --verbs verbs.txt --out out
```

Extraction with OpenAI enrichment:

```powershell
python derive_verbs.py --verbs verbs.txt --out out --enrich-openai
```

Refresh the OpenAI cache after prompt/schema changes:

```powershell
python derive_verbs.py --verbs verbs.txt --out out --enrich-openai --refresh-openai-cache
```

Useful options:

- `--verbs`: comma-separated verbs or a text file with one base verb per line.
- `--out`: output stem; writes both `.csv` and `.json`.
- `--include-ge-prefix`: include `ge-` candidates, disabled by default to avoid participle false positives.
- `--enrich-openai`: fill missing or weak Anki-critical fields through OpenAI.
- `--max-openai-rows N`: test enrichment on a small batch.
- `--validate-only`: validate discovered data without writing output.

## Build Decks

Validate current deck data:

```powershell
python build_anki_deck.py --input out.json --format csv --out-dir anki_export --validate-only
```

Build one complete default deck:

```powershell
python build_anki_deck.py --input out.json --format apkg --out German_Prefix_Verbs.apkg --deck-name "German Prefix Verbs"
```

Export CSV files instead:

```powershell
python build_anki_deck.py --input out.json --format csv --out-dir anki_export
```

## Recommended Two-Deck Output

Translation-only deck:

```powershell
python build_anki_deck.py --input out.json --format apkg --out German_Prefix_Verbs_Translations.apkg --deck-name "German Prefix Verbs - Translations" --card-types translation_de_to_es,translation_es_to_de
```

Practice deck without translation cards:

```powershell
python build_anki_deck.py --input out.json --format apkg --out German_Prefix_Verbs_Practice.apkg --deck-name "German Prefix Verbs - Practice" --card-types context_cloze,prefix_cloze,separability,contrast
```

CSV equivalents:

```powershell
python build_anki_deck.py --input out.json --format csv --out-dir anki_export_translations --card-types translation_de_to_es,translation_es_to_de
python build_anki_deck.py --input out.json --format csv --out-dir anki_export_practice --card-types context_cloze,prefix_cloze,separability,contrast
```

## Prefix Semantics Deck

`build_prefix_semantics_cards.py` builds a separate deck for prefix meaning intuitions, such as `ab-` suggesting separation/removal or `zurück-` suggesting return/restoration. These cards are not hard rules; they teach useful semantic tendencies and caveats.

Build an `.apkg` deck using examples from `out.json`:

```powershell
python build_prefix_semantics_cards.py --input out.json --format apkg --out German_Prefix_Semantics.apkg --deck-name "German Prefix Semantics"
```

Export CSV instead:

```powershell
python build_prefix_semantics_cards.py --input out.json --out prefix_semantics.csv
```

Generate reverse cards too, asking from meaning intuition to prefix:

```powershell
python build_prefix_semantics_cards.py --input out.json --include-reverse --out prefix_semantics.csv
```

Validate without writing files:

```powershell
python build_prefix_semantics_cards.py --input out.json --validate-only
```

If `--input` is omitted, the script still generates cards from its built-in prefix table and fallback examples.

## Card Types

Default card types:

- `context_cloze`: complete a German sentence with the verb, participle, or separated prefix.
- `prefix_cloze`: complete only the missing prefix.
- `translation_de_to_es`: German verb to Spanish meaning.
- `translation_es_to_de`: disambiguated Spanish prompt to German verb.
- `separability`: separability plus present/perfect behavior.
- `contrast`: contextual multiple choice among verbs from the same base family.

Optional:

- `construction`: grammar-pattern recall cards. These are disabled by default because construction details are usually better shown on card backs.

Select card types with:

```powershell
python build_anki_deck.py --input out.json --format apkg --out custom.apkg --card-types context_cloze,prefix_cloze,contrast
```

## Quality Rules

The deck builder skips low-quality senses by default. A sense may be considered low quality if it has missing examples, missing cloze answers, generic constructions, invalid verbal answers, placeholder Wiktionary content, bad participles, or unreliable separability forms.

Strict validation checks every sense, including skipped low-quality rows:

```powershell
python build_anki_deck.py --input out.json --validate-only --strict
```

Use strict mode when improving the dataset itself. For normal deck generation, omit `--strict`; low-quality rows are skipped automatically.

## Tests

```powershell
python -m pytest -q
```

## Output Schema

The enriched JSON supports nested senses. CSV output is flattened to one row per sense. Important fields include:

```text
base, derived, sense_id, prefix, separability, gloss_de, gloss_es, gloss_en,
example_de, example_es, example_en, example_de_with_blank, answer,
construction, construction_es, present_3sg, perfect_auxiliary, participle_ii,
is_quality_ok, quality_flags, wiktionary_url, source
```

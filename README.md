# Verb Prefix Derivations

Script to collect common prefixed (derived) German verbs from the German Wiktionary and export them as CSV and JSON.

## Requirements

- Python 3.12
- Dependencies: `requests`, `beautifulsoup4`, `pytest` (for tests only)

Install dependencies into the current environment:

```bash
python -m pip install --upgrade pip
python -m pip install requests beautifulsoup4 pytest
```

## Usage

Run the scraper by providing either a comma separated list of verbs or a file (one verb per line). The script enforces a fixed schema with glosses and optional translations/examples.

```bash
python derive_verbs.py \
  --verbs "gehen,nehmen,stehen,tragen,ziehen" \
  --out data/derivados
```

Key options:

- `--verbs` comma separated list of base verbs or a path to a text file.
- `--out` output file stem; the script writes both `.csv` and `.json`.

The output rows/objects use these columns:
`base, derived, prefix, separability, pos, gloss_de, gloss_es, gloss_en, example, wiktionary_url`

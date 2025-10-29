"""Utilities to derive German verb compounds from Wiktionary entries.

This module provides a command line interface that accepts a list of German
verbs and scrapes https://de.wiktionary.org to discover prefixed compounds.
The output is written both as CSV and JSON files with fixed schemas.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (Dict, Iterable, Iterator, List, Optional, Sequence, Set,
                    Tuple, Union)
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL_TEMPLATE = "https://de.wiktionary.org/wiki/{}"
DERIVED_SECTION_TITLES = {
    "Abgeleitete Begriffe",
    "Zusammensetzungen",
    "Wortbildungen",
    "Wortbildungen (Verb)",
    "Verwandte Begriffe",
}

# Prefix definitions without trailing hyphen to simplify matching.
SEPARABLE_PREFIXES: Dict[str, str] = {
    "ab": "ab-",
    "an": "an-",
    "auf": "auf-",
    "aus": "aus-",
    "bei": "bei-",
    "ein": "ein-",
    "fest": "fest-",
    "fort": "fort-",
    "her": "her-",
    "hin": "hin-",
    "los": "los-",
    "mit": "mit-",
    "nach": "nach-",
    "vor": "vor-",
    "weg": "weg-",
    "weiter": "weiter-",
    "zu": "zu-",
    "zurück": "zurück-",
    "zusammen": "zusammen-",
}
INSEPARABLE_PREFIXES: Dict[str, str] = {
    "be": "be-",
    "emp": "emp-",
    "ent": "ent-",
    "er": "er-",
    "ge": "ge-",
    "miss": "miss-",
    "ver": "ver-",
    "zer": "zer-",
}
AMBIGUOUS_PREFIXES: Dict[str, str] = {
    "durch": "durch-",
    "hinter": "hinter-",
    "über": "über-",
    "unter": "unter-",
    "um": "um-",
    "wider": "wider-",
}

# Build a single ordered list of (prefix_str, label_with_dash, separability)
PREFIX_ORDER: Sequence[Tuple[str, str, str]] = tuple(
    sorted(
        list((p, lbl, "separable") for p, lbl in SEPARABLE_PREFIXES.items())
        + list((p, lbl, "inseparable") for p, lbl in INSEPARABLE_PREFIXES.items())
        + list((p, lbl, "ambiguous") for p, lbl in AMBIGUOUS_PREFIXES.items()),
        key=lambda item: len(item[0]),
        reverse=True,  # longest prefix first
    )
)

CSV_HEADERS = [
    "base",
    "derived",
    "prefix",
    "separability",
    "pos",
    "gloss_de",
    "gloss_es",
    "gloss_en",
    "example",
    "wiktionary_url",
]


@dataclass
class PageContent:
    """HTML content fetched from Wiktionary."""
    url: str
    soup: BeautifulSoup


@dataclass
class DerivedVerb:
    """Structured representation of a derived German verb."""
    base: str
    derived: str
    prefix: str
    separability: str
    pos: str
    gloss_de: str
    gloss_es: str
    gloss_en: str
    example: str
    wiktionary_url: str


class WiktionaryClient:
    """HTTP client with retry, caching, and rate limiting for Wiktionary."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 0.8,
        min_interval: float = 0.7,
        user_agent: str = "verb-praefixe-collector/0.1 (+https://example.org)",
    ) -> None:
        self.session = requests.Session()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.min_interval = min_interval
        self.last_request: float = 0.0
        self.cache: Dict[str, Optional[PageContent]] = {}
        self.headers = {"User-Agent": user_agent}

    def fetch(self, lemma: str) -> Optional[PageContent]:
        """Fetch a lemma page returning parsed HTML or None if unavailable."""
        normalized = lemma.strip()
        if not normalized:
            return None
        cache_key = normalized.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = BASE_URL_TEMPLATE.format(quote(normalized, safe="/-"))
        for attempt in range(1, self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self.session.get(url, headers=self.headers, timeout=20)
            except requests.RequestException:
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 404:
                self.cache[cache_key] = None
                return None

            if 200 <= response.status_code < 300:
                soup = BeautifulSoup(response.text, "html.parser")
                content = PageContent(url=response.url, soup=soup)
                self.cache[cache_key] = content
                return content

            if response.status_code >= 500 or response.status_code == 429:
                self._sleep_backoff(attempt)
                continue

            # Other HTTP errors → abort
            break

        self.cache[cache_key] = None
        return None

    def _respect_rate_limit(self) -> None:
        """Sleep if the previous request was too recent."""
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.monotonic()

    def _sleep_backoff(self, attempt: int) -> None:
        """Pause using exponential backoff."""
        sleep_for = self.backoff_factor * (2 ** (attempt - 1))
        time.sleep(sleep_for)


def derive_for_bases(
    bases: Iterable[str],
    *,
    max_per_prefix: Optional[int] = None,
    include_ambiguous: bool = False,
    client: Optional[WiktionaryClient] = None,
    progress: bool = True,
) -> List[DerivedVerb]:
    """Collect derived verbs for a list of base verbs."""
    wiktionary = client or WiktionaryClient()
    all_results: List[DerivedVerb] = []

    for base in [b.strip().lower() for b in bases if b.strip()]:
        if not base:
            continue
        if progress:
            print(f"[{base}] processing")
        base_page = wiktionary.fetch(base)
        if not base_page:
            if progress:
                print(f"  warning: no Wiktionary entry found for '{base}'.")
            continue

        german_section = extract_deutsch_section(base_page.soup)
        if not german_section:
            if progress:
                print(f"  warning: missing 'Deutsch' section for '{base}'.")
            continue

        candidates = extract_candidate_lemmas_anywhere(german_section, base)
        if progress:
            print(f"  found {len(candidates)} raw candidates for '{base}'.")

        # Fallback: si nada salió de las secciones, escanea toda la página
        if not candidates:
            wide = extract_candidate_lemmas_anywhere(base_page.soup, base)
            if progress:
                print(f"  fallback(anywhere): +{len(wide)} candidates for '{base}'.")
            candidates |= wide        

        seen: Set[str] = set()
        prefix_counter: Dict[str, int] = {}

        for candidate in sorted(candidates):
            prefix_info = identify_prefix(candidate, base)
            if not prefix_info:
                continue

            prefix_label, separability = prefix_info
            if separability == "ambiguous" and not include_ambiguous:
                continue

            if max_per_prefix is not None:
                taken = prefix_counter.get(prefix_label, 0)
                if taken >= max_per_prefix:
                    continue

            if candidate in seen:
                continue

            candidate_page = wiktionary.fetch(candidate)
            if not candidate_page:
                continue

            entry = extract_verb_entry(
                base=base,
                derived=candidate,
                prefix_label=prefix_label,
                separability=separability,
                page=candidate_page,
            )
            if not entry:
                continue

            all_results.append(entry)
            seen.add(candidate)
            prefix_counter[prefix_label] = prefix_counter.get(prefix_label, 0) + 1
            if progress:
                gloss_preview = (entry.gloss_de or "").split(".")[0][:80]
                print(f"    ok: {candidate} ({prefix_label}, {separability}) -> {gloss_preview}")

    return all_results


def extract_candidate_lemmas_anywhere(
    section: Union[BeautifulSoup, Sequence[object]], base: str
) -> Set[str]:
    """Collect candidate lemmas by scanning the given German section (or whole page)."""
    candidates: Set[str] = set()

    def iter_anchors() -> Iterable[Tag]:
        if hasattr(section, "find_all"):
            yield from section.find_all("a", href=True)
        else:
            for node in section:
                if isinstance(node, Tag):
                    yield from node.find_all("a", href=True)

    all_prefixes = [p for p, _, _ in PREFIX_ORDER]
    for anchor in iter_anchors():
        href = anchor["href"]
        lemma = normalize_lemma_from_href(href)
        if not lemma:
            continue
        lower = lemma.lower()
        if "_" in lower:
            continue
        if any(lower.startswith(pref) for pref in all_prefixes) and lower.endswith(base) and len(lower) > len(base):
            candidates.add(lower)

    return candidates


def normalize_lemma_from_href(href: str) -> Optional[str]:
    """Normalize a Wiktionary anchor href into a lemma string."""
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        parsed = urlparse(href)
        if not parsed.path.startswith("/wiki/"):
            return None
        path = parsed.path
    else:
        path = href

    if not path.startswith("/wiki/"):
        return None

    target = path.split("/wiki/", 1)[1]
    target = target.split("#", 1)[0]
    if ":" in target:  # skip namespaces
        return None
    return unquote(target).strip()


def extract_deutsch_section(soup: BeautifulSoup) -> List[object]:
    """Return nodes within the German section of a Wiktionary page.

    More robust: accept ids like 'Deutsch', 'Deutsch_(1)', etc.
    Fallback: return whole document children if not found.
    """
    # try: any h2/span whose id starts with 'Deutsch'
    for h2 in soup.find_all("h2"):
        span = h2.find("span", id=True)
        if span and span.get("id", "").startswith("Deutsch"):
            nodes: List[object] = []
            for sibling in h2.next_siblings:
                if isinstance(sibling, Tag) and sibling.name == "h2":
                    break
                nodes.append(sibling)
            return nodes

    # fallback — better to keep working than to drop everything
    return list(soup.body.children) if soup.body else list(soup.children)



def identify_prefix(lemma: str, base: str) -> Optional[Tuple[str, str]]:
    """Identify prefix label and separability for a derived lemma."""
    # Check the longest matching prefix first.
    for prefix, label, separability in PREFIX_ORDER:
        if lemma.startswith(prefix):
            remainder = lemma[len(prefix):]
            if remainder == base:
                return label, separability
    return None


def extract_verb_entry(
    base: str,
    derived: str,
    prefix_label: str,
    separability: str,
    page: PageContent,
) -> Optional[DerivedVerb]:
    """Extract relevant information from a derived verb page."""
    german_nodes = extract_deutsch_section(page.soup)
    if not german_nodes:
        return None

    verb_header = find_heading(german_nodes, ("Verb",))
    if not verb_header:
        return None

    gloss_de = extract_first_gloss(verb_header)
    example = extract_example(verb_header)
    translations = extract_translations(verb_header)

    return DerivedVerb(
        base=base,
        derived=derived,
        prefix=prefix_label,
        separability=separability,
        pos="Verb",
        gloss_de=gloss_de,
        gloss_es=translations.get("spanisch", ""),
        gloss_en=translations.get("englisch", ""),
        example=example,
        wiktionary_url=page.url,
    )


def find_heading(
    section_nodes: Sequence[object],
    ids_or_titles: Tuple[str, ...],
) -> Optional[Tag]:
    """Find the first heading matching one of the provided labels."""
    normalized_targets = tuple(item.lower() for item in ids_or_titles)
    for node in section_nodes:
        if not isinstance(node, Tag):
            continue
        if node.name not in {"h3", "h4", "h5"}:
            continue
        span = node.find("span", id=True)
        if span and span.get("id", "").split("_")[0].lower() in normalized_targets:
            return node
        heading_text = node.get_text(strip=True).lower()
        if heading_text in normalized_targets:
            return node
    return None


def extract_first_gloss(verb_heading: Tag) -> str:
    """Extract the first definition gloss for a verb entry."""
    gloss = ""
    for node in iterate_section_after_heading(verb_heading):
        if isinstance(node, Tag):
            # 'Bedeutungen' section
            if node.name in {"h3", "h4", "h5"} and "bedeut" in node.get_text(strip=True).lower():
                gloss = _gloss_from_definition_block(node)
                if gloss:
                    break
            # Fallback: first list or definition block
            if node.name in {"ol", "dl"} and not gloss:
                first_li = node.find("li")
                gloss = clean_text(first_li or node)
                if gloss:
                    break
    return gloss


def _gloss_from_definition_block(heading: Tag) -> str:
    """Return the first bullet under a Bedeutungen section."""
    for node in iterate_section_after_heading(heading):
        if isinstance(node, Tag):
            if node.name == "ol":
                first_li = node.find("li")
                if first_li:
                    return clean_text(first_li)
            if node.name in {"h3", "h4", "h5"}:
                return ""
    return ""


def extract_example(verb_heading: Tag) -> str:
    """Retrieve one example sentence if available."""
    for node in iterate_section_after_heading(verb_heading):
        if isinstance(node, Tag):
            if node.name in {"h3", "h4", "h5"}:
                title = node.get_text(strip=True).lower()
                if any(keyword in title for keyword in ("beispiel", "beispiele")):
                    sentence = _first_sentence_in_block(node)
                    if sentence:
                        return sentence
            # stop if a new Verb section starts (rare within same Deutsch block)
            if node.name in {"h3", "h4"} and node is not verb_heading:
                span = node.find("span", id=True)
                if span and span.get("id", "").startswith("Verb"):
                    break
    return ""


def _first_sentence_in_block(heading: Tag) -> str:
    """Get the first sentence within list or paragraph after a heading."""
    for node in iterate_section_after_heading(heading):
        if isinstance(node, Tag):
            if node.name in {"ul", "ol"}:
                first_item = node.find("li")
                if first_item:
                    return clean_text(first_item)
            if node.name == "p":
                return clean_text(node)
            if node.name in {"h3", "h4", "h5"}:
                return ""
    return ""


def extract_translations(verb_heading: Tag) -> Dict[str, str]:
    """Extract Spanish and English translations if present."""
    translations: Dict[str, str] = {}
    for node in iterate_section_after_heading(verb_heading):
        if isinstance(node, Tag):
            if node.name in {"h3", "h4", "h5"}:
                title = node.get_text(strip=True).lower()
                if "übersetz" in title:
                    translations.update(_parse_translation_tables(node))
                elif node is not verb_heading and title.startswith("verb"):
                    break
    return translations


def _parse_translation_tables(heading: Tag) -> Dict[str, str]:
    """Parse translation tables following a heading."""
    collected: Dict[str, str] = {}
    for node in iterate_section_after_heading(heading):
        if isinstance(node, Tag):
            if node.name == "table":
                parsed = _extract_translations_from_table(node)
                for key, value in parsed.items():
                    if key not in collected and value:
                        collected[key] = value
            if node.name in {"h3", "h4", "h5"}:
                break
    return collected


def _extract_translations_from_table(table: Tag) -> Dict[str, str]:
    """Convert a Wiktionary translation table into a language map."""
    result: Dict[str, str] = {}
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        language = clean_text(cells[0]).lower()
        if not language:
            continue
        if language not in {"spanisch", "englisch"}:
            continue
        value = clean_text(cells[1])
        result[language] = value
    return result


def iterate_section_after_heading(heading: Tag) -> Iterator[object]:
    """Yield siblings after a heading until the next same-level heading or H2."""
    stop_tags = {"h2"}
    level = heading.name
    current_id = _heading_id(heading)

    for sibling in heading.next_siblings:
        if isinstance(sibling, Tag):
            if sibling.name in stop_tags:
                break
            # Stop if a same/higher level heading of a different subsection appears
            if _is_same_or_higher_level(level, sibling.name):
                sibling_id = _heading_id(sibling)
                if sibling_id != current_id:
                    break
                if sibling_id is None and sibling.name == heading.name:
                    break
        yield sibling


def _is_same_or_higher_level(current: str, other: str) -> bool:
    """Return True if other heading level is same or higher priority."""
    order = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    return order.get(other, 10) <= order.get(current, 10)


def _heading_id(tag: Tag) -> Optional[str]:
    """Extract the span id for a heading if present."""
    span = tag.find("span", id=True)
    if span:
        return span.get("id")
    return None


def clean_text(element: Optional[Tag]) -> str:
    """Normalize text content by removing references and whitespace."""
    if element is None:
        return ""
    snippet = BeautifulSoup(str(element), "html.parser")
    for sup in snippet.find_all("sup", class_="reference"):
        sup.decompose()
    text = snippet.get_text(" ", strip=True)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def write_outputs(
    derived_verbs: Sequence[DerivedVerb],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Persist results to CSV and JSON files."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for item in derived_verbs:
            writer.writerow(asdict(item))

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            [asdict(item) for item in derived_verbs],
            json_file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Derive prefixed German verbs from Wiktionary."
    )
    parser.add_argument(
        "--verbs",
        type=str,
        help="Comma-separated list of base verbs (e.g., 'gehen,nehmen').",
        default=None,
    )
    parser.add_argument(
        "--verbs-file",
        type=Path,
        help="Path to a text file with one verb per line.",
        default=Path("verbs.txt"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Path to the CSV output file.",
        default=Path("out.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Path to the JSON output file.",
        default=Path("out.json"),
    )
    parser.add_argument(
        "--max-per-prefix",
        type=int,
        default=None,
        help="Maximum number of derived verbs to keep per prefix.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Include verbs with ambiguous prefixes (durch-, über-, usw.).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def load_verbs_from_args(args: argparse.Namespace) -> List[str]:
    """Load the list of verbs based on CLI arguments."""
    verbs: Set[str] = set()
    if args.verbs:
        verbs.update(v.strip() for v in args.verbs.split(",") if v.strip())
    if args.verbs_file:
        if not args.verbs_file.exists():
            raise FileNotFoundError(f"Verb list file not found: {args.verbs_file}")
        content = args.verbs_file.read_text(encoding="utf-8")
        verbs.update(line.strip() for line in content.splitlines() if line.strip())
    if not verbs:
        raise ValueError("No verbs provided. Use --verbs or --verbs-file.")
    return sorted(verbs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point."""
    args = parse_args(argv)
    try:
        verbs = load_verbs_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    derived = derive_for_bases(
        verbs,
        max_per_prefix=args.max_per_prefix,
        include_ambiguous=args.include_ambiguous,
        progress=not args.quiet,
    )
    if not derived:
        print("No derived verbs found.", file=sys.stderr)
        return 2

    write_outputs(derived, args.out_csv, args.out_json)
    if not args.quiet:
        print(f"Wrote {len(derived)} rows to {args.out_csv} and {args.out_json}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

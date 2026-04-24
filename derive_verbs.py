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
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL_TEMPLATE = "https://de.wiktionary.org/wiki/{}"
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

NON_LEMMA_TITLE_KEYWORDS = ("konjugation", "partizip", "partizip ii")
TARGET_TRANSLATION_LANGUAGES = {"englisch", "spanisch"}
METADATA_LABELS = {
    "worttrennung",
    "aussprache",
    "silbentrennung",
    "grammatik",
    "referenzen",
    "herkunft",
    "ipa",
    "hörbeispiele",
    "transitiv",
    "intransitiv",
    "trans.",
    "intrans.",
    "reflexiv",
    "refl.",
    "pronominal",
    "impersonal",
    "impers.",
    "unpersönlich",
}
FLEXION_KEYWORDS = (
    "präteritum",
    "partizip",
    "konjugation",
    "imperativ",
    "futur",
    "wortform",
    "person ",
    "ipa",
    "hörbeispiel",
    "hörbeispiele",
)


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


def find_verb_anchor(german_nodes: Sequence[object]) -> Optional[Tag]:
    """
    Return an anchor (heading tag) that marks the start of the Verb entry.
    Wiktionary DE often has a 'Wortart' heading with the word 'Verb' in the
    following paragraph/table. We accept either:
      - a heading with id/text 'Verb', OR
      - a heading containing 'Wortart' when the immediate block mentions 'Verb'.
    """
    direct = find_heading(german_nodes, ("Verb",))
    if direct:
        return direct

    for node in german_nodes:
        if not isinstance(node, Tag):
            continue
        heading_tag = _extract_heading_tag(node)
        if not heading_tag:
            continue
        title = heading_tag.get_text(strip=True).lower()
        if "wortart" in title:
            window: List[Tag] = []
            for sibling in iterate_section_after_heading(heading_tag):
                if isinstance(sibling, Tag):
                    if _extract_heading_tag(sibling):
                        break
                    window.append(sibling)
                if len(window) >= 8:
                    break
            window_text = " ".join(clean_text(x) for x in window if isinstance(x, Tag)).lower()
            if "verb" in window_text:
                return heading_tag

    return None


def page_title_text(soup: BeautifulSoup) -> str:
    title = soup.find("h1")
    return clean_text(title) if title else ""


def derive_for_bases(
    bases: Iterable[str],
    *,
    client: Optional[WiktionaryClient] = None,
) -> List[DerivedVerb]:
    """Collect derived verbs for a list of base verbs."""
    wiktionary = client or WiktionaryClient()
    all_results: List[DerivedVerb] = []

    for base in normalize_bases(bases):
        print(f"[{base}] processing")

        base_page = wiktionary.fetch(base)
        if not base_page:
            print(f"  warning: no Wiktionary entry found for '{base}'.")
            continue

        german_section = extract_deutsch_section(base_page.soup)
        if not german_section:
            print(f"  warning: missing 'Deutsch' section for '{base}'.")
            continue

        candidates = extract_candidate_lemmas_anywhere(german_section, base)
        print(f"  found {len(candidates)} raw candidates for '{base}'.")

        if not candidates:
            wide = extract_candidate_lemmas_anywhere(base_page.soup, base)
            print(f"  fallback(anywhere): +{len(wide)} candidates for '{base}'.")
            candidates |= wide

        seen: Set[str] = set()

        for candidate in sorted(candidates):
            prefix_info = identify_prefix(candidate, base)
            if not prefix_info:
                continue

            prefix_label, separability = prefix_info
            if separability == "ambiguous":
                continue

            if candidate in seen:
                continue

            candidate_page = wiktionary.fetch(candidate)
            if not candidate_page:
                print(f"    skip: no page for {candidate}")
                continue

            page_title = page_title_text(candidate_page.soup).lower()
            if looks_like_non_lemma_title(page_title):
                print(f"    skip: looks like inflection/non-lemma page for {candidate} -> {page_title}")
                continue

            entry = extract_verb_entry(
                base=base,
                derived=candidate,
                prefix_label=prefix_label,
                separability=separability,
                page=candidate_page,
            )
            if not entry:
                print(f"    skip: could not parse verb entry for {candidate}")
                continue

            all_results.append(entry)
            seen.add(candidate)
            gloss_preview = (entry.gloss_de or "").split(".")[0][:80]
            print(f"    ok: {candidate} ({prefix_label}, {separability}) -> {gloss_preview}")

    return all_results


def normalize_bases(bases: Iterable[str]) -> List[str]:
    """Return normalized base verbs once, in deterministic order."""
    return sorted({base.strip().lower() for base in bases if base.strip()})


def looks_like_non_lemma_title(title: str) -> bool:
    """Detect pages that are likely inflection or conjugation pages."""
    return any(keyword in title for keyword in NON_LEMMA_TITLE_KEYWORDS)


def extract_candidate_lemmas_anywhere(section: Union[BeautifulSoup, Sequence[object]], base: str) -> Set[str]:
    """Collect candidate lemmas by scanning the given German section (or whole page)."""
    candidates: Set[str] = set()

    for anchor in iter_anchor_tags(section):
        href = anchor["href"]
        lemma = normalize_lemma_from_href(href)
        if not lemma:
            continue
        lower = lemma.lower()
        if "_" in lower or len(lower) <= len(base):
            continue
        if identify_prefix(lower, base):
            candidates.add(lower)

    return candidates


def iter_anchor_tags(section: Union[BeautifulSoup, Sequence[object]]) -> Iterable[Tag]:
    """Yield links from either a soup/tree node or a sequence of section nodes."""
    if hasattr(section, "find_all"):
        yield from section.find_all("a", href=True)
        return

    for node in section:
        if isinstance(node, Tag):
            yield from node.find_all("a", href=True)


def normalize_lemma_from_href(href: str) -> Optional[str]:
    """Normalize a Wiktionary anchor href into a lemma string."""
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        parsed = urlparse(href)
        if not parsed.path.startswith("/wiki/"):
            return None
        path = parsed.path
    elif href.startswith("./"):
        path = "/wiki/" + href[2:]
    else:
        path = href

    if not path.startswith("/wiki/"):
        return None

    target = path.split("/wiki/", 1)[1]
    target = target.split("?", 1)[0]
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
        span_ids = [span.get("id", "") for span in h2.find_all("span", id=True) if span.get("id")]
        heading_text = h2.get_text(strip=True).lower()
        if (
            any(sid.lower().startswith("deutsch") or sid.lower().split("_")[0] == "deutsch" for sid in span_ids)
            or "deutsch" in heading_text
        ):
            nodes: List[object] = []
            container: Tag = h2.parent if isinstance(h2.parent, Tag) else h2
            for sibling in container.next_siblings:
                if isinstance(sibling, Tag):
                    # stop when the next language heading (wrapped in mw-heading2) starts
                    classes = sibling.get("class", [])
                    if sibling.name == "div" and classes and "mw-heading2" in classes:
                        break
                    if sibling.name == "h2":
                        break
                nodes.append(sibling)

            def flatten(items: Iterable[object]) -> Iterator[object]:
                for item in items:
                    if isinstance(item, Tag):
                        classes = item.get("class", []) or []
                        if item.name == "section":
                            yield from flatten(list(item.children))
                            continue
                        if item.name == "div" and any(cls.startswith("mw-heading") for cls in classes):
                            yield from flatten(list(item.children))
                            continue
                    yield item

            return list(flatten(nodes))

    # fallback — better to keep working than to drop everything
    return list(soup.body.children) if soup.body else list(soup.children)


def identify_prefix(lemma: str, base: str) -> Optional[Tuple[str, str]]:
    """Identify prefix label and separability for a derived lemma."""
    # Check the longest matching prefix first.
    for prefix, label, separability in PREFIX_ORDER:
        if lemma.startswith(prefix):
            remainder = lemma[len(prefix) :]
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
    """Extract relevant information from a derived verb page (robust on DE Wiktionary)."""
    german_nodes = extract_deutsch_section(page.soup)
    if not german_nodes:
        return None

    anchor = find_verb_anchor(german_nodes)
    scan_nodes = entry_scan_nodes(anchor, german_nodes)

    gloss_de = extract_first_gloss(scan_nodes)
    if not gloss_de:
        return None

    translations = extract_translations_from_nodes(scan_nodes)
    if not translations:
        translations = extract_translations_from_nodes(german_nodes)

    return DerivedVerb(
        base=base,
        derived=derived,
        prefix=prefix_label,
        separability=separability,
        pos="Verb",  # we anchored on/near Wortart: Verb (or we had good DE gloss)
        gloss_de=gloss_de,
        gloss_es=translations.get("spanisch", ""),
        gloss_en=translations.get("englisch", ""),
        example=extract_example(scan_nodes),
        wiktionary_url=page.url,
    )


def entry_scan_nodes(
    anchor: Optional[Tag],
    fallback_nodes: Sequence[object],
) -> List[object]:
    """Return the section nodes to inspect for one verb entry."""
    if anchor is None:
        return list(fallback_nodes)
    return list(iterate_section_after_heading(anchor))


def find_heading(
    section_nodes: Sequence[object],
    ids_or_titles: Tuple[str, ...],
) -> Optional[Tag]:
    """Find the first heading matching one of the provided labels."""
    normalized_targets = tuple(item.lower() for item in ids_or_titles)
    for node in section_nodes:
        if not isinstance(node, Tag):
            continue
        heading_tag = _extract_heading_tag(node)
        if not heading_tag:
            continue
        span_ids = [span.get("id", "") for span in heading_tag.find_all("span", id=True) if span.get("id")]
        for span_id in span_ids:
            base_id = span_id.split("_")[0].lower()
            if base_id in normalized_targets:
                return heading_tag
        heading_text = heading_tag.get_text(strip=True).lower()
        if heading_text in normalized_targets:
            return heading_tag
        if any(heading_text.startswith(target) for target in normalized_targets):
            return heading_tag
    return None


def extract_first_gloss(nodes: Sequence[object]) -> str:
    """Extract the first meaningful German definition from entry nodes."""
    node_list = list(nodes)
    gloss = extract_labeled_block_text(node_list, "bedeut")
    if gloss:
        return gloss

    for node in node_list:
        if not isinstance(node, Tag):
            continue
        if node.name == "ol":
            text = clean_text(node.find("li") or node)
        elif node.name == "dl":
            text = clean_text(node)
        elif node.name == "p":
            text = clean_text(node)
        else:
            continue
        if is_content_text(text):
            return text

    for node in node_list:
        if isinstance(node, Tag):
            text = clean_text(node)
            if is_content_text(text):
                return text
    return ""


def extract_labeled_block_text(nodes: Sequence[object], label: str) -> str:
    """Find the first list/paragraph after a labeled Wiktionary block."""
    for index, node in enumerate(nodes):
        if not isinstance(node, Tag):
            continue

        heading = _extract_heading_tag(node)
        if heading and label in clean_text(heading).lower():
            text = _first_text_in_block(heading)
            if text:
                return text

        if node.name == "p" and label in clean_text(node).lower():
            text = _first_text_after_marker(nodes[index + 1 :])
            if text:
                return text

    return ""


def is_content_text(text: str) -> bool:
    """Return whether text looks like entry content rather than metadata."""
    if not text:
        return False
    lowered = text.lower()
    label = lowered.rstrip(":")
    if label in METADATA_LABELS:
        return False
    if lowered.endswith(":"):
        return False
    if lowered.startswith(("ipa", "hörbeispiel", "hörbeispiele")):
        return False
    if "bearbeiten" in lowered or "[bearbeiten]" in lowered:
        return False
    return not any(keyword in lowered for keyword in FLEXION_KEYWORDS)


def extract_example(nodes: Sequence[object]) -> str:
    """Retrieve one example sentence if available."""
    example = extract_labeled_block_text(nodes, "beispiel")
    if example:
        return example
    return ""


def _first_text_in_block(heading: Tag) -> str:
    """Get the first useful list item or paragraph after a heading."""
    for node in iterate_section_after_heading(heading):
        if isinstance(node, Tag):
            if node.name in {"ul", "ol", "dl"}:
                first_item = node.find(["li", "dd"])
                text = clean_text(first_item or node)
                if text:
                    return text
            if node.name == "p":
                text = clean_text(node)
                if text:
                    return text
            if _extract_heading_tag(node):
                return ""
    return ""


def _first_text_after_marker(nodes: Sequence[object]) -> str:
    """Return the first useful text node after a paragraph marker."""
    for node in nodes:
        if not isinstance(node, Tag):
            continue
        if _extract_heading_tag(node):
            return ""
        if node.name in {"ul", "ol", "dl"}:
            first_item = node.find(["li", "dd"])
            text = clean_text(first_item or node)
            if text:
                return text
        if node.name == "p":
            text = clean_text(node)
            if text:
                return text
    return ""


def extract_translations_from_nodes(nodes: Sequence[object]) -> Dict[str, str]:
    """Extract Spanish and English translations from entry nodes."""
    node_list = list(nodes)
    collected: Dict[str, str] = {}

    for index, node in enumerate(node_list):
        if not isinstance(node, Tag):
            continue

        heading = _extract_heading_tag(node)
        if heading and "übersetz" in clean_text(heading).lower():
            merge_translations(collected, _parse_translation_tables(heading))
            if collected:
                return collected

        if node.name == "p" and "übersetz" in clean_text(node).lower():
            merge_translations(
                collected,
                _parse_translation_tables_after_marker(node_list[index + 1 :]),
            )
            if collected:
                return collected

    return collected


def merge_translations(target: Dict[str, str], source: Dict[str, str]) -> None:
    """Add translations without overwriting the first non-empty value."""
    for language, value in source.items():
        if value and language not in target:
            target[language] = value


def _parse_translation_tables(heading: Tag) -> Dict[str, str]:
    """Parse translation tables following a heading."""
    collected: Dict[str, str] = {}
    for node in iterate_section_after_heading(heading):
        if isinstance(node, Tag):
            if _extract_heading_tag(node):
                break
            merge_translations(collected, _extract_translations_from_node_tables(node))
    return collected


def _parse_translation_tables_after_marker(nodes: Sequence[object]) -> Dict[str, str]:
    """Parse translation tables after a paragraph marker such as 'Übersetzungen:'."""
    collected: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, Tag):
            continue
        if _extract_heading_tag(node):
            break
        merge_translations(collected, _extract_translations_from_node_tables(node))
    return collected


def _extract_translations_from_node_tables(node: Tag) -> Dict[str, str]:
    """Parse translation tables in a node or its descendants."""
    collected: Dict[str, str] = {}
    tables = [node] if node.name == "table" else node.find_all("table")
    for table in tables:
        merge_translations(collected, _extract_translations_from_table(table))
    return collected


def _extract_translations_from_table(table: Tag) -> Dict[str, str]:
    """Convert a Wiktionary translation table into a language map."""
    result: Dict[str, str] = {}
    for item in table.find_all("li"):
        parsed = _extract_translation_from_list_item(item)
        if parsed is None:
            continue
        language, value = parsed
        if value and language not in result:
            result[language] = value
    return result


def _extract_translation_from_list_item(item: Tag) -> Optional[Tuple[str, str]]:
    """Parse a single Wiktionary translation list item."""
    text = clean_text(item)
    if ":" not in text:
        return None

    language = text.split(":", 1)[0].strip().lower()
    if language not in TARGET_TRANSLATION_LANGUAGES:
        return None

    values = [clean_text(span) for span in item.find_all("span", lang=True) if clean_text(span)]
    if not values:
        return language, ""

    return language, ", ".join(dict.fromkeys(values))


def _extract_heading_tag(node: Tag, levels: Sequence[str] = ("h3", "h4", "h5")) -> Optional[Tag]:
    """Return the first heading tag within a container matching the desired levels."""
    if node.name in levels:
        return node

    direct = node.find(list(levels), recursive=False)
    if direct:
        return direct

    for child in node.find_all(recursive=False):
        if not isinstance(child, Tag):
            continue
        classes = child.get("class", []) or []
        if child.name in {"section", "div"} or any(cls.startswith("mw-heading") for cls in classes):
            nested = child.find(list(levels), recursive=False)
            if nested:
                return nested
    return None


def iterate_section_after_heading(heading: Tag) -> Iterator[object]:
    """Yield siblings after a heading until the next same-level heading or H2."""
    stop_tags = {"h2"}
    level = heading.name
    current_id = _heading_id(heading)
    container = heading
    if isinstance(heading.parent, Tag):
        parent_classes = heading.parent.get("class", [])
        if (
            heading.parent.name == "div"
            and parent_classes
            and any(cls.startswith("mw-heading") for cls in parent_classes)
        ):
            container = heading.parent

    for sibling in container.next_siblings:
        if isinstance(sibling, Tag):
            if sibling.name in stop_tags:
                break
            candidate = _extract_heading_tag(sibling, levels=("h2", "h3", "h4", "h5", "h6"))
            if candidate:
                candidate_name = candidate.name
                if candidate_name in stop_tags:
                    break
                if _is_same_or_higher_level(level, candidate_name):
                    candidate_id = _heading_id(candidate)
                    if candidate_id != current_id:
                        break
                    if candidate_id is None and candidate_name == heading.name:
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


def load_verbs_from_args(args: argparse.Namespace) -> List[str]:
    """Load the list of verbs based on CLI arguments."""
    source = args.verbs.strip()
    if not source:
        raise ValueError("No verbs provided. Use --verbs with a list or file path.")

    verbs_path = Path(source)
    if verbs_path.exists():
        content = verbs_path.read_text(encoding="utf-8")
        verbs = {line.strip() for line in content.splitlines() if line.strip()}
    else:
        verbs = {verb.strip() for verb in source.split(",") if verb.strip()}

    if not verbs:
        raise ValueError("No verbs provided. Use --verbs with a list or file path.")
    return sorted(verbs)


def output_paths(out_stem: Path) -> Tuple[Path, Path]:
    """Return CSV and JSON paths for one output stem."""
    stem = out_stem.with_suffix("") if out_stem.suffix in {".csv", ".json"} else out_stem
    return Path(f"{stem}.csv").resolve(), Path(f"{stem}.json").resolve()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Derive prefixed German verbs from Wiktionary.")
    parser.add_argument(
        "--verbs",
        type=str,
        help=("Comma-separated base verbs or a path to a text file with one verb per line."),
        default="verbs.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output file stem. '.csv' and '.json' are written beside it.",
        default=Path("out"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point."""
    args = parse_args(argv)
    try:
        verbs = load_verbs_from_args(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    derived = derive_for_bases(verbs)

    csv_path, json_path = output_paths(args.out)
    write_outputs(derived, csv_path, json_path)

    if derived:
        print(f"Wrote {len(derived)} rows to {csv_path} and {json_path}.")
    else:
        print(f"No derived verbs found. Wrote empty files to {csv_path} and {json_path}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

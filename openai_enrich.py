"""OpenAI enrichment for German prefixed verb rows.

This module is deliberately optional: importing it does not require the OpenAI
SDK. The SDK and dotenv support are loaded only when an ``OpenAIEnricher`` is
created and used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_VERSION = "anki-quality-v2"

SYSTEM_PROMPT = """You are a German-Spanish-English lexicographic assistant for Anki deck generation.
Your task is to enrich German prefixed verbs for language learners.
Return only valid JSON matching the requested schema.
Do not invent rare meanings unless they are common enough for learners.
Prefer common contemporary Standard German.
If the source data is poor, say so in quality_flags but still provide a useful learner-oriented entry when possible.

For Anki cloze fields, the blank must train the German verb, prefix, separated prefix, or participle. Never blank a noun, object, adjective, or unrelated word.

For example_de_with_blank:
- It must contain exactly one blank marker: ___.
- The answer must be the exact text removed from the sentence.
- The answer must be derived, the separated prefix, a conjugated verbal form, or participle_ii.
- Do not use answers like nouns or objects.

For separable verbs:
- present_3sg must be written as separated form, e.g. "bringt zurück", "fährt ab", "geht weg".
- Do not write "zurückbringt" for a separable verb.
- participle_ii should be the standard Perfekt participle, e.g. "zurückgebracht".

For inseparable verbs:
- present_3sg must be written as one word, e.g. "verbringt", "bekommt".
- participle_ii normally has no inserted "ge" after the prefix, e.g. "verbracht", "bekommen".

perfect_auxiliary must be exactly "haben" or "sein".
Return contemporary Standard German unless the word is genuinely rare or archaic.
"""

ENRICHMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "derived": {"type": "string"},
        "base": {"type": "string"},
        "senses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sense_id": {"type": "integer"},
                    "gloss_de_simple": {"type": "string"},
                    "gloss_es": {"type": "string"},
                    "gloss_en": {"type": "string"},
                    "construction": {"type": "string"},
                    "construction_es": {"type": "string"},
                    "example_de": {"type": "string"},
                    "example_es": {"type": "string"},
                    "example_en": {"type": "string"},
                    "example_de_with_blank": {"type": "string"},
                    "answer": {"type": "string"},
                    "present_example_de": {"type": "string"},
                    "perfect_example_de": {"type": "string"},
                    "register": {
                        "type": "string",
                        "enum": ["common", "formal", "colloquial", "rare", "archaic", "domain-specific", "unknown"],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["A1", "A2", "B1", "B2", "C1", "C2", "unknown"],
                    },
                    "frequency_bucket": {
                        "type": "string",
                        "enum": ["high", "medium", "low", "rare", "unknown"],
                    },
                    "is_reflexive": {"type": "boolean"},
                    "takes_accusative": {"type": ["boolean", "null"]},
                    "takes_dative": {"type": ["boolean", "null"]},
                    "fixed_preposition": {"type": "string"},
                    "present_3sg": {"type": "string"},
                    "perfect_auxiliary": {"type": "string"},
                    "participle_ii": {"type": "string"},
                    "separable_sentence_pattern": {"type": "string"},
                    "anki_hint_es": {"type": "string"},
                    "quality_flags": {"type": "array", "items": {"type": "string"}},
                    "suitable_for_anki": {"type": "boolean"},
                },
                "required": [
                    "sense_id",
                    "gloss_de_simple",
                    "gloss_es",
                    "gloss_en",
                    "construction",
                    "construction_es",
                    "example_de",
                    "example_es",
                    "example_en",
                    "example_de_with_blank",
                    "answer",
                    "present_example_de",
                    "perfect_example_de",
                    "register",
                    "difficulty",
                    "frequency_bucket",
                    "is_reflexive",
                    "takes_accusative",
                    "takes_dative",
                    "fixed_preposition",
                    "present_3sg",
                    "perfect_auxiliary",
                    "participle_ii",
                    "separable_sentence_pattern",
                    "anki_hint_es",
                    "quality_flags",
                    "suitable_for_anki",
                ],
            },
        },
    },
    "required": ["derived", "base", "senses"],
}


class OpenAIEnricher:
    """Cached OpenAI enrichment client."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        cache_path: Path = Path(".cache/openai_enrichment.jsonl"),
        refresh_cache: bool = False,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.cache_path = cache_path
        self.refresh_cache = refresh_cache
        self.client = client
        self.cache: Dict[str, Dict[str, Any]] = {} if refresh_cache else self._load_cache()

    def enrich(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return enriched payload for a row, using cache when possible."""
        key = self.cache_key(row)
        if not self.refresh_cache and key in self.cache:
            return self.cache[key]

        client = self._client()
        if client is None:
            return None

        try:
            result = self._call_structured_outputs(client, row)
        except Exception:
            try:
                result = self._call_json_mode(client, row)
            except Exception:
                return None

        if not isinstance(result, dict):
            return None
        self.cache[key] = result
        self._append_cache(key, result)
        return result

    def cache_key(self, row: Dict[str, Any]) -> str:
        """Return stable cache key for the source fields that affect enrichment."""
        relevant = {
            "prompt_version": PROMPT_VERSION,
            "base": row.get("base", ""),
            "derived": row.get("derived", ""),
            "prefix": row.get("prefix", ""),
            "separability": row.get("separability", ""),
            "gloss_de": row.get("gloss_de", ""),
            "gloss_es": row.get("gloss_es", ""),
            "gloss_en": row.get("gloss_en", ""),
            "example_de": row.get("example_de") or row.get("example", ""),
        }
        payload = json.dumps(relevant, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _client(self) -> Optional[Any]:
        if self.client is not None:
            return self.client

        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        self.client = OpenAI()
        return self.client

    def _call_structured_outputs(self, client: Any, row: Dict[str, Any]) -> Dict[str, Any]:
        """Use Responses API structured output parsing when available."""
        if hasattr(client, "responses") and hasattr(client.responses, "parse"):
            model_class = enrichment_model_class()
            if model_class is not None:
                response = client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(row)},
                    ],
                    text_format=model_class,
                )
                parsed = extract_parsed_response(response)
                if parsed is not None:
                    return model_to_dict(parsed)

        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "verb_enrichment",
                        "strict": True,
                        "schema": ENRICHMENT_SCHEMA,
                    }
                },
            )
            return parse_response_text(response)
        raise RuntimeError("Responses API unavailable")

    def _call_json_mode(self, client: Any, row: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to JSON mode through Responses or Chat Completions."""
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
                text={"format": {"type": "json_object"}},
            )
            return parse_response_text(response)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        if not self.cache_path.exists():
            return cache
        with self.cache_path.open("r", encoding="utf-8") as cache_file:
            for line in cache_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("key")
                result = record.get("result")
                if isinstance(key, str) and isinstance(result, dict):
                    cache[key] = result
        return cache

    def _append_cache(self, key: str, result: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as cache_file:
            cache_file.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")


def build_user_prompt(row: Dict[str, Any]) -> str:
    """Build the user prompt for one enrichment row."""
    return f"""Enrich this German prefixed verb.

Base verb: {row.get("base", "")}
Derived verb: {row.get("derived", "")}
Prefix: {row.get("prefix", "")}
Separability: {row.get("separability", "")}

Wiktionary German gloss:
{row.get("gloss_de", "")}

Wiktionary Spanish translation:
{row.get("gloss_es", "")}

Wiktionary English translation:
{row.get("gloss_en", "")}

Wiktionary example:
{row.get("example_de") or row.get("example", "")}

Important cloze rules:
- Never blank a noun, object, adjective, or unrelated word.
- The answer must be the exact text removed from example_de_with_blank.
- The answer must be a German verb form, separated prefix, or participle.

Return JSON with this schema:
{json.dumps(ENRICHMENT_SCHEMA, ensure_ascii=False)}
"""


def parse_response_text(response: Any) -> Dict[str, Any]:
    """Extract and parse JSON text from a Responses API object."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return json.loads(output_text)

    output = getattr(response, "output", None)
    if isinstance(output, list):
        for message in output:
            for content in getattr(message, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    return json.loads(text)
                if isinstance(content, dict) and content.get("text"):
                    return json.loads(content["text"])
    raise ValueError("No JSON text found in OpenAI response")


def extract_parsed_response(response: Any) -> Optional[Any]:
    """Extract parsed Pydantic object from Responses parse output."""
    direct = getattr(response, "output_parsed", None)
    if direct is not None:
        return direct
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for message in output:
            for content in getattr(message, "content", []) or []:
                parsed = getattr(content, "parsed", None)
                if parsed is not None:
                    return parsed
    return None


def model_to_dict(model: Any) -> Dict[str, Any]:
    """Convert Pydantic v1/v2 models or plain dicts to a dictionary."""
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True)
    if hasattr(model, "dict"):
        return model.dict(by_alias=True)
    raise TypeError("Unsupported parsed response object")


def enrichment_model_class() -> Optional[Any]:
    """Create Pydantic models lazily so pydantic remains optional."""
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        return None

    class EnrichedSense(BaseModel):
        sense_id: int
        gloss_de_simple: str
        gloss_es: str
        gloss_en: str
        construction: str
        construction_es: str
        example_de: str
        example_es: str
        example_en: str
        example_de_with_blank: str
        answer: str
        present_example_de: str
        perfect_example_de: str
        register_: str = Field(alias="register")
        difficulty: str
        frequency_bucket: str
        is_reflexive: bool
        takes_accusative: Optional[bool]
        takes_dative: Optional[bool]
        fixed_preposition: str
        present_3sg: str
        perfect_auxiliary: str
        participle_ii: str
        separable_sentence_pattern: str
        anki_hint_es: str
        quality_flags: List[str]
        suitable_for_anki: bool

    class EnrichmentPayload(BaseModel):
        derived: str
        base: str
        senses: List[EnrichedSense]

    return EnrichmentPayload

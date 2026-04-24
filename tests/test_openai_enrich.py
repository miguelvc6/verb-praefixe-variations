import json

from openai_enrich import OpenAIEnricher


class FakeResponse:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def sample_payload():
    return {
        "base": "kommen",
        "derived": "ankommen",
        "senses": [
            {
                "sense_id": 1,
                "gloss_de_simple": "ankommen",
                "gloss_es": "llegar",
                "gloss_en": "arrive",
                "construction": "irgendwo ankommen",
                "construction_es": "llegar a algún lugar",
                "example_de": "Der Zug ist angekommen.",
                "example_es": "El tren llegó.",
                "example_en": "The train arrived.",
                "example_de_with_blank": "Der Zug ist ___.",
                "answer": "angekommen",
                "register": "common",
                "difficulty": "A1",
                "frequency_bucket": "high",
                "is_reflexive": False,
                "takes_accusative": None,
                "takes_dative": None,
                "fixed_preposition": "",
                "present_3sg": "Er kommt an.",
                "perfect_auxiliary": "ist",
                "participle_ii": "angekommen",
                "separable_sentence_pattern": "Er kommt an.",
                "anki_hint_es": "llegar",
                "quality_flags": [],
                "suitable_for_anki": True,
            }
        ],
    }


def test_enrich_writes_and_reads_cache(tmp_path):
    client = FakeClient(sample_payload())
    cache_path = tmp_path / "cache.jsonl"
    row = {"base": "kommen", "derived": "ankommen", "prefix": "an-", "separability": "separable"}

    enricher = OpenAIEnricher(cache_path=cache_path, client=client)
    assert enricher.enrich(row)["senses"][0]["gloss_es"] == "llegar"
    assert client.responses.calls == 1

    cached = OpenAIEnricher(cache_path=cache_path, client=client)
    assert cached.enrich(row)["senses"][0]["gloss_es"] == "llegar"
    assert client.responses.calls == 1


def test_refresh_cache_calls_again(tmp_path):
    client = FakeClient(sample_payload())
    row = {"base": "kommen", "derived": "ankommen"}
    OpenAIEnricher(cache_path=tmp_path / "cache.jsonl", client=client).enrich(row)
    OpenAIEnricher(cache_path=tmp_path / "cache.jsonl", client=client, refresh_cache=True).enrich(row)
    assert client.responses.calls == 2


def test_missing_api_key_without_client_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    enricher = OpenAIEnricher(cache_path=tmp_path / "cache.jsonl")
    assert enricher.enrich({"base": "kommen", "derived": "ankommen"}) is None

import json

from openai_enrich import PROMPT_VERSION, OpenAIEnricher, build_user_prompt, parse_response_text


def test_build_user_prompt_includes_verbal_answer_rules():
    prompt = build_user_prompt({"base": "bringen", "derived": "zurückbringen"})
    assert "answer must be the exact text removed" in prompt
    assert "Never blank a noun" in prompt


def test_cache_key_changes_with_prompt_version(monkeypatch, tmp_path):
    enricher = OpenAIEnricher(cache_path=tmp_path / "cache.jsonl", client=object())
    key_v1 = enricher.cache_key({"base": "bringen", "derived": "zurückbringen"})
    monkeypatch.setattr("openai_enrich.PROMPT_VERSION", PROMPT_VERSION + "-changed")
    key_v2 = enricher.cache_key({"base": "bringen", "derived": "zurückbringen"})
    assert key_v1 != key_v2


def test_parse_response_text_handles_mock_response():
    class Response:
        output_text = json.dumps({"base": "bringen", "derived": "zurückbringen", "senses": []})

    assert parse_response_text(Response())["derived"] == "zurückbringen"

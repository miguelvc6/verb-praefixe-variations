import csv
import json

from derive_verbs import (
    DerivedVerb,
    VerbSense,
    detect_quality_flags,
    flatten_derived_verbs,
    identify_prefix,
    normalize_example,
    normalize_lemma_from_href,
    write_outputs,
)


def test_identify_prefix_excludes_ge_by_default():
    assert identify_prefix("ankommen", "kommen") == ("an-", "separable")
    assert identify_prefix("bekommen", "kommen") == ("be-", "inseparable")
    assert identify_prefix("gekommen", "kommen") is None
    assert identify_prefix("gekommen", "kommen", include_ge_prefix=True) == ("ge-", "inseparable")


def test_normalize_lemma_from_href():
    assert normalize_lemma_from_href("/wiki/ankommen#Deutsch") == "ankommen"
    assert normalize_lemma_from_href("./zur%C3%BCckgehen") == "zurückgehen"
    assert normalize_lemma_from_href("/wiki/Hilfe:Inhalt") is None


def test_detect_quality_flags_for_placeholder_and_participle():
    flags = detect_quality_flags(
        {
            "base": "kommen",
            "derived": "gekommen",
            "prefix": "ge-",
            "gloss_de": "Dieser Abschnitt fehlt noch.",
            "gloss_es": "",
            "gloss_en": "",
            "example_de": "",
        }
    )
    assert "placeholder_gloss" in flags
    assert "missing_example_de" in flags
    assert "suspected_participle" in flags


def test_normalize_example_strips_quotes_and_refs():
    assert normalize_example(' "Ich komme morgen an." [1] ') == "Ich komme morgen an."


def test_flatten_and_write_outputs_are_backward_compatible(tmp_path):
    verb = DerivedVerb(
        base="kommen",
        derived="ankommen",
        prefix="an-",
        separability="separable",
        pos="Verb",
        wiktionary_url="https://example.test",
        source="wiktionary",
        senses=[
            VerbSense(
                sense_id=1,
                gloss_de="eintreffen",
                gloss_es="llegar",
                gloss_en="arrive",
                example_de="Der Zug ist angekommen.",
                is_quality_ok=True,
                quality_flags=[],
            )
        ],
    )
    rows = flatten_derived_verbs([verb])
    assert rows[0]["example"] == "Der Zug ist angekommen."
    assert rows[0]["example_de"] == "Der Zug ist angekommen."

    csv_path = tmp_path / "out.csv"
    json_path = tmp_path / "out.json"
    write_outputs([verb], csv_path, json_path)

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert row["example"] == "Der Zug ist angekommen."

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["senses"][0]["gloss_es"] == "llegar"

from pathlib import Path

from build_anki_deck import (
    generate_cards,
    load_input,
    make_context_blank,
    normalize_flat_row,
    validate_cards,
    write_csv_export,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sample_verbs.json"


def test_load_input_normalizes_nested_json():
    senses = load_input(FIXTURE)
    assert len(senses) == 6
    assert senses[0].derived == "ankommen"
    assert senses[-1].is_quality_ok is False


def test_old_schema_normalization_and_context_blank():
    sense = normalize_flat_row(
        {
            "base": "kommen",
            "derived": "ankommen",
            "prefix": "an-",
            "separability": "separable",
            "gloss_de": "eintreffen",
            "gloss_es": "llegar",
            "gloss_en": "arrive",
            "example": "Der Zug ist angekommen.",
            "wiktionary_url": "https://example.test",
        }
    )
    assert sense.example_de == "Der Zug ist angekommen."
    assert make_context_blank(sense) == ("Der Zug ist ___.", "angekommen")


def test_generate_cards_skips_low_quality_and_adds_contrast():
    senses = load_input(FIXTURE)
    cards = generate_cards(senses)
    assert not validate_cards(cards)
    assert any(card.cardtype == "context" for card in cards)
    assert any(card.cardtype == "construction" for card in cards)
    assert any(card.cardtype == "contrast" and "Familia: stellen" in card.front for card in cards)
    assert all("aussetzen" not in card.key for card in cards)


def test_csv_export_writes_grouped_files(tmp_path):
    cards = generate_cards(load_input(FIXTURE))
    write_csv_export(cards, tmp_path)
    assert (tmp_path / "anki_cards_context.csv").exists()
    assert (tmp_path / "anki_cards_contrast.csv").exists()

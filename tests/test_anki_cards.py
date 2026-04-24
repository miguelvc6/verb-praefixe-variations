from build_anki_deck import (
    AnkiVerbSense,
    build_perfect_phrase,
    generate_cards,
    generate_contrast_cards,
    generate_context_cards,
    generate_construction_cards,
    generate_prefix_cloze_cards,
    generate_translation_de_to_es_cards,
    generate_translation_es_to_de_cards,
)


def sense(**overrides):
    data = {
        "base": "bringen",
        "derived": "zurückbringen",
        "sense_id": 1,
        "prefix": "zurück-",
        "separability": "separable",
        "gloss_de": "wiederbringen",
        "gloss_es": "devolver",
        "gloss_en": "to bring back",
        "example_de": "Wilhelm bringt die geliehenen Fahrzeuge zurück.",
        "example_de_with_blank": "Wilhelm bringt die geliehenen Fahrzeuge ___.",
        "answer": "zurück",
        "construction": "etwas zurückbringen",
        "present_3sg": "bringt zurück",
        "perfect_auxiliary": "haben",
        "participle_ii": "zurückgebracht",
        "takes_accusative": True,
        "is_quality_ok": True,
    }
    data.update(overrides)
    return AnkiVerbSense(**data)


def test_build_perfect_phrase_conjugates_auxiliary():
    assert build_perfect_phrase(sense()) == "Er hat es zurückgebracht."
    assert (
        build_perfect_phrase(
            sense(
                derived="fortlaufen",
                prefix="fort-",
                base="laufen",
                perfect_auxiliary="sein",
                participle_ii="fortgelaufen",
                takes_accusative=False,
            )
        )
        == "Er ist fortgelaufen."
    )


def test_prefix_cloze_handles_separated_prefix():
    cards = generate_prefix_cloze_cards(
        sense(
            base="fahren",
            derived="abfahren",
            prefix="ab-",
            example_de="Der Zug fährt am Hauptbahnhof ab.",
            example_de_with_blank="",
            answer="ab",
            present_3sg="fährt ab",
            participle_ii="abgefahren",
        )
    )
    assert cards
    assert "Der Zug fährt am Hauptbahnhof ___." in cards[0].front
    assert cards[0].answer == "ab"


def test_context_rejects_noun_answer():
    cards = generate_context_cards(
        sense(
            derived="vorbringen",
            prefix="vor-",
            example_de="So bringt ihr eure Zeichnungen vor.",
            example_de_with_blank="So bringt ihr eure _____ vor.",
            answer="Zeichnungen",
            construction="etwas vorbringen",
            present_3sg="bringt vor",
            participle_ii="vorgebracht",
        )
    )
    assert cards == []


def test_context_accepts_prefix_answer():
    cards = generate_context_cards(sense())
    assert len(cards) == 1
    assert cards[0].answer == "zurück"


def test_context_cloze_front_has_no_hint():
    cards = generate_context_cards(sense(anki_hint_es="devolver algo", gloss_es="devolver"))
    assert cards
    assert "Pista:" not in cards[0].front
    assert "devolver" not in cards[0].front


def test_prefix_cloze_front_has_no_base():
    cards = generate_prefix_cloze_cards(
        sense(
            base="stellen",
            derived="abstellen",
            prefix="ab-",
            example_de="Sie stellt die Tasche im Flur ab.",
            example_de_with_blank="",
            answer="ab",
            present_3sg="stellt ab",
            participle_ii="abgestellt",
        )
    )
    assert cards
    assert "Base:" not in cards[0].front
    assert "stellen" not in cards[0].front


def test_translation_de_to_es_exists():
    cards = generate_translation_de_to_es_cards(
        sense(
            derived="zurücksetzen",
            prefix="zurück-",
            gloss_es="devolver / restablecer algo",
            example_de="Er hat den Stuhl an seinen Platz zurückgesetzt.",
            example_es="Él devolvió la silla a su lugar.",
            construction="zurücksetzen + Akkusativobjekt",
            participle_ii="zurückgesetzt",
        )
    )
    assert cards
    assert "zurücksetzen" in cards[0].front
    assert "devolver" in cards[0].back
    assert "Er hat den Stuhl" in cards[0].back


def test_translation_cards_do_not_duplicate_or_show_redundant_fields():
    de_to_es = generate_translation_de_to_es_cards(
        sense(
            derived="zusammenstehen",
            gloss_es="estar juntos o unidos",
            construction="zusammenstehen [Subjekt]",
            construction_es="zusammenstehen [sujeto]",
            wiktionary_url="https://de.wiktionary.org/wiki/zusammenstehen",
        )
    )[0]
    assert de_to_es.back.count("estar juntos o unidos") == 1
    assert "Construcción ES:" not in de_to_es.back
    assert "Separabilidad:" not in de_to_es.back
    assert "Fuente:" not in de_to_es.back

    es_to_de = generate_translation_es_to_de_cards(
        sense(
            derived="zusammenstehen",
            gloss_es="estar juntos o unidos",
            construction="zusammenstehen [Subjekt]",
            construction_es="zusammenstehen [sujeto]",
            wiktionary_url="https://de.wiktionary.org/wiki/zusammenstehen",
        )
    )[0]
    assert "Significado:" in es_to_de.back
    assert "Construcción ES:" not in es_to_de.back
    assert "Separabilidad:" not in es_to_de.back
    assert "Fuente:" not in es_to_de.back


def test_translation_es_to_de_skips_ambiguous_bare_prompt():
    assert generate_translation_es_to_de_cards(sense(gloss_es="traer", construction_es="", anki_hint_es="")) == []


def test_translation_es_to_de_allows_specific_prompt():
    cards = generate_translation_es_to_de_cards(sense(gloss_es="devolver algo a su posición anterior"))
    assert cards
    assert "devolver algo" in cards[0].front


def test_translation_es_to_de_skips_prompt_that_leaks_german_answer():
    cards = generate_translation_es_to_de_cards(
        sense(
            derived="zusammenstellen",
            construction_es="zusammenstellen + objeto en acusativo",
            gloss_es="componer o colocar algo en un lugar común",
        )
    )
    assert cards
    assert "zusammenstellen" not in cards[0].front
    assert "componer" in cards[0].front


def test_translation_de_to_es_does_not_repeat_lemma_as_construction():
    cards = generate_translation_de_to_es_cards(
        sense(
            derived="zusammenstehen",
            construction="zusammenstehen [Subjekt]",
            gloss_es="estar juntos o unidos",
        )
    )
    assert cards
    assert cards[0].front.count("zusammenstehen") == 1


def test_construction_rejects_generic_construction():
    cards = generate_construction_cards(sense(construction="Subject + verb stem + object"))
    assert cards == []


def test_construction_not_in_default_but_optional():
    assert not any(card.cardtype == "construction" for card in generate_cards([sense()]))
    optional = generate_cards([sense(takes_dative=True, construction="jemandem etwas zurückbringen")], card_types=["construction"])
    assert any(card.cardtype == "construction" for card in optional)


def test_contrast_cards_are_contextual():
    group = [
        sense(),
        sense(
            derived="mitbringen",
            prefix="mit-",
            gloss_es="traer consigo",
            example_de="Kannst du bitte Brot vom Supermarkt mitbringen?",
            example_de_with_blank="Kannst du bitte Brot vom Supermarkt ___?",
            answer="mitbringen",
            construction="etwas mitbringen",
            present_3sg="bringt mit",
            participle_ii="mitgebracht",
        ),
        sense(
            derived="herbringen",
            prefix="her-",
            gloss_es="traer aquí",
            example_de="Kannst du den Stuhl herbringen?",
            example_de_with_blank="Kannst du den Stuhl ___?",
            answer="herbringen",
            construction="etwas herbringen",
            present_3sg="bringt her",
            participle_ii="hergebracht",
        ),
    ]
    cards = generate_contrast_cards(group)
    assert cards
    assert "___" in cards[0].front
    assert cards[0].answer in cards[0].front
    assert cards[0].answer in cards[0].back

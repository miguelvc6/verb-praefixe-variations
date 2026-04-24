from build_anki_deck import (
    AnkiVerbSense,
    build_perfect_phrase,
    generate_contrast_cards,
    generate_context_cards,
    generate_construction_cards,
    generate_prefix_cloze_cards,
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


def test_construction_rejects_generic_construction():
    cards = generate_construction_cards(sense(construction="Subject + verb stem + object"))
    assert cards == []


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

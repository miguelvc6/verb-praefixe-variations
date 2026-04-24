from derive_verbs import DerivedVerb, VerbSense, detect_quality_flags, is_quality_ok, needs_openai_enrichment


def test_incomplete_wiktionary_row_is_not_quality_ok():
    row = {
        "base": "bringen",
        "derived": "abbringen",
        "prefix": "ab-",
        "separability": "separable",
        "gloss_de": "transitiv :",
        "gloss_es": "apartar",
        "gloss_en": "remove",
        "example_de": "Bringst du den Fleck hier ab?",
        "example_de_with_blank": "",
        "answer": "",
        "construction": "",
        "present_3sg": "",
        "perfect_auxiliary": "",
        "participle_ii": "",
    }
    flags = detect_quality_flags(row)
    assert "metadata_gloss" in flags
    assert "missing_example_de_with_blank" in flags
    assert "missing_answer" in flags
    assert "missing_construction" in flags
    assert "missing_present_3sg" in flags
    assert "missing_perfect_auxiliary" in flags
    assert "missing_participle_ii" in flags
    assert is_quality_ok(flags) is False

    sense = VerbSense(
        gloss_de=row["gloss_de"],
        gloss_es=row["gloss_es"],
        gloss_en=row["gloss_en"],
        example_de=row["example_de"],
        quality_flags=flags,
    )
    verb = DerivedVerb("bringen", "abbringen", "ab-", "separable", "Verb", [sense], "https://example.test")
    assert needs_openai_enrichment(verb) is True


def test_noun_answer_is_invalid():
    flags = detect_quality_flags(
        {
            "base": "bringen",
            "derived": "vorbringen",
            "prefix": "vor-",
            "separability": "separable",
            "gloss_de": "etwas äußern",
            "gloss_es": "exponer",
            "gloss_en": "to present",
            "example_de": "So bringt ihr eure Zeichnungen vor.",
            "example_de_with_blank": "So bringt ihr eure _____ vor.",
            "answer": "Zeichnungen",
            "construction": "etwas vorbringen",
            "present_3sg": "bringt vor",
            "perfect_auxiliary": "haben",
            "participle_ii": "vorgebracht",
        }
    )
    assert "invalid_context_answer" in flags
    assert is_quality_ok(flags) is False


def test_separable_present_cannot_be_one_word():
    flags = detect_quality_flags(
        {
            "base": "bringen",
            "derived": "zurückbringen",
            "prefix": "zurück-",
            "separability": "separable",
            "gloss_de": "wiederbringen",
            "gloss_es": "devolver",
            "gloss_en": "to bring back",
            "example_de": "Er bringt das Buch zurück.",
            "example_de_with_blank": "Er bringt das Buch ___.",
            "answer": "zurück",
            "construction": "etwas zurückbringen",
            "present_3sg": "zurückbringt",
            "perfect_auxiliary": "haben",
            "participle_ii": "zurückgebracht",
        }
    )
    assert "invalid_separable_present_3sg" in flags


def test_perfect_auxiliary_is_normalized_in_ingestion():
    flags = detect_quality_flags(
        {
            "base": "bringen",
            "derived": "zurückbringen",
            "prefix": "zurück-",
            "separability": "separable",
            "gloss_de": "wiederbringen",
            "gloss_es": "devolver",
            "gloss_en": "to bring back",
            "example_de": "Er bringt das Buch zurück.",
            "example_de_with_blank": "Er bringt das Buch ___.",
            "answer": "zurück",
            "construction": "etwas zurückbringen",
            "present_3sg": "bringt zurück",
            "perfect_auxiliary": "hat",
            "participle_ii": "zurückgebracht",
        }
    )
    assert "invalid_perfect_auxiliary" not in flags

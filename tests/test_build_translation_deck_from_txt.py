from build_translation_deck_from_txt import find_base_verb, parse_translation_blocks


def test_parse_translation_blocks_skips_intro_and_builds_family_tags():
    text = """Aquí tienes traducciones suficientemente discriminadoras para Anki:

ablegen
quitarse una prenda o accesorio
depositar / dejar algo en un lugar

zusammenziehen
contraer / encoger
mudarse juntos
"""

    notes = parse_translation_blocks(text, ["legen", "ziehen"])

    assert [note.german for note in notes] == ["ablegen", "zusammenziehen"]
    assert notes[0].spanish == ("quitarse una prenda o accesorio", "depositar / dejar algo en un lugar")
    assert "family::legen" in notes[0].tags
    assert "family::ziehen" in notes[1].tags


def test_find_base_verb_prefers_longest_suffix():
    assert find_base_verb("anstehen", ["gehen", "stehen"]) == "stehen"
    assert find_base_verb("ablegen", ["legen", "geben"]) == "legen"

#!/usr/bin/env python3
"""Build Anki cards for German verb-prefix semantic intuitions.

This script generates "Type D" cards: prefix as semantic clue.

It can read an existing out.json/out.csv produced by the prefixed-verb pipeline,
extract good example verbs per prefix, and export either:

- a CSV file that can be imported into Anki, or
- an .apkg deck if genanki is installed.

The generated cards are intentionally *not* hard rules. German prefixes are
polysemous and often lexicalized. The cards teach useful semantic tendencies.

Example usage:

    python build_prefix_semantics_cards.py --input out.json --out prefix_semantics.csv

    python build_prefix_semantics_cards.py --input out.json --format apkg --out prefix_semantics.apkg

    python build_prefix_semantics_cards.py --input out.json --include-reverse --out prefix_semantics.csv

If no input file is provided, the script still generates cards from the built-in
prefix semantic table, using built-in examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PrefixInfo:
    prefix: str
    separability: str
    intuition_es: str
    patterns_es: List[str]
    caveat_es: str
    fallback_examples: List[tuple[str, str]] = field(default_factory=list)
    priority: int = 100


@dataclass
class Example:
    derived: str
    gloss_es: str = ""
    example_de: str = ""
    construction: str = ""
    source: str = ""


@dataclass
class Card:
    cardtype: str
    prefix: str
    front: str
    back: str
    tags: List[str]


PREFIXES: Dict[str, PrefixInfo] = {
    "ab-": PrefixInfo(
        "ab-", "separable", "alejamiento, separación, retirada, reducción o finalización",
        ["mover algo lejos de un punto de referencia", "quitar o separar algo", "interrumpir, apagar o terminar una acción"],
        "No siempre significa simplemente 'hacia abajo'; muchas veces indica separación o desconexión.",
        [("abfahren", "salir / partir"), ("abstellen", "dejar / colocar / apagar"), ("abbringen", "apartar / disuadir")], 10,
    ),
    "an-": PrefixInfo(
        "an-", "separable", "contacto, aproximación, inicio o fijación a una superficie",
        ["acercarse o llegar a un punto", "poner algo en contacto con otra cosa", "empezar una acción"],
        "Es uno de los prefijos más polisémicos; úsalo como intuición, no como regla exacta.",
        [("ankommen", "llegar"), ("anfangen", "empezar"), ("anbringen", "colocar / fijar")], 10,
    ),
    "auf-": PrefixInfo(
        "auf-", "separable", "apertura, subida, aparición, activación o acumulación",
        ["abrir o poner en estado abierto", "levantar o subir", "empezar a funcionar o activarse", "reunir/acumular algo"],
        "Puede ser espacial, aspectual o idiomático; no todas las acepciones son transparentes.",
        [("aufstehen", "levantarse"), ("aufmachen", "abrir"), ("aufbringen", "reunir / enojar")], 10,
    ),
    "aus-": PrefixInfo(
        "aus-", "separable", "salida, extracción, agotamiento, apagado o realización completa",
        ["sacar algo hacia fuera", "apagar o dejar de funcionar", "hacer algo hasta el final", "distribuir o extender hacia fuera"],
        "La idea de 'fuera' se extiende metafóricamente a apagar, acabar o completar.",
        [("aussteigen", "bajarse / salir de un vehículo"), ("ausmachen", "apagar / acordar"), ("ausbringen", "esparcir / distribuir")], 10,
    ),
    "bei-": PrefixInfo(
        "bei-", "separable", "adición, acompañamiento, cercanía o aportar algo a alguien",
        ["añadir algo a una situación", "acompañar o estar junto a", "aportar/proporcionar algo"],
        "En muchos verbos modernos el significado está lexicalizado y debe aprenderse con ejemplos.",
        [("beibringen", "enseñar / aportar"), ("beistehen", "asistir / apoyar"), ("beitragen", "contribuir")], 20,
    ),
    "ein-": PrefixInfo(
        "ein-", "separable", "entrada, inserción, incorporación, encierro o ajuste hacia dentro",
        ["meter o introducir algo", "incorporar algo a un sistema", "ajustar/configurar algo", "empezar un proceso desde fuera hacia dentro"],
        "No equivale siempre a 'in'; en verbos técnicos puede significar configurar o presentar.",
        [("einsteigen", "subir / entrar"), ("einstellen", "ajustar / contratar / cesar"), ("einbringen", "introducir / aportar")], 10,
    ),
    "fest-": PrefixInfo(
        "fest-", "separable", "fijación, firmeza, determinación o constatación",
        ["hacer que algo quede fijo", "establecer algo de forma definitiva", "constatar o determinar un hecho"],
        "A menudo pasa de una idea física de 'fijo' a una idea abstracta de 'determinado'.",
        [("feststellen", "constatar / determinar"), ("festlegen", "fijar / establecer"), ("festhalten", "sujetar / retener / dejar constancia")], 10,
    ),
    "fort-": PrefixInfo(
        "fort-", "separable", "continuación o alejamiento progresivo",
        ["seguir haciendo algo", "llevar algo lejos", "irse o alejarse"],
        "Puede significar tanto 'continuar' como 'alejarse'; el verbo concreto decide.",
        [("fortsetzen", "continuar"), ("fortfahren", "continuar / proseguir"), ("fortbringen", "llevarse / retirar")], 30,
    ),
    "her-": PrefixInfo(
        "her-", "separable", "movimiento hacia el hablante o hacia el punto de referencia",
        ["traer algo hacia aquí", "venir desde allí hacia aquí", "orientar el movimiento hacia el hablante"],
        "Contrasta con hin-, que suele apuntar alejándose del hablante.",
        [("herbringen", "traer hacia aquí"), ("herkommen", "venir de / venir hacia aquí"), ("herstellen", "producir / restablecer")], 10,
    ),
    "hin-": PrefixInfo(
        "hin-", "separable", "movimiento hacia allí, alejándose del hablante",
        ["llevar algo hacia otro lugar", "dirigir la acción hacia un punto externo", "poner algo en un lugar concreto"],
        "Contrasta con her-: hin- suele ser 'hacia allí', her- suele ser 'hacia aquí'.",
        [("hinbringen", "llevar allí"), ("hinstellen", "poner allí"), ("hingehen", "ir allí")], 10,
    ),
    "los-": PrefixInfo(
        "los-", "separable", "inicio repentino, soltarse o ponerse en marcha",
        ["empezar de repente", "soltar o desprender", "ponerse en movimiento"],
        "Muchas veces expresa comienzo brusco o liberación, pero no siempre es composicional.",
        [("losgehen", "empezar / ponerse en marcha"), ("loslassen", "soltar"), ("losfahren", "arrancar / salir")], 30,
    ),
    "mit-": PrefixInfo(
        "mit-", "separable", "acompañamiento, participación o llevar algo consigo",
        ["hacer algo junto con otros", "llevar algo contigo", "participar en una acción"],
        "Suele ser bastante transparente, pero algunos verbos tienen usos lexicalizados.",
        [("mitkommen", "venir con alguien"), ("mitbringen", "traer consigo"), ("mitmachen", "participar")], 10,
    ),
    "nach-": PrefixInfo(
        "nach-", "separable", "posterioridad, seguimiento, imitación o llevar algo después",
        ["hacer algo después", "seguir a alguien o algo", "imitar o reproducir", "entregar algo que quedó atrás"],
        "No siempre significa 'después'; también puede indicar seguimiento o imitación.",
        [("nachbringen", "traer después algo olvidado"), ("nachmachen", "imitar"), ("nachdenken", "reflexionar")], 20,
    ),
    "vor-": PrefixInfo(
        "vor-", "separable", "delante, anticipación, presentación o exposición",
        ["poner o presentar algo delante", "hacer algo por adelantado", "exponer una idea, argumento o actuación"],
        "La idea de 'delante' puede ser espacial, temporal o discursiva.",
        [("vorstellen", "presentar / imaginar"), ("vorbringen", "exponer / presentar un argumento"), ("vorkommen", "ocurrir / aparecer")], 10,
    ),
    "weg-": PrefixInfo(
        "weg-", "separable", "alejar, retirar, quitar o llevarse",
        ["mover algo lejos del lugar actual", "retirar o eliminar algo", "irse de un sitio"],
        "Muy cercano a ab- y fort- en algunos contextos; se aprende bien por contraste.",
        [("wegbringen", "llevarse / retirar"), ("weggehen", "irse"), ("wegnehmen", "quitar")], 10,
    ),
    "weiter-": PrefixInfo(
        "weiter-", "separable", "continuación, avance o transmisión hacia adelante",
        ["seguir haciendo algo", "hacer progresar algo", "pasar/transmitir algo a otra persona"],
        "Suele ser bastante transparente: weiter ≈ más adelante / continuar.",
        [("weiterbringen", "hacer avanzar"), ("weitergehen", "continuar"), ("weitergeben", "transmitir / pasar")], 20,
    ),
    "zu-": PrefixInfo(
        "zu-", "separable", "cierre, dirección hacia algo, adición o asignación",
        ["cerrar algo", "dirigir algo hacia un destino", "añadir o asignar"],
        "Es polisémico: zu- puede expresar cierre, dirección o adición según el verbo.",
        [("zumachen", "cerrar"), ("zulegen", "aumentar / adquirir"), ("zubringen", "pasar tiempo / llevar hacia")], 20,
    ),
    "zurück-": PrefixInfo(
        "zurück-", "separable", "retorno, devolución o vuelta a un estado anterior",
        ["llevar algo de vuelta", "volver a un lugar o estado anterior", "devolver o restablecer algo"],
        "Suele ser transparente, pero el verbo concreto decide si es volver, devolver o restablecer.",
        [("zurückbringen", "traer/llevar de vuelta"), ("zurückkommen", "volver"), ("zurücksetzen", "restablecer / devolver a posición anterior")], 10,
    ),
    "zusammen-": PrefixInfo(
        "zusammen-", "separable", "unión, reunión, cooperación o colapso hacia un conjunto",
        ["juntar varias cosas o personas", "hacer algo en común", "resumir o comprimir", "derrumbarse/colapsar en algunos verbos"],
        "La idea básica es 'junto', pero puede especializarse mucho según el verbo.",
        [("zusammenbringen", "juntar / unir"), ("zusammenfassen", "resumir"), ("zusammenarbeiten", "colaborar")], 10,
    ),
    "be-": PrefixInfo(
        "be-", "inseparable", "hacer transitivo el verbo, dirigir la acción hacia un objeto o afectar algo completamente",
        ["convertir una acción en algo que recae sobre un objeto", "cubrir o tratar algo con la acción", "hacer que el verbo exija acusativo con más frecuencia"],
        "be- es muy abstracto y productivo; muchas veces cambia la valencia más que aportar un significado espacial.",
        [("bekommen", "recibir"), ("bestellen", "pedir / encargar"), ("belegen", "ocupar / demostrar / cubrir")], 10,
    ),
    "emp-": PrefixInfo(
        "emp-", "inseparable", "recepción, percepción o inicio de un proceso interno",
        ["recibir o percibir algo", "sentir o experimentar algo", "empezar a desarrollarse en algunos verbos"],
        "Es poco productivo en el alemán moderno; normalmente se memoriza verbo por verbo.",
        [("empfangen", "recibir"), ("empfinden", "sentir / percibir"), ("empfehlen", "recomendar")], 50,
    ),
    "ent-": PrefixInfo(
        "ent-", "inseparable", "alejamiento, extracción, privación, desarrollo o reversión",
        ["quitar o retirar algo", "escapar de algo", "desarrollarse o surgir", "hacer lo contrario de una acción en ciertos verbos"],
        "Puede significar tanto 'quitar' como 'surgir/desarrollarse'; conviene aprenderlo con ejemplos.",
        [("entkommen", "escapar"), ("entstehen", "surgir / originarse"), ("entfernen", "alejar / quitar")], 10,
    ),
    "er-": PrefixInfo(
        "er-", "inseparable", "resultado conseguido, cambio de estado, obtención o culminación",
        ["alcanzar un resultado", "hacer que algo llegue a existir", "lograr u obtener algo mediante la acción"],
        "Muy abstracto y frecuentemente lexicalizado; no intentes predecir todos los significados solo por er-.",
        [("erreichen", "alcanzar"), ("erklären", "explicar"), ("erbringen", "producir / aportar / rendir")], 10,
    ),
    "ge-": PrefixInfo(
        "ge-", "inseparable", "prefijo poco productivo en verbos simples; aparece sobre todo en participios y algunos verbos lexicalizados",
        ["marca muchos participios II: gemacht, gebracht, gestellt", "aparece en algunos verbos lexicalizados: gefallen, gelingen, gehören"],
        "No lo trates como un prefijo semántico regular para derivar nuevos verbos; su papel principal para estudiantes es morfológico.",
        [("gefallen", "gustar"), ("gelingen", "salir bien / lograrse"), ("gehören", "pertenecer")], 80,
    ),
    "miss-": PrefixInfo(
        "miss-", "inseparable", "error, fracaso, mala ejecución o valoración negativa",
        ["hacer algo mal", "fallar en una acción", "indicar una desviación negativa"],
        "No todos los verbos con miss- son frecuentes; muchos son formales o menos comunes.",
        [("missverstehen", "malentender"), ("misslingen", "fracasar / salir mal"), ("missbrauchen", "abusar / usar mal")], 30,
    ),
    "ver-": PrefixInfo(
        "ver-", "inseparable", "cambio de estado, error, consumo, dispersión o intensificación",
        ["transformar algo en otro estado", "hacer algo mal o equivocarse", "gastar/consumir algo", "alejar o dispersar en algunos verbos"],
        "ver- es uno de los prefijos más difíciles: muchas acepciones son lexicalizadas.",
        [("verstehen", "entender"), ("verbringen", "pasar tiempo"), ("verlegen", "trasladar / extraviar")], 10,
    ),
    "zer-": PrefixInfo(
        "zer-", "inseparable", "destrucción, fragmentación o separación en partes",
        ["romper algo en pedazos", "deshacer una unidad", "destruir la integridad de algo"],
        "Suele ser más transparente que ver-: zer- apunta a ruptura o desintegración.",
        [("zerbrechen", "romperse / romper en pedazos"), ("zerstören", "destruir"), ("zerlegen", "desmontar / descomponer")], 10,
    ),
    "durch-": PrefixInfo(
        "durch-", "ambiguous", "atravesar, completar de principio a fin o penetrar algo",
        ["pasar a través de algo", "hacer algo completamente", "revisar o recorrer algo de principio a fin"],
        "Puede ser separable o inseparable con cambios de acento y significado: durchfahren ≠ durchfahren en todos los usos.",
        [("durchgehen", "pasar por / revisar"), ("durchführen", "realizar / llevar a cabo"), ("durchbrechen", "atravesar / romper")], 20,
    ),
    "hinter-": PrefixInfo(
        "hinter-", "ambiguous", "detrás, dejar atrás, ocultar o respaldar desde atrás",
        ["poner o dejar algo detrás", "estar detrás de una acción", "ocultar o depositar algo"],
        "Menos frecuente como prefijo productivo; muchos verbos se aprenden individualmente.",
        [("hinterlassen", "dejar atrás"), ("hinterfragen", "cuestionar"), ("hinterlegen", "depositar")], 60,
    ),
    "über-": PrefixInfo(
        "über-", "ambiguous", "sobre, por encima, exceso, transferencia o revisión",
        ["pasar por encima de algo", "transferir algo", "revisar o comprobar", "exceder un límite"],
        "Puede ser separable o inseparable, y el acento cambia el significado: übersetzen puede ser 'traducir' o 'cruzar'.",
        [("übersetzen", "traducir / cruzar"), ("überlegen", "reflexionar / poner encima"), ("überprüfen", "comprobar")], 10,
    ),
    "unter-": PrefixInfo(
        "unter-", "ambiguous", "debajo, subordinación, interrupción o inclusión dentro de un grupo",
        ["poner debajo", "interrumpir una acción", "subordinar o clasificar", "alojar o colocar en un lugar"],
        "Puede ser separable o inseparable; aprende cada verbo con su acento y ejemplo.",
        [("unterbringen", "alojar / colocar"), ("unterbrechen", "interrumpir"), ("unterstellen", "poner debajo / atribuir")], 20,
    ),
    "um-": PrefixInfo(
        "um-", "ambiguous", "alrededor, cambio de dirección, transformación o inversión",
        ["mover alrededor de algo", "cambiar de posición o dirección", "transformar o reorganizar", "derribar en algunos verbos separables"],
        "El contraste separable/inseparable es importante: umfahren puede significar 'atropellar/derribar' o 'rodear'.",
        [("umstellen", "reorganizar / rodear"), ("umziehen", "mudarse / cambiarse de ropa"), ("umfahren", "rodear / atropellar según acento")], 10,
    ),
    "wider-": PrefixInfo(
        "wider-", "ambiguous", "oposición, resistencia o contradicción",
        ["actuar contra algo", "resistir o contradecir", "devolver oposición a una acción"],
        "No debe confundirse con wieder- ('de nuevo'). wider- es más formal y significa oposición.",
        [("widersprechen", "contradecir"), ("widerstehen", "resistir"), ("widerlegen", "refutar")], 40,
    ),
    "wieder-": PrefixInfo(
        "wieder-", "separable", "repetición, retorno o recuperación",
        ["hacer algo de nuevo", "volver a un estado anterior", "recuperar o restaurar algo"],
        "No lo confundas con wider-: wieder- = de nuevo; wider- = contra.",
        [("wiederkommen", "volver"), ("wiederholen", "repetir"), ("wiederherstellen", "restaurar")], 20,
    ),
}


def normalize_prefix(prefix: str) -> str:
    prefix = (prefix or "").strip()
    return prefix if not prefix or prefix.endswith("-") else f"{prefix}-"


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_bool(value: Any, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "ja"}


def load_examples(path: Optional[Path]) -> Dict[str, List[Example]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return load_examples_json(path)
    if path.suffix.lower() == ".csv":
        return load_examples_csv(path)
    raise ValueError(f"Unsupported input extension: {path.suffix}. Use .json or .csv")


def is_usable_sense(verb_row: Dict[str, Any], sense_row: Dict[str, Any]) -> bool:
    quality_ok = parse_bool(sense_row.get("is_quality_ok", verb_row.get("is_quality_ok", True)))
    if not quality_ok:
        return False

    derived = clean_text(verb_row.get("derived", sense_row.get("derived", "")))
    gloss_es = clean_text(sense_row.get("gloss_es", verb_row.get("gloss_es", "")))
    example_de = clean_text(sense_row.get("example_de", verb_row.get("example_de", "") or verb_row.get("example", "")))
    if not derived or not gloss_es:
        return False
    if example_de and len(example_de.split()) > 22:
        return False

    flags = sense_row.get("quality_flags", verb_row.get("quality_flags", []))
    flags_text = flags.lower() if isinstance(flags, str) else json.dumps(flags, ensure_ascii=False).lower()
    hard_bad = {"semantic_mismatch", "invalid_context_answer", "placeholder_gloss", "placeholder_example", "metadata_gloss"}
    return not any(flag in flags_text for flag in hard_bad)


def add_example(result: Dict[str, List[Example]], prefix: str, example: Example) -> None:
    if not example.derived:
        return
    bucket = result.setdefault(prefix, [])
    if not any(existing.derived == example.derived for existing in bucket):
        bucket.append(example)


def load_examples_json(path: Path) -> Dict[str, List[Example]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, List[Example]] = {}
    if not isinstance(payload, list):
        return result
    for item in payload:
        if not isinstance(item, dict):
            continue
        prefix = normalize_prefix(str(item.get("prefix", "")))
        if not prefix:
            continue
        derived = clean_text(item.get("derived", ""))
        source = clean_text(item.get("source", ""))
        senses = item.get("senses")
        if isinstance(senses, list) and senses:
            for sense in senses:
                if not isinstance(sense, dict) or not is_usable_sense(item, sense):
                    continue
                add_example(result, prefix, Example(
                    derived=derived,
                    gloss_es=clean_text(sense.get("gloss_es", "") or item.get("gloss_es", "")),
                    example_de=clean_text(sense.get("example_de", "") or item.get("example_de", "") or item.get("example", "")),
                    construction=clean_text(sense.get("construction", "")),
                    source=source,
                ))
        elif is_usable_sense(item, item):
            add_example(result, prefix, Example(
                derived=derived,
                gloss_es=clean_text(item.get("gloss_es", "")),
                example_de=clean_text(item.get("example_de", "") or item.get("example", "")),
                construction=clean_text(item.get("construction", "")),
                source=source,
            ))
    return result


def load_examples_csv(path: Path) -> Dict[str, List[Example]]:
    result: Dict[str, List[Example]] = {}
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            prefix = normalize_prefix(str(row.get("prefix", "")))
            if prefix and is_usable_sense(row, row):
                add_example(result, prefix, Example(
                    derived=clean_text(row.get("derived", "")),
                    gloss_es=clean_text(row.get("gloss_es", "")),
                    example_de=clean_text(row.get("example_de", "") or row.get("example", "")),
                    construction=clean_text(row.get("construction", "")),
                    source=clean_text(row.get("source", "")),
                ))
    return result


def example_score(ex: Example) -> tuple[int, int, str]:
    return (0 if ex.example_de else 1, len(ex.example_de.split()) if ex.example_de else 999, ex.derived)


def select_examples(prefix: str, examples_by_prefix: Dict[str, List[Example]], max_examples: int) -> List[Example]:
    examples = sorted(examples_by_prefix.get(prefix, []), key=example_score)[:max_examples]
    if len(examples) >= max_examples:
        return examples
    existing = {ex.derived for ex in examples}
    for derived, gloss_es in PREFIXES[prefix].fallback_examples:
        if derived not in existing:
            examples.append(Example(derived=derived, gloss_es=gloss_es))
            existing.add(derived)
        if len(examples) >= max_examples:
            break
    return examples


def render_front(info: PrefixInfo) -> str:
    return f"¿Qué intuición semántica suele aportar el prefijo alemán {info.prefix}?"


def render_back(info: PrefixInfo, examples: Sequence[Example]) -> str:
    parts: List[str] = [
        f"<b>{html.escape(info.prefix)}</b> — {html.escape(info.separability)}",
        "",
        "<b>Intuición principal</b>",
        html.escape(info.intuition_es),
    ]
    if info.patterns_es:
        parts += ["", "<b>Patrones frecuentes</b>", "<ul>"]
        for pattern in info.patterns_es:
            parts.append(f"<li>{html.escape(pattern)}</li>")
        parts.append("</ul>")
    if examples:
        parts += ["", "<b>Ejemplos</b>", "<ul>"]
        for ex in examples:
            line = f"<b>{html.escape(ex.derived)}</b>"
            if ex.gloss_es:
                line += f" = {html.escape(ex.gloss_es)}"
            if ex.example_de:
                line += f"<br><i>{html.escape(ex.example_de)}</i>"
            parts.append(f"<li>{line}</li>")
        parts.append("</ul>")
    parts += [
        "",
        "<b>Cuidado</b>",
        html.escape(info.caveat_es),
        "",
        "No lo memorices como una regla mecánica: úsalo como una pista para interpretar verbos nuevos.",
    ]
    return "<br>\n".join(parts)


def render_reverse_front(info: PrefixInfo) -> str:
    return "¿Qué prefijo alemán suele asociarse con esta intuición?\n\n" + info.intuition_es


def tags_for(info: PrefixInfo, direction: str) -> List[str]:
    return [
        "deck::german_prefix_verbs",
        "cardtype::prefix_semantics",
        f"prefix::{info.prefix.rstrip('-')}",
        f"separability::{info.separability}",
        direction,
    ]


def build_cards(
    examples_by_prefix: Dict[str, List[Example]],
    *,
    include_prefixes: Optional[set[str]] = None,
    exclude_prefixes: Optional[set[str]] = None,
    max_examples: int = 4,
    include_reverse: bool = False,
    include_ambiguous: bool = True,
) -> List[Card]:
    cards: List[Card] = []
    for prefix, info in sorted(PREFIXES.items(), key=lambda item: (item[1].priority, item[0])):
        if include_prefixes is not None and prefix not in include_prefixes:
            continue
        if exclude_prefixes is not None and prefix in exclude_prefixes:
            continue
        if not include_ambiguous and info.separability == "ambiguous":
            continue
        examples = select_examples(prefix, examples_by_prefix, max_examples)
        cards.append(Card("prefix_semantics", prefix, render_front(info), render_back(info, examples), tags_for(info, "direction::prefix_to_meaning")))
        if include_reverse:
            cards.append(Card("prefix_semantics_reverse", prefix, render_reverse_front(info), render_back(info, examples), tags_for(info, "direction::meaning_to_prefix")))
    return cards


def stable_guid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_int_id(text: str, modulo: int = 10**10) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def write_csv(cards: Sequence[Card], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["guid", "cardtype", "prefix", "front", "back", "tags"])
        writer.writeheader()
        for card in cards:
            writer.writerow({
                "guid": stable_guid(f"{card.cardtype}:{card.prefix}:{card.front}"),
                "cardtype": card.cardtype,
                "prefix": card.prefix,
                "front": card.front,
                "back": card.back,
                "tags": " ".join(card.tags),
            })


def sanitize_tag(tag: str) -> str:
    return re.sub(r"\s+", "_", tag)


def write_apkg(cards: Sequence[Card], out_path: Path, deck_name: str) -> None:
    try:
        import genanki  # type: ignore
    except ImportError as exc:
        raise RuntimeError("genanki is not installed. Install it with `pip install genanki` or export as CSV instead.") from exc

    model = genanki.Model(
        stable_int_id("german-prefix-semantics-model-v1"),
        "German Prefix Semantics Model",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Prefix"}, {"name": "CardType"}],
        templates=[{"name": "Card 1", "qfmt": '<div class="card front">{{Front}}</div>', "afmt": '{{FrontSide}}<hr id="answer"><div class="card back">{{Back}}</div>'}],
        css="""
.card { font-family: Arial, sans-serif; font-size: 20px; text-align: left; color: #111; background: #fff; line-height: 1.45; }
.front { font-size: 22px; }
.back { font-size: 18px; }
ul { margin-top: 0.3em; }
""",
    )
    deck = genanki.Deck(stable_int_id(deck_name), deck_name)
    for card in cards:
        note = genanki.Note(
            model=model,
            fields=[card.front, card.back, card.prefix, card.cardtype],
            tags=[sanitize_tag(tag) for tag in card.tags],
            guid=stable_guid(f"{card.cardtype}:{card.prefix}:{card.front}"),
        )
        deck.add_note(note)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(out_path))


def parse_prefix_list(value: str) -> set[str]:
    return {normalize_prefix(raw.strip()) for raw in value.split(",") if raw.strip()}


def validate_cards(cards: Sequence[Card]) -> None:
    if not cards:
        raise ValueError("No cards generated.")
    seen = set()
    for card in cards:
        if not card.front.strip():
            raise ValueError(f"Empty front for {card.prefix}")
        if not card.back.strip():
            raise ValueError(f"Empty back for {card.prefix}")
        if card.prefix not in PREFIXES:
            raise ValueError(f"Unknown prefix: {card.prefix}")
        key = (card.cardtype, card.prefix, card.front)
        if key in seen:
            raise ValueError(f"Duplicate card: {key}")
        seen.add(key)


def infer_format(out_path: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return "apkg" if out_path.suffix.lower() == ".apkg" else "csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Anki cards for German prefix semantic intuitions.")
    parser.add_argument("--input", type=Path, default=None, help="Optional out.json/out.csv from the verb-prefix pipeline. Used to select examples.")
    parser.add_argument("--out", type=Path, default=Path("prefix_semantics.csv"), help="Output path. Use .csv or .apkg, or set --format explicitly.")
    parser.add_argument("--format", choices=["csv", "apkg"], default=None, help="Output format. Defaults to extension of --out.")
    parser.add_argument("--deck-name", default="German Prefix Semantics", help="Deck name for .apkg export.")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum examples shown per prefix card.")
    parser.add_argument("--include-reverse", action="store_true", help="Also generate meaning-to-prefix cards. Disabled by default.")
    parser.add_argument("--exclude-ambiguous", action="store_true", help="Skip ambiguous prefixes such as um-, über-, unter-, durch-.")
    parser.add_argument("--only-prefixes", type=str, default="", help="Comma-separated prefix allowlist, e.g. 'an,auf,ver,zurück'.")
    parser.add_argument("--exclude-prefixes", type=str, default="", help="Comma-separated prefix denylist.")
    parser.add_argument("--validate-only", action="store_true", help="Validate generation but do not write files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples_by_prefix = load_examples(args.input)
    cards = build_cards(
        examples_by_prefix,
        include_prefixes=parse_prefix_list(args.only_prefixes) if args.only_prefixes else None,
        exclude_prefixes=parse_prefix_list(args.exclude_prefixes) if args.exclude_prefixes else None,
        max_examples=max(args.max_examples, 0),
        include_reverse=args.include_reverse,
        include_ambiguous=not args.exclude_ambiguous,
    )
    validate_cards(cards)

    if args.validate_only:
        print(f"Validation ok: {len(cards)} cards generated.")
        return 0

    output_format = infer_format(args.out, args.format)
    if output_format == "csv":
        write_csv(cards, args.out)
    else:
        write_apkg(cards, args.out, args.deck_name)
    print(f"Wrote {len(cards)} prefix semantic cards to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

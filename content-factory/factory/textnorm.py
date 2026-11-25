"""Spoken-text normalisation (German + English) per display word.

Subtitles keep the display form ("16,1 Mrd. km"); the TTS gets the spoken form
("sechzehn Komma eins Milliarden Kilometer"). Normalising word by word keeps a 1:1
mapping display word -> spoken chunk, so word timings stay exact.
"""
from __future__ import annotations

import re

try:
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None

UNITS_DE = {
    "km": "Kilometer", "m": "Meter", "cm": "Zentimeter", "mm": "Millimeter", "kg": "Kilogramm", "g": "Gramm",
    "t": "Tonnen", "%": "Prozent", "°c": "Grad Celsius", "°": "Grad", "k": "Kelvin", "mj": "Megajoule",
    "kwh": "Kilowattstunden", "mw": "Megawatt", "gw": "Gigawatt", "ghz": "Gigahertz", "gb": "Gigabyte",
    "tb": "Terabyte", "mph": "Meilen pro Stunde", "km/h": "Kilometer pro Stunde", "km/s": "Kilometer pro Sekunde",
    "mrd.": "Milliarden", "mrd": "Milliarden", "mio.": "Millionen", "mio": "Millionen", "tsd.": "Tausend",
    "mi": "Meilen", "jh.": "Jahrhundert", "z.b.": "zum Beispiel", "ca.": "circa", "usw.": "und so weiter",
    "bzw.": "beziehungsweise", "dr.": "Doktor", "nr.": "Nummer", "&": "und", "×": "mal", "x": "mal",
    "~": "rund", "≈": "ungefähr", "+": "plus", "→": "", "–": "", "—": "",
    "ki": "K I", "usa": "U S A", "eu": "E U", "nasa": "Nasa", "dna": "D N A", "crispr": "Crisper",
    "fda": "F D A", "esa": "E S A", "ai": "A I", "gpu": "G P U", "cpu": "C P U",
}
UNITS_EN = {"km": "kilometers", "%": "percent", "°c": "degrees Celsius", "×": "times", "~": "about",
            "mi": "miles", "mj": "megajoules", "&": "and", "ai": "A I"}

SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _num_de(token: str, year_hint: bool) -> str:
    t = token.replace(".", "") if re.fullmatch(r"\d{1,3}(\.\d{3})+", token) else token
    t = t.replace(",", ".") if re.fullmatch(r"\d+,\d+", t) else t
    if num2words is None:
        return token
    try:
        if year_hint and re.fullmatch(r"1[1-9]\d\d", t):
            return num2words(int(t), lang="de", to="year")
        if re.fullmatch(r"\d+\.", token):  # ordinal "18." -> achtzehnte
            return num2words(int(token[:-1]), lang="de", to="ordinal")
        return num2words(float(t) if "." in t else int(t), lang="de")
    except Exception:
        return token


def _num_en(token: str, year_hint: bool) -> str:
    t = token.replace(",", "")
    if num2words is None:
        return token
    try:
        if year_hint and re.fullmatch(r"1[1-9]\d\d|20\d\d", t):
            return num2words(int(t), lang="en", to="year")
        return num2words(float(t) if "." in t else int(t), lang="en")
    except Exception:
        return token


MONTHS = ("januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober",
          "november", "dezember", "jahrhundert", "jahrtausend", "mal", "platz", "stock")


def spoken_word(word: str, lang: str = "de", prev: str = "", nxt: str = "") -> str:
    sup = re.fullmatch(r"(.*?)10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)(.*)", word)
    if sup:
        n = sup.group(2).translate(SUPERSCRIPT)
        return (sup.group(1) + ("zehn hoch " + _num_de(n, False) if lang == "de" else "ten to the " + _num_en(n, False))
                + sup.group(3)).strip()
    w = word.translate(SUPERSCRIPT)
    m = re.match(r"^([\"„“‚(\[]*)(.*?)([\"“”‘)\].,!?;:…]*)$", w)
    lead, core, trail = m.groups() if m else ("", w, "")
    units = UNITS_DE if lang == "de" else UNITS_EN
    low = core.lower()
    if trail.startswith(".") and low + "." in units:  # abbreviation dot belongs to the word
        return (units[low + "."] + trail[1:]).strip()
    # German ordinal: "18. November" / "am 18."
    if lang == "de" and re.fullmatch(r"\d{1,3}", core) and trail == "." and nxt and (
            nxt[:1].islower() or nxt.lower().strip(".,") in MONTHS):
        base = num2words(int(core), lang="de", to="ordinal") if num2words else core
        return base + ("n" if prev.lower() in ("am", "vom", "zum", "im", "beim", "seit dem") else "")
    if low in units:
        return (units[low] + trail).strip()
    # 10^25 / 10²⁵
    pm = re.fullmatch(r"10\^?(\d+)", core)
    if pm and "^" in core:
        n = pm.group(1)
        return ("zehn hoch " + _num_de(n, False) if lang == "de" else "ten to the " + _num_en(n, False)) + trail
    # number with attached unit: 16,1km, 300MJ, 49%
    nm = re.fullmatch(r"([~≈]?)(\d[\d.,]*)([a-zA-Z%°/×]+\.?)?", core)
    if nm:
        approx, num, unit = nm.groups()
        year_hint = prev.lower() in ("im", "jahr", "seit", "bis", "in", "von", "in the", "since") or (len(num) == 4 and not unit)
        if num.endswith(".") and lang == "de" and not unit:
            spoken = _num_de(num, year_hint)
        else:
            num = num.rstrip(".")
            spoken = _num_de(num, year_hint) if lang == "de" else _num_en(num, year_hint)
        if unit:
            spoken += " " + units.get(unit.lower(), unit)
        if approx:
            spoken = units.get(approx, "") + " " + spoken
        return (spoken + trail).strip()
    if re.fullmatch(r"[A-ZÄÖÜ]{2,5}", core) and low in units:
        return units[low] + trail
    return lead + core + trail


def spoken_words(words: list[str], lang: str = "de") -> list[str]:
    out, prev = [], ""
    for i, w in enumerate(words):
        out.append(spoken_word(w, lang, prev, words[i + 1] if i + 1 < len(words) else ""))
        prev = w
    return out


def spoken_text(text: str, lang: str = "de") -> str:
    return " ".join(x for x in spoken_words(text.split(), lang) if x)

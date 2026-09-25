"""Text normalization: Indic transliteration, accent folding, tokenization, phonetic skeletons.

All nine Indic scripts in the data (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
Telugu, Kannada, Malayalam) share the ISCII-derived Unicode layout, so one offset table
(offset = codepoint - block_start) transliterates every one of them.
"""
from __future__ import annotations

import re

import polars as pl

# offset -> latin. Consonants carry no vowel; the inherent 'a' is inserted by _INHERENT.
_OFF = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x50: "om", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f",
    0x5F: "y", 0x60: "ri", 0x61: "li", 0x64: " ", 0x65: " ",
    0x70: "n", 0x71: "", 0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k",
}
_BLOCKS = range(0x0900, 0x0D80, 0x80)
_TABLE: dict[int, str] = {}
for _b in _BLOCKS:
    for _o in range(0x80):
        _TABLE[_b + _o] = _OFF.get(_o, str(_o - 0x66) if 0x66 <= _o <= 0x6F else "")
_TABLE[0x09CE] = "t"  # Bengali khanda ta
_TABLE[0x0B71] = "v"  # Oriya wa
for _z in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x200E, 0x200F):
    _TABLE[_z] = ""

_cons = "".join(chr(b + o) for b in _BLOCKS for o in list(range(0x15, 0x3A)) + list(range(0x58, 0x60)))
_marks = "".join(chr(b + o) for b in _BLOCKS for o in (0x3C,))
# consonant (+nukta) directly followed by another consonant or a letter-ish char gets inherent 'a'
_INHERENT = re.compile(f"([{_cons}][{_marks}]?)(?=[{_cons}])")
INDIC_RE = r"[ऀ-ൿ]"


def translit_indic(s: str) -> str:
    return _INHERENT.sub(r"\1a", s).translate(_TABLE)


def fold(col: pl.Expr) -> pl.Expr:
    """Indic->latin, accents stripped, lowercase, '&'->and, apostrophes dropped, other punct -> space."""
    return (
        pl.when(col.str.contains(INDIC_RE))
        .then(col.map_elements(translit_indic, return_dtype=pl.String))
        .otherwise(col)
        .str.normalize("NFKD")
        .str.replace_all(r"\p{Mn}", "")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"['’`]", "")
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.strip_chars()
    )


_SKEL = [
    (re.compile(r"ph"), "f"), (re.compile(r"[sc]h"), "s"), (re.compile(r"([kgtdbj])h"), r"\1"),
    (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"[cq]"), "k"), (re.compile(r"w"), "v"),
    (re.compile(r"z"), "s"), (re.compile(r"x"), "ks"),
]
_VOW = re.compile(r"[aeiouyh]")
_REP = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Phonetic consonant skeleton of one folded token: 'private'~'praivet' -> 'prvt'. Digits unchanged."""
    if tok.isdigit():
        return tok.lstrip("0") or "0"
    for pat, rep in _SKEL:
        tok = pat.sub(rep, tok)
    return _REP.sub(r"\1", tok[:1] + _VOW.sub("", tok[1:]))

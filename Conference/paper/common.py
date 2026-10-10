"""Shared helpers: inline markup parsing and layout constants for both renderers."""

import re

# ---------------------------------------------------------------- layout spec
# Matches the reference PDF: A4, Times New Roman, two columns.
PAGE_W, PAGE_H = 595.44, 841.68        # A4 in points
MARGIN_L, MARGIN_R = 47.0, 19.0
MARGIN_T, MARGIN_B = 55.0, 51.0
COL_GAP = 19.4
COL_W = (PAGE_W - MARGIN_L - MARGIN_R - COL_GAP) / 2.0   # ~255.0 pt
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R

BODY_SIZE = 8.5
BODY_LEAD = 10.4
TITLE_SIZE = 15.5
AUTHOR_SIZE = 11.0
AFFIL_SIZE = 9.0
CORR_SIZE = 8.5
HEADING_SIZE = 10.0
SUBSEC_SIZE = 9.5
ABSTRACT_SIZE = 8.5
TABLE_SIZE = 6.6
CAPTION_SIZE = 7.6
REF_SIZE = 7.6

FONT = "Times New Roman"

# ------------------------------------------------------------- inline markup
_TAG = re.compile(r"<(/?)(b|i|sup|sub|sc)>")


def parse_runs(text, bold=False, italic=False, size=None, sc=False):
    """Split inline markup into a list of run dicts.

    Each run is {text, bold, italic, sup, sub, sc, size}.
    Unmarked text carries the supplied defaults.
    """
    runs = []
    stack = []
    pos = 0
    text = text.replace("<br/>", "\n")

    def emit(chunk):
        if not chunk:
            return
        # <br/> inside the chunk becomes a run whose text is a newline, which
        # each renderer maps onto its own hard-break mechanism.
        for i, piece in enumerate(chunk.split("\n")):
            runs.append({
                "text": piece + ("\n" if i < chunk.count("\n") else ""),
                "bold": bold or ("b" in stack),
                "italic": italic or ("i" in stack),
                "sc": sc or ("sc" in stack),
                "sup": "sup" in stack,
                "sub": "sub" in stack,
                "size": size,
            })

    for m in _TAG.finditer(text):
        emit(text[pos:m.start()])
        if m.group(1):
            if m.group(2) in stack:
                # remove the innermost occurrence
                for k in range(len(stack) - 1, -1, -1):
                    if stack[k] == m.group(2):
                        del stack[k]
                        break
        else:
            stack.append(m.group(2))
        pos = m.end()
    emit(text[pos:])
    return runs or [{"text": "", "bold": bold, "italic": italic, "sc": sc,
                     "sup": False, "sub": False, "size": size}]


_NUMERIC_ENTITY = re.compile(r"&#(\d+);")
_NAMED = (
    ("&nbsp;", "\u00a0"),
    ("&mdash;", "\u2014"),
    ("&ndash;", "\u2013"),
    ("&hellip;", "\u2026"),
    ("&times;", "\u00d7"),
)


def normalise(text):
    """Decode the HTML entities used in the content file.

    Numeric references are decoded first and ``&amp;`` last, so that an escaped
    entity such as ``&amp;#215;`` survives as literal text instead of being
    decoded twice.
    """
    text = _NUMERIC_ENTITY.sub(lambda m: chr(int(m.group(1))), text)
    for a, b in _NAMED:
        text = text.replace(a, b)
    return text.replace("&amp;", "&")
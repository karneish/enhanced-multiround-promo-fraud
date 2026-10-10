#!/usr/bin/env python
"""Build the AMDL-TPNE paper as a PDF in the reference document's IEEE two-column format."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether,
                                NextPageTemplate, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

import content as C
from common import (parse_runs, normalise, BODY_LEAD, BODY_SIZE, TITLE_SIZE, AUTHOR_SIZE,
                    AFFIL_SIZE, CORR_SIZE, HEADING_SIZE, SUBSEC_SIZE, ABSTRACT_SIZE,
                    TABLE_SIZE, CAPTION_SIZE, REF_SIZE, PAGE_W, PAGE_H, MARGIN_L,
                    MARGIN_R, MARGIN_T, MARGIN_B, COL_GAP, COL_W, TEXT_W)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "AMDL-TPNE_Paper.pdf")

FRAME_H = PAGE_H - MARGIN_T - MARGIN_B

# ------------------------------------------------------------------- styles
def style(name, size, leading=None, align=TA_JUSTIFY, space_before=0, space_after=0, **kw):
    return ParagraphStyle(name, fontName="Times-Roman", fontSize=size,
                          leading=leading or size * 1.22, alignment=align,
                          spaceBefore=space_before, spaceAfter=space_after, **kw)


S_TITLE = style("title", TITLE_SIZE, 18, TA_CENTER)
S_AUTHOR = style("author", AUTHOR_SIZE, 13.5, TA_CENTER)
S_AFFIL = style("affil", AFFIL_SIZE, 10.8, TA_CENTER)
S_CORR = style("corr", CORR_SIZE, 10.5, TA_CENTER)
S_ABSTRACT = style("abstract", ABSTRACT_SIZE, BODY_LEAD, TA_JUSTIFY)
S_BODY = style("body", BODY_SIZE, BODY_LEAD, TA_JUSTIFY)
S_HEAD = style("head", HEADING_SIZE, HEADING_SIZE + 2.4, TA_CENTER,
               space_before=8, space_after=3.5)
S_SUBHEAD = style("subhead", SUBSEC_SIZE, SUBSEC_SIZE + 2.0, TA_LEFT,
                  space_before=5, space_after=2.5)
S_FIGCAP = style("figcap", CAPTION_SIZE, CAPTION_SIZE + 2.0, TA_JUSTIFY, space_after=7)
S_TABCAP = style("tabcap", CAPTION_SIZE, CAPTION_SIZE + 2.0, TA_CENTER, space_after=3)
S_REF = style("ref", REF_SIZE, REF_SIZE + 1.8, TA_JUSTIFY,
              leftIndent=12, firstLineIndent=-12)
S_TAB = style("tab", TABLE_SIZE, TABLE_SIZE + 1.8, TA_LEFT)
S_TABH = style("tabh", TABLE_SIZE, TABLE_SIZE + 1.8, TA_LEFT)
S_EQ = style("eq", BODY_SIZE, BODY_SIZE + 2.6, TA_CENTER,
             space_before=4, space_after=4)


# ------------------------------------------------------- inline markup -> RL
def rl_markup(text, bold=False, italic=False, size=None, sc=False):
    """Convert our inline markup into reportlab Paragraph markup."""
    out = []
    for r in parse_runs(normalise(text), bold=bold, italic=italic, size=size, sc=sc):
        parts = r["text"].split("\n")
        for i, part in enumerate(parts):
            if i:
                out.append("<br/>")
            if not part:
                continue
            t = (part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            if r["sc"]:
                t = t.upper()
            if bold or r["bold"]:
                t = f"<b>{t}</b>"
            if italic or r["italic"]:
                t = f"<i>{t}</i>"
            if r["sup"]:
                t = f"<super>{t}</super>"
            elif r["sub"]:
                t = f"<sub>{t}</sub>"
            out.append(t)
    return "".join(out)


# --------------------------------------------------------------- flowables
def P(text, st, **kw):
    return Paragraph(rl_markup(text), st, **kw)


def para_text(text, st, bold=False, italic=False, sc=False):
    return Paragraph(rl_markup(text, bold, italic, None, sc), st)


def title_block():
    f = [P(C.TITLE, S_TITLE), Spacer(1, 5)]

    parts = []
    for i, (name, mark, corr) in enumerate(C.AUTHORS):
        if i:
            parts.append(",&nbsp;&nbsp;")
        star = "*" if corr else ""
        parts.append(f"{name}<super>{mark}{star}</super>")
    f.append(Paragraph("".join(parts), S_AUTHOR))
    f.append(Spacer(1, 3))

    for mark, aff in C.AFFILIATIONS:
        f.append(Paragraph(f"<super>{mark}</super>{aff}", S_AFFIL))
    f.append(Spacer(1, 4))

    c = C.CORRESPONDING
    f.append(Paragraph(f"<i>Corresponding author: {c['name']}</i>", S_CORR))
    f.append(Paragraph(
        f"<i>Email: {c['email']}</i>", S_CORR))
    f.append(Spacer(1, 9))

    f.append(Paragraph(
        rl_markup("<b>ABSTRACT</b>&nbsp;&mdash;&nbsp;") + rl_markup(C.ABSTRACT),
        ParagraphStyle("ab", parent=S_ABSTRACT, leftIndent=9, rightIndent=9)))
    f.append(Spacer(1, 4))
    f.append(Paragraph(
        rl_markup("<b>KEYWORDS</b>&nbsp;&mdash;&nbsp;")
        + rl_markup(C.KEYWORDS, italic=True),
        ParagraphStyle("kw", parent=S_ABSTRACT, leftIndent=9, rightIndent=9)))
    return f


def measure(flowables, width):
    """Total height a list of flowables will occupy at the given width."""
    total = 0.0
    for fl in flowables:
        try:
            w, h = fl.wrap(width, 1e6)
        except Exception:
            h = getattr(fl, "_h", 0.0)
        total += h
        total += getattr(fl, "getSpaceBefore", lambda: 0.0)()
        total += getattr(fl, "getSpaceAfter", lambda: 0.0)()
    return total


def build_flowables():
    f = []
    for kind, payload in C.BODY:
        if kind == "h":
            level, text = payload
            if level == 1:
                f.append(Paragraph(rl_markup(text, bold=True, sc=True), S_HEAD))
            else:
                f.append(Paragraph(rl_markup(text, bold=True, italic=True), S_SUBHEAD))
        elif kind == "p":
            f.append(P(payload, S_BODY))
        elif kind == "eq":
            f.append(P(payload, S_EQ))
        elif kind == "fig":
            path, number, caption = payload
            f.extend(figure(path, caption))
        elif kind == "tab":
            f.extend(table(payload))
    return f


def figure(path, caption):
    full = os.path.join(HERE, path)
    iw, ih = ImageReader(full).getSize()
    w = COL_W
    h = w * ih / iw
    img = Image(full, width=w, height=h)
    img.hAlign = "CENTER"
    cap = Paragraph(rl_markup(caption), S_FIGCAP)
    return [Spacer(1, 7), KeepTogether([img, Spacer(1, 3), cap])]


def table(spec):
    size = spec.get("size", TABLE_SIZE)
    head, rows = spec["head"], spec["rows"]
    widths = spec.get("widths") or [1.0 / len(head)] * len(head)
    total = sum(widths)
    cw = [COL_W * w / total for w in widths]

    ts = ParagraphStyle("t", parent=S_TAB, fontSize=size, leading=size + 1.8)
    th = ParagraphStyle("th", parent=ts)

    data = [[Paragraph(rl_markup(c, bold=True), th) for c in head]]
    for r in rows:
        data.append([Paragraph(rl_markup(c), ts) for c in r])

    t = Table(data, colWidths=cw, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, (0, 0, 0)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, (0, 0, 0)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, (0, 0, 0)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    return [Spacer(1, 7), Paragraph(rl_markup(spec["caption"]), S_TABCAP),
            t, Spacer(1, 8)]


def references():
    f = [Spacer(1, 7), Paragraph(rl_markup("REFERENCES", bold=True, sc=True), S_HEAD)]
    for i, ref in enumerate(C.REFERENCES, 1):
        f.append(Paragraph(f"[{i}]&nbsp;&nbsp;" + rl_markup(normalise(ref)), S_REF))
    return f


# ------------------------------------------------------------------- pages
def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 7.5)
    canvas.drawCentredString(PAGE_W / 2.0, MARGIN_B - 20, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


class Paper(BaseDocTemplate):
    def __init__(self, path, title_h, **kw):
        BaseDocTemplate.__init__(self, path, pagesize=(PAGE_W, PAGE_H),
                                 leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                                 topMargin=MARGIN_T, bottomMargin=MARGIN_B,
                                 title=C.TITLE,
                                 author="DR. R.G. SURESH KUMAR; KARNEISH S; KAVIN VENTHAN J",
                                 subject="Multi-round adversarial promo fraud detection",
                                 **kw)
        def frame(x, w, h):
            return Frame(x, MARGIN_B, w, h, leftPadding=0, rightPadding=0,
                         topPadding=0, bottomPadding=0)

        # Frames grow upward from their y origin, so the title band sits directly
        # on top of the two columns: title origin = bottom margin + column height.
        col_h = FRAME_H - title_h
        title_frame = Frame(MARGIN_L, MARGIN_B + col_h, TEXT_W, title_h,
                            leftPadding=0, rightPadding=0,
                            topPadding=0, bottomPadding=0)

        col1 = frame(MARGIN_L, COL_W, col_h)
        col2 = frame(MARGIN_L + COL_W + COL_GAP, COL_W, col_h)
        col1_full = frame(MARGIN_L, COL_W, FRAME_H)
        col2_full = frame(MARGIN_L + COL_W + COL_GAP, COL_W, FRAME_H)

        self.addPageTemplates([
            PageTemplate(id="first", frames=[title_frame, col1, col2], onPage=draw_page),
            PageTemplate(id="later", frames=[col1_full, col2_full], onPage=draw_page),
        ])


def main():
    head = title_block()
    title_h = measure(head, TEXT_W) + 6

    doc = Paper(OUT, title_h)
    story = head + [FrameBreak(), NextPageTemplate("later")]
    story.extend(build_flowables())
    story.extend(references())
    doc.build(story)
    print("wrote", OUT, f"(title block {title_h:.1f} pt)")


if __name__ == "__main__":
    main()
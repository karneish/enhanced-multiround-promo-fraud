# -*- coding: utf-8 -*-
"""Build the filled-in Phase I project report (.docx) from the RGCET format template.

The template (PROJECT PHASE_I REPORT_FORMAT_RGCET_new_2025.docx) supplies the
college's page setup, Times New Roman styles, numbering and the college emblem.
This script regenerates word/document.xml completely with this project's content
so that no template placeholder or previous-project text survives, then repacks
the package. Styles, theme, numbering and the emblem are reused byte-for-byte.

Content: "Real-Time Adaptive 3-Model Fraud Detection for Multi-Round Promotion
and Reward Abuse".
"""
import os
import re
import zipfile
from xml.sax.saxutils import escape as _esc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "PROJECT PHASE_I REPORT_FORMAT_RGCET_new_2025.docx")
DST = os.path.join(HERE, "PROJECT PHASE_I REPORT_RGCET_2025.docx")

TITLE = ("REAL-TIME ADAPTIVE 3-MODEL FRAUD DETECTION FOR "
         "MULTI-ROUND PROMOTION AND REWARD ABUSE")
SHORT_TITLE = "Real-Time Adaptive 3-Model Fraud Detection"
AUTHOR = "KARNEISH S"
MONTH_YEAR = "SEPTEMBER 2026"
ACADEMIC_YEAR = "2026 - 2027"
GUIDE_PLACEHOLDER = "Ms./Mr. ____________________"
HOD_PLACEHOLDER = "Dr. ____________________"
REG_PLACEHOLDER = "Register No.: ____________________"

# full usable text width on US Letter with 1 inch margins (twips)
TEXT_W = 9360

# ---------------------------------------------------------------- primitives


def esc(t):
    return _esc(str(t))


def run(text, b=False, i=False, u=False, font=None, sz=None, color=None,
        caps=False, sup=False):
    rpr = []
    if font:
        rpr.append('<w:rFonts w:ascii="%s" w:hAnsi="%s" w:cs="%s"/>'
                   % (font, font, font))
    if b:
        rpr.append("<w:b/>")
    if i:
        rpr.append("<w:i/>")
    if u:
        rpr.append('<w:u w:val="single"/>')
    if caps:
        rpr.append("<w:caps/>")
    if color:
        rpr.append('<w:color w:val="%s"/>' % color)
    if sz:
        rpr.append('<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz))
    if sup:
        rpr.append('<w:vertAlign w:val="superscript"/>')
    rpr = ("<w:rPr>%s</w:rPr>" % "".join(rpr)) if rpr else ""
    return ('<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>'
            % (rpr, esc(text)))


def para(runs="", style=None, jc=None, ind=None, spacing=None, shade=None,
         keep=False, page_break_before=False, borders=None):
    ppr = []
    if style:
        ppr.append('<w:pStyle w:val="%s"/>' % style)
    if keep:
        ppr.append("<w:keepNext/><w:keepLines/>")
    if page_break_before:
        ppr.append("<w:pageBreakBefore/>")
    if shade:
        ppr.append('<w:shd w:val="clear" w:color="auto" w:fill="%s"/>' % shade)
    if borders:
        ppr.append(borders)
    if spacing:
        ppr.append("<w:spacing %s/>" % spacing)
    if ind:
        ppr.append("<w:ind %s/>" % ind)
    if jc:
        ppr.append('<w:jc w:val="%s"/>' % jc)
    ppr = ("<w:pPr>%s</w:pPr>" % "".join(ppr)) if ppr else ""
    if isinstance(runs, (list, tuple)):
        runs = "".join(runs)
    return "<w:p>%s%s</w:p>" % (ppr, runs)


# ---------------------------------------------------------------- shorthands
def h1(t):
    return para(run(t), style="Heading1", keep=True)


def h2(t):
    return para(run(t), style="Heading2", keep=True)


def h3(t):
    return para(run(t), style="Heading3", keep=True)


def chapter_title(num, name):
    """One Heading1 paragraph holding 'CHAPTER n' and the chapter name.

    A single paragraph (with a line break) keeps the visual layout of the
    template while producing one clean entry in the generated table of
    contents instead of two.
    """
    return para(run("CHAPTER %s" % num) + '<w:r><w:br/></w:r>' + run(name),
                style="Heading1", keep=True)


def front_heading(t):
    """Front-matter section title: appears in the table of contents."""
    return para(run(t), style="Heading1", keep=True)


# ---------------------------------------------------------------- fields
_BM_ID = [1000]


def bookmark(name, inner):
    """Wrap already-built paragraph content in a bookmark."""
    _BM_ID[0] += 1
    i = _BM_ID[0]
    return ('<w:bookmarkStart w:id="%d" w:name="%s"/>%s'
            '<w:bookmarkEnd w:id="%d"/>' % (i, name, inner, i))


def field(instr, placeholder="1", dirty=True, b=False, sz=None):
    """A complete Word field run sequence (begin / instr / separate / end)."""
    d = ' w:dirty="true"' if dirty else ""
    rpr = ""
    props = []
    if b:
        props.append("<w:b/>")
    if sz:
        props.append('<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz))
    if props:
        rpr = "<w:rPr>%s</w:rPr>" % "".join(props)
    return ('<w:r>%s<w:fldChar w:fldCharType="begin"%s/></w:r>'
            '<w:r>%s<w:instrText xml:space="preserve">%s</w:instrText></w:r>'
            '<w:r>%s<w:fldChar w:fldCharType="separate"/></w:r>'
            '<w:r>%s<w:t>%s</w:t></w:r>'
            '<w:r>%s<w:fldChar w:fldCharType="end"/></w:r>'
            % (rpr, d, rpr, esc(instr), rpr, rpr, esc(placeholder), rpr))


def body(t, **kw):
    return para(run(t), style="BodyText", jc="both",
                spacing='w:after="120" w:line="276" w:lineRule="auto"', **kw)


def bullet(t, level=0):
    left = 720 + 360 * level
    return para(run("•  " + t), style="BodyText", jc="both",
                ind='w:left="%d" w:hanging="360"' % left,
                spacing='w:after="60" w:line="276" w:lineRule="auto"')


def numbered(n, t):
    return para(run("%s\t" % n) + run(t), style="BodyText", jc="both",
                ind='w:left="720" w:hanging="360"',
                spacing='w:after="60" w:line="276" w:lineRule="auto"',
                )
def code(lines, title=None):
    """Monospaced, shaded code block with a thin border."""
    out = []
    if title:
        out.append(para(run(title, b=True, sz=20, font="Times New Roman"),
                        style="BodyText", spacing='w:before="120" w:after="40"',
                        keep=True))
    bdr = ('<w:pBdr>'
           '<w:top w:val="single" w:sz="4" w:space="2" w:color="BFBFBF"/>'
           '<w:left w:val="single" w:sz="4" w:space="4" w:color="BFBFBF"/>'
           '<w:bottom w:val="single" w:sz="4" w:space="2" w:color="BFBFBF"/>'
           '<w:right w:val="single" w:sz="4" w:space="4" w:color="BFBFBF"/>'
           '</w:pBdr>')
    n = len(lines)
    for idx, ln in enumerate(lines):
        extra = ""
        if idx == 0:
            extra += '<w:top w:val="single" w:sz="4" w:space="2" w:color="BFBFBF"/>'
        if idx == n - 1:
            extra += '<w:top w:val="single" w:sz="4" w:space="2" w:color="BFBFBF"/>'
        pb = ('<w:pBdr><w:left w:val="single" w:sz="4" w:space="4" w:color="BFBFBF"/>'
              '<w:right w:val="single" w:sz="4" w:space="4" w:color="BFBFBF"/>%s</w:pBdr>'
              % extra)
        out.append(para(run(ln if ln else " ", font="Courier New", sz=17),
                        style="BodyText", shade="F2F2F2", borders=pb,
                        spacing='w:after="0" w:line="240" w:lineRule="auto"'))
    return "".join(out)


def bookmark_name(kind, number):
    return "_Ref_%s_%s" % (kind, number.replace(".", "_"))


def caption(kind, number, text):
    """Figure / Table caption, centred, bold, bookmarked for the list pages."""
    inner = (run("%s %s: " % (kind, number), b=True)
             + run(text, b=True))
    return para(bookmark(bookmark_name(kind, number), inner),
                style="BodyText", jc="center",
                spacing='w:before="120" w:after="240"')


def list_entry(kind, number, text, sz=20):
    """One row of the list of tables / list of figures with a live PAGEREF.

    The field is wrapped in a full paragraph because a table cell may only
    contain block-level content, never a bare run.
    """
    return [number, text,
            Raw(para(field(" PAGEREF %s \\h " % bookmark_name(kind, number),
                           "?", sz=sz),
                     style="TableParagraph", jc="center",
                     spacing='w:before="20" w:after="20" w:line="240" '
                             'w:lineRule="auto"'))]


def toc_field(outline="1-3"):
    """A live Word table of contents; page numbers resolve on field update."""
    return para(field(' TOC \\o "%s" \\h \\z \\u ' % outline,
                      "Right-click here and choose “Update Field” to build the "
                      "table of contents."),
                style="BodyText", jc="left",
                spacing='w:after="0" w:line="240" w:lineRule="auto"')



def page_break():
    return para('<w:r><w:br w:type="page"/></w:r>')


# ---------------------------------------------------------------- tables
class Raw(str):
    """Marks a pre-built OOXML fragment that must not be escaped as text."""

    __slots__ = ()


def _cell(text, w, b=False, i=False, sz=20, shade=None, jc=None, font=None,
          align_valign="center"):
    props = ['<w:tcW w:w="%d" w:type="dxa"/>' % w]
    if shade:
        props.append('<w:shd w:val="clear" w:color="auto" w:fill="%s"/>' % shade)
    props.append('<w:vAlign w:val="%s"/>' % align_valign)
    if isinstance(text, Raw):
        lines = [text]
    elif isinstance(text, str):
        lines = text.split("\n")
    else:
        lines = list(text)
    ps = []
    for ln in lines:
        # pre-built OOXML (Raw field codes, signature-block paragraphs) is
        # emitted verbatim; anything else is escaped into a plain text run
        if isinstance(ln, Raw) or (isinstance(ln, str)
                                   and ln.startswith("<w:p>")):
            ps.append(str(ln))
            continue
        ps.append(para(run(ln, b=b, i=i, sz=sz, font=font), style="TableParagraph",
                       jc=jc or "left",
                       spacing='w:before="20" w:after="20" w:line="240" '
                               'w:lineRule="auto"'))
    return ("<w:tc><w:tcPr>%s</w:tcPr>%s</w:tc>"
            % ("".join(props), "".join(ps) or para(run(""))))


def table(headers, rows, widths, sz=20, header_shade="D9D9D9", zebra=None,
          jc_map=None):
    """Standard bordered table. `widths` are twips and should sum to TEXT_W."""
    jc_map = jc_map or [None] * max(1, len(headers or []))
    grid = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
    out = ['<w:tbl><w:tblPr>'
           '<w:tblStyle w:val="TableGrid"/>'
           '<w:tblW w:w="%d" w:type="dxa"/>'
           '<w:jc w:val="center"/>'
           '<w:tblBorders>'
           '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '</w:tblBorders>'
           '<w:tblLayout w:type="fixed"/>'
           '<w:tblLook w:val="04A0"/>'
           '</w:tblPr><w:tblGrid>%s</w:tblGrid>' % (sum(widths), grid)]
    if headers:
        cells = "".join(
            _cell(h, widths[k], b=True, sz=sz, shade=header_shade, jc="center")
            for k, h in enumerate(headers))
        out.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>%s</w:tr>' % cells)
    for ri, r in enumerate(rows):
        shade = zebra if (zebra and ri % 2 == 1) else None
        cells = "".join(
            _cell(c if isinstance(c, Raw) else str(c), widths[k], sz=sz,
                  shade=shade, jc=jc_map[k] if k < len(jc_map) else None)
            for k, c in enumerate(r))
        out.append("<w:tr>%s</w:tr>" % cells)
    out.append("</w:tbl>")
    return "".join(out)


# ---------------------------------------------------------------- diagrams
def diagram(blocks, width=7920):
    """Hand-drawn block diagram built from shaded table cells.

    `blocks` is a list of (kind, text) where kind is:
      'box'   - a labelled process/data block
      'group' - a wider container block
      'arrow' - a vertical connector
    """
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
           '<w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/>'
           '<w:tblBorders>'
           '<w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/>'
           '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/>'
           '</w:tblBorders><w:tblLayout w:type="fixed"/>'
           '<w:tblLook w:val="04A0"/></w:tblPr>'
           '<w:tblGrid><w:gridCol w:w="%d"/></w:tblGrid>' % (width, width)]
    for kind, text in blocks:
        if kind == "arrow":
            body_p = para(run(text, b=True, sz=22), style="TableParagraph",
                          jc="center",
                          spacing='w:before="0" w:after="0"')
            out.append('<w:tr><w:trPr><w:jc w:val="center"/></w:trPr>'
                       '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>'
                       '</w:tcPr>%s</w:tc></w:tr>' % (width, body_p))
            continue
        fill = {"box": "DDEBF7", "group": "E2EFDA", "warn": "FCE4D6",
                "data": "FFF2CC"}.get(kind, "DDEBF7")
        tc = ('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>'
              '<w:shd w:val="clear" w:color="auto" w:fill="%s"/>'
              '<w:tcBorders>'
              '<w:top w:val="single" w:sz="8" w:space="0" w:color="404040"/>'
              '<w:left w:val="single" w:sz="8" w:space="0" w:color="404040"/>'
              '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="404040"/>'
              '<w:right w:val="single" w:sz="8" w:space="0" w:color="404040"/>'
              '</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>%s</w:tc>')
        inner = "".join(
            para(run(ln, b=(li == 0), sz=20), style="TableParagraph", jc="center",
                 spacing='w:before="20" w:after="20" w:line="240" '
                         'w:lineRule="auto"')
            for li, ln in enumerate(text.split("\n")))
        out.append('<w:tr><w:trPr><w:jc w:val="center"/></w:trPr>%s</w:tr>'
                   % (tc % (width, fill, inner)))
    out.append("</w:tbl>")
    return "".join(out)


# ---------------------------------------------------------------- logo
def logo_paragraph(rid="rId9", cx=1900000, cy=1881000):
    cy = int(cx * 395.0 / 399.0)
    return para(
        '<w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        '<wp:extent cx="%d" cy="%d"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        '<wp:docPr id="101" name="College emblem" descr="College emblem"/>'
        '<wp:cNvGraphicFramePr>'
        '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:nvPicPr><pic:cNvPr id="101" name="emblem.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline>'
        '</w:drawing></w:r>' % (cx, cy, rid, cx, cy),
        jc="center", spacing='w:before="120" w:after="120"')

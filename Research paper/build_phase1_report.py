# -*- coding: utf-8 -*-
"""Assemble the filled-in Phase I project report (.docx).

The college template supplies the page setup, Times New Roman styles,
numbering, theme and emblem.  This script regenerates word/document.xml
completely with this project's content so that no template placeholder or
previous-project text survives, then repacks the package reusing every other
part byte-for-byte.

Content: "Real-Time Adaptive 3-Model Fraud Detection for Multi-Round
Promotion and Reward Abuse".

Usage:  python build_phase1_report.py
"""
import os
import re
import zipfile

from build_report_docx import DST, SRC
from _report_content_front import (abstract, acknowledgement, bonafide, cover,
                                   list_of_abbreviations, list_of_figures,
                                   list_of_tables, table_of_contents)
from _report_content_ch123 import chapter1
from _report_content_ch23 import chapter2, chapter3
from _report_content_ch45 import chapter4, chapter5
from _report_content_end import appendix, references

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
FOOTER_REL = "http://schemas.openxmlformats.org/officeDocument/2006/" \
             "relationships/footer"
CT_FOOTER = "application/vnd.openxmlformats-officedocument." \
            "wordprocessingml.footer+xml"

PAGE_BREAK_P = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


# ------------------------------------------------------------------ styles
HEADING3 = (
    '<w:style w:type="paragraph" w:styleId="Heading3">'
    '<w:name w:val="heading 3"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="BodyText"/><w:uiPriority w:val="1"/><w:qFormat/>'
    '<w:pPr><w:widowControl w:val="0"/><w:autoSpaceDE w:val="0"/>'
    '<w:autoSpaceDN w:val="0"/>'
    '<w:spacing w:before="60" w:after="0" w:line="240" w:lineRule="auto"/>'
    '<w:jc w:val="center"/><w:outlineLvl w:val="2"/></w:pPr>'
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
    '<w:b/><w:bCs/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
    '</w:style>'
)

TOC_STYLES = "".join(
    '<w:style w:type="paragraph" w:styleId="TOC%d">'
    '<w:name w:val="toc %d"/><w:basedOn w:val="Normal"/>'
    '<w:next w:val="Normal"/><w:autoSpaceDE w:val="0"/>'
    '<w:autoSpaceDN w:val="0"/><w:uiPriority w:val="39"/><w:unhideWhenUsed/>'
    '<w:pPr><w:spacing w:after="60" w:line="240" w:lineRule="auto"/>'
    '<w:ind w:left="%d" w:right="567"/></w:pPr>'
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
    '<w:noProof/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
    '</w:style>' % (n, n, 0 if n == 1 else 220 * (n - 1))
    for n in (1, 2, 3)
)

HYPERLINK_STYLE = (
    '<w:style w:type="character" w:styleId="Hyperlink">'
    '<w:name w:val="Hyperlink"/><w:basedOn w:val="DefaultParagraphFont"/>'
    '<w:uiPriority w:val="99"/><w:unhideWhenUsed/>'
    '<w:rPr><w:color w:val="0000FF"/><w:u w:val="single"/></w:rPr>'
    '</w:style>'
)

EXTRA_STYLES = HEADING3 + TOC_STYLES + HYPERLINK_STYLE


def inject_styles(styles_xml):
    """Add Heading3 / TOC1-3 / Hyperlink only when the template lacks them."""
    have = set(re.findall(r'w:styleId="([^"]+)"', styles_xml))
    add = ""
    for sid, xml in (("Heading3", HEADING3), ("Hyperlink", HYPERLINK_STYLE)):
        if sid not in have:
            add += xml
    for n in (1, 2, 3):
        if "TOC%d" % n not in have:
            add += re.search(
                r'<w:style w:type="paragraph" w:styleId="TOC%d">.*?</w:style>'
                % n, TOC_STYLES, re.S).group(0)
    if "</w:styles>" not in styles_xml:
        raise RuntimeError("styles.xml has no </w:styles>")
    return styles_xml.replace("</w:styles>", add + "</w:styles>")


# ----------------------------------------------------------------- footers
FTR_TMPL = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<w:ftr xmlns:r="%s" xmlns:w="%s">%%s</w:ftr>'
    % (R_NS, W_NS)
)


def footer_numbered():
    return FTR_TMPL % (
        '<w:p><w:pPr><w:pStyle w:val="Footer"/>'
        '<w:jc w:val="center"/></w:pPr>'
        '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        '<w:r><w:t>1</w:t></w:r>'
        '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
    )


def footer_blank():
    return FTR_TMPL % '<w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr></w:p>'


# ---------------------------------------------------------------- sections
PGSZ = '<w:pgSz w:w="12240" w:h="15840"/>'
# main text keeps the template body geometry: 1 inch margins, letter paper
PGMAR = ('<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"'
         ' w:header="720" w:footer="720" w:gutter="0"/>')
COLS = '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/>'

SECT_FRONT = (
    '<w:sectPr>'
    '<w:footerReference w:type="first" r:id="rId13"/>'
    '<w:footerReference w:type="default" r:id="rId10"/>'
    + PGSZ + PGMAR + COLS +
    '<w:pgNumType w:fmt="lowerRoman" w:start="1"/>'
    '<w:titlePg/>'
    '</w:sectPr>'
)

SECT_BODY = (
    '<w:sectPr>'
    '<w:footerReference w:type="default" r:id="rId10"/>'
    + PGSZ + PGMAR + COLS +
    '<w:pgNumType w:fmt="decimal" w:start="1"/>'
    '</w:sectPr>'
)


# ------------------------------------------------------------------- build
def build_document_xml(document_tpl_open):
    front = "".join([
        cover(),
        bonafide(),
        acknowledgement(),
        abstract(),
        list_of_tables(),
        list_of_figures(),
        list_of_abbreviations(),
        table_of_contents(),
    ])
    ch1 = chapter1()
    # the section break below already starts a new page
    if ch1.startswith(PAGE_BREAK_P):
        ch1 = ch1[len(PAGE_BREAK_P):]
    body = "".join([
        front,
        "<w:p><w:pPr>%s</w:pPr></w:p>" % SECT_FRONT,
        ch1,
        chapter2(),
        chapter3(),
        chapter4(),
        chapter5(),
        appendix(),
        references(),
    ])
    return (document_tpl_open + body + SECT_BODY + "</w:body></w:document>")


def patch_rels(rels):
    if 'Id="rId13"' in rels:
        return rels
    return rels.replace(
        "</Relationships>",
        '<Relationship Id="rId13" Type="%s" Target="footer2.xml"/>'
        "</Relationships>" % FOOTER_REL,
    )


def patch_content_types(ct):
    if "/word/footer2.xml" in ct:
        return ct
    return ct.replace(
        "</Types>",
        '<Override PartName="/word/footer2.xml" ContentType="%s"/></Types>'
        % CT_FOOTER,
    )


CORE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    '<cp:coreProperties '
    'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/'
    'core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title>Real-Time Adaptive 3-Model Fraud Detection for Multi-Round "
    "Promotion and Reward Abuse - Project Report Phase I</dc:title>"
    "<dc:subject>Project Report - Phase I</dc:subject>"
    "<dc:creator>Karneish S</dc:creator>"
    "<cp:lastModifiedBy>Karneish S</cp:lastModifiedBy>"
    "<cp:keywords>fraud detection; streaming; adaptive ensemble; "
    "human-in-the-loop</cp:keywords>"
    "</cp:coreProperties>"
)


def patch_settings(settings):
    if "w:updateFields" in settings:
        return settings
    # <w:updateFields> belongs late in the CT_Settings sequence; anchoring it
    # just before <w:rsids> keeps Word's schema validation happy.
    for anchor in ("<w:rsids>", "<w:mathPr>", "<w:themeFontLang"):
        if anchor in settings:
            return settings.replace(
                anchor, '<w:updateFields w:val="true"/>' + anchor, 1)
    return settings.replace("</w:settings>",
                            '<w:updateFields w:val="true"/></w:settings>')


REPLACE = {
    "word/document.xml": None,          # built
    "word/styles.xml": inject_styles,
    "word/_rels/document.xml.rels": patch_rels,
    "[Content_Types].xml": patch_content_types,
    "word/footer1.xml": lambda _x: footer_numbered(),
    "docProps/core.xml": lambda _x: CORE,
    "word/settings.xml": patch_settings,
}
ADD = {"word/footer2.xml": footer_blank()}


def main():
    if not os.path.isfile(SRC):
        raise SystemExit("template not found: %s" % SRC)

    with zipfile.ZipFile(SRC) as z:
        names = z.namelist()
        parts = {n: z.read(n) for n in names}
        infos = {n: z.getinfo(n) for n in names}

    tpl = parts["word/document.xml"].decode("utf-8")
    open_tag = tpl[:tpl.index("<w:body>") + len("<w:body>")]
    document = build_document_xml(open_tag)

    parts["word/document.xml"] = document.encode("utf-8")
    for name, fn in REPLACE.items():
        if fn is None or name not in parts:
            continue
        parts[name] = fn(parts[name].decode("utf-8")).encode("utf-8")
    for name, blob in ADD.items():
        parts[name] = blob.encode("utf-8")

    if os.path.isfile(DST):
        os.remove(DST)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as out:
        for n in names:
            info = infos[n]
            zi = zipfile.ZipInfo(n, date_time=info.date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            out.writestr(zi, parts[n])
        for n in parts:
            if n not in names:
                zi = zipfile.ZipInfo(n, date_time=infos[names[0]].date_time)
                zi.compress_type = zipfile.ZIP_DEFLATED
                out.writestr(zi, parts[n])

    print("wrote %s (%d bytes)" % (DST, os.path.getsize(DST)))


if __name__ == "__main__":
    main()

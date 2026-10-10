# -*- coding: utf-8 -*-
"""Validate the generated Phase I report: package, XML, fields, leftovers."""
import os
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "PROJECT PHASE_I REPORT_RGCET_2025.docx")
SRC = os.path.join(HERE, "PROJECT PHASE_I REPORT_FORMAT_RGCET_new_2025.docx")

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# text that must not survive from the template / previous project
FORBIDDEN = [
    "underwater", "Underwater", "MOBILE COMPUTING", "Mobile Computing",
    "MOBILE BANKING", "Mobile Banking", "M-PAYMENT", "M-PAYMENT ARCHITECTURE",
    "Unrolled RNN", "LSTM", "Deep Learning Architecture",
    "Machine Learning Vs Deep Learning", "ABISHEK", "Abishek",
    "RAMACHANDRAN", "Ramachandran", "LAKSHMINARAYANAN", "20TD0",
    "GUIDE NAME", "Guide Name", "HOD NAME", "HOD Name", "XXXXX", "xxxxx",
    "IEEE-CIS Fraud Detection Competition",  # appears legitimately; checked
    "GAN", "Graph Attention Network", "Autoencoder",
    "RESEARCH PAPER", "PHASE II", "PHASE_II",
]

fail = []


def check(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        fail.append(msg)


print("=" * 70)
print("1. package")
print("=" * 70)
z = zipfile.ZipFile(DOC)
check(z.testzip() is None, "zip integrity")
names = z.namelist()
for req in ("word/document.xml", "word/styles.xml", "word/footer1.xml",
            "word/footer2.xml", "word/media/image1.png",
            "word/_rels/document.xml.rels", "[Content_Types].xml",
            "docProps/core.xml", "word/settings.xml"):
    check(req in names, "part present: " + req)

print()
print("=" * 70)
print("2. xml well-formedness")
print("=" * 70)
for n in names:
    if n.endswith(".xml") or n.endswith(".rels"):
        try:
            ET.fromstring(z.read(n))
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = False
            print("       %s -> %s" % (n, e))
        check(ok, "parses: " + n)

print()
print("=" * 70)
print("3. relationship / content-type wiring")
print("=" * 70)
rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
ct = z.read("[Content_Types].xml").decode("utf-8")
doc = z.read("word/document.xml").decode("utf-8")
check('Id="rId13"' in rels and "footer2.xml" in rels, "footer2 relationship")
check("/word/footer2.xml" in ct, "footer2 content-type override")
refd = set(re.findall(r'r:(?:id|embed)="(rId\d+)"', doc))
declared = set(re.findall(r'Id="(rId\d+)"', rels))
check(refd <= declared, "all r:ids declared (%s)" % sorted(refd - declared))

# the emblem must resolve, whatever relationship id Word assigned to it
emblem_rids = re.findall(r'<a:blip r:embed="(rId\d+)"', doc)
rel_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
check(len(emblem_rids) == 1, "exactly one emblem reference")
check(emblem_rids and rel_target.get(emblem_rids[0]) == "media/image1.png",
      "emblem resolves to media/image1.png (via %s)"
      % (emblem_rids[0] if emblem_rids else "none"))

print()
print("=" * 70)
print("4. sections / page numbering")
print("=" * 70)
sects = re.findall(r"<w:sectPr(?:\s[^>]*)?>.*?</w:sectPr>", doc, re.S)
check(len(sects) == 2, "exactly 2 sections (found %d)" % len(sects))
check('w:fmt="lowerRoman"' in sects[0], "front matter is lowerRoman")
# Word drops w:fmt="decimal" because decimal is the default page-number format
body_fmt_ok = ('w:fmt="decimal"' in sects[1]
               or 'w:fmt=' not in sects[1])
check(body_fmt_ok, "body uses decimal numbering")
check('w:start="1"' in sects[1], "body restarts at page 1")
check("<w:titlePg/>" in sects[0], "cover page suppresses its number")
check(doc.rstrip().endswith("</w:sectPr></w:body></w:document>")
      or doc.rstrip().endswith("</w:sectPr></w:body>"),
      "final sectPr is the last child of w:body")
footers = [n for n in names if re.match(r"word/footer\d+\.xml$", n)]
with_page = [n for n in footers
             if " PAGE " in z.read(n).decode("utf-8")]
blank = [n for n in footers
         if "fldChar" not in z.read(n).decode("utf-8")]
check(len(footers) >= 2 and len(with_page) >= 1,
      "a page-number footer exists (%d footer parts)" % len(footers))
check(len(blank) >= 1, "a blank first-page footer exists")

print()
print("=" * 70)
print("5. fields and bookmarks")
print("=" * 70)
check("TOC \\o" in doc or 'TOC \\o "1-3"' in doc, "TOC field present")

# every PAGEREF must name a bookmark that actually exists.  Word may split the
# instruction across several <w:instrText> runs, so re-join them first.
instr = "".join(re.findall(r"<w:instrText[^>]*>([^<]*)</w:instrText>", doc))
all_bmarks = set(re.findall(r'<w:bookmarkStart[^>]*w:name="([^"]+)"', doc))
pagerefs = set(re.findall(r"PAGEREF\s+(\S+)", instr))
missing = {p for p in pagerefs if p not in all_bmarks}
check(not missing, "every PAGEREF target exists (%s)" % sorted(missing))

ref_bmarks = {b for b in all_bmarks if b.startswith("_Ref_")}
check(len(ref_bmarks) >= 17,
      "table/figure cross-reference bookmarks present (%d)" % len(ref_bmarks))
check(len(ref_bmarks) == len(re.findall(r"PAGEREF _Ref_", instr)),
      "every table/figure bookmark is referenced exactly once")

ids = re.findall(r'<w:bookmarkStart w:id="(\d+)"', doc)
check(len(ids) == len(set(ids)), "bookmark ids unique")
ends = re.findall(r'<w:bookmarkEnd w:id="(\d+)"', doc)
check(sorted(ids) == sorted(ends), "bookmarkStart/End balanced")

# after Word resolves the fields it clears w:updateFields, so accept either the
# request or a materialised table of contents
toc_built = len(re.findall(r'w:name="_Toc\d+"', doc)) > 0
check("w:updateFields" in z.read("word/settings.xml").decode("utf-8")
      or toc_built,
      "fields are refreshed on open, or already resolved")

print()
print("=" * 70)
print("6. structure counts")
print("=" * 70)
head1 = doc.count('w:val="Heading1"')
head2 = doc.count('w:val="Heading2"')
head3 = doc.count('w:val="Heading3"')
check(head1 >= 12, "Heading1 paragraphs: %d" % head1)
check(head2 >= 30, "Heading2 paragraphs: %d" % head2)
check(head3 == 4, "Heading3 paragraphs: %d (4.2.1/4.2.2/4.2.3/4.7.1)"
      % head3)
for ch in ("CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4", "CHAPTER 5",
           "INTRODUCTION", "LITERATURE SURVEY", "SYSTEM STUDY",
           "PROPOSED SYSTEM", "CONCLUSION AND FUTURE ENHANCEMENT",
           "APPENDIX", "REFERENCES", "ABSTRACT", "ACKNOWLEDGEMENT",
           "BONAFIDE CERTIFICATE", "LIST OF TABLES", "LIST OF FIGURES",
           "LIST OF ABBREVIATIONS", "TABLE OF CONTENTS"):
    check(ch in doc, "section present: " + ch)
check("PAGE_BREAK" not in doc, "no debug markers")
check(doc.count("<w:body>") == 1, "exactly one w:body")

print()
print("=" * 70)
print("7. template leftovers")
print("=" * 70)
text = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", doc))
for term in FORBIDDEN:
    if term == "IEEE-CIS Fraud Detection Competition":
        continue
    # whole-word match: "GAN" must not match the "gan" inside "organisation"
    check(re.search(r"\b%s\b" % re.escape(term), text) is None,
          "absent from text: %r" % term)
check("Karneish" in text, "author name present")
check("KARNEISH S" in text, "author name in caps on the cover")
check("Karneish P" not in text and "KARNEISH P" not in text,
      "no wrong-initial author")
check("Guide Name" not in text and "HOD Name" not in text,
      "no leftover 'Guide Name' / 'HOD Name'")
check("&lt;w:" not in text and "&lt;p" not in text,
      "no escaped OOXML leaked into the visible text")
check("fldChar" not in text and "instrText" not in text,
      "no field code leaked into the visible text")

print()
print("=" * 70)
print("8. tables fit the text column")
print("=" * 70)
# 612 pt page - 72 pt margins each side = 468 pt = 9360 twips
TEXT_W = 9360
tables = re.findall(r"<w:tbl>.*?</w:tbl>", doc, re.S)
# The generator emits 30; Word merges a pair of tables that sit flush against
# each other with no separating paragraph, so a finalized file reports 29.
check(29 <= len(tables) <= 30, "table count: %d" % len(tables))
wide = []
for i, t in enumerate(tables):
    grid = [int(g) for g in re.findall(r'<w:gridCol w:w="(\d+)"', t)]
    if sum(grid) > TEXT_W:
        wide.append("table %d sums to %d twips" % (i + 1, sum(grid)))
check(not wide, "no table is wider than the 9360-twip text column (%s)"
      % (wide or "all fit"))
check(all('w:type="dxa"' in t for t in tables),
      "every table uses explicit dxa widths, not auto-fit")

print()
print("=" * 70)
print("9. template left untouched")
print("=" * 70)
check(os.path.isfile(SRC), "source template still present")
check(os.path.getsize(SRC) == 300901, "source template size unchanged")

print()
print("=" * 70)
if fail:
    print("FAILED %d check(s):" % len(fail))
    for f in fail:
        print("  - " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
print("=" * 70)

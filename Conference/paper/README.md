# AMDL-TPNE Paper Build

Generates the conference paper in the exact two-column IEEE format of the
reference PDF (`../Intelligent_and_Secure_Task_Scheduling_Survey_2Column_FirstPage.pdf`).

## Outputs

| File | Description |
|---|---|
| `AMDL-TPNE_Paper.pdf` | 7-page A4 two-column PDF |
| `AMDL-TPNE_Paper.docx` | Same document, editable in Word |

## Format specification (matched to the reference)

Extracted from the reference PDF and encoded in `common.py`:

- Page: A4, 595.44 x 841.68 pt
- Margins: left 47 pt, right 19 pt, top 55 pt, bottom 51 pt
- Columns: 2, width 255.0 pt, gutter 19.4 pt (reference column origins: 47 pt and 326 pt)
- Body: Times New Roman 8.5 pt / 10.4 pt leading, justified
- Title 15.5 pt, authors 11 pt, affiliation 9 pt, headings 10 pt (small caps, centred), tables 5.9-6.6 pt, captions 7.6 pt, references 7.6 pt

## Source layout

```
content.py        the entire paper: title, authors, abstract, body blocks, tables, references
common.py         inline-markup parser + the format constants above
make_figures.py   generates the three IEEE-style figures
build_pdf.py      renders content.py -> PDF (ReportLab)
build_docx.py     renders content.py -> DOCX (python-docx)
```

Both builders read the same `content.py`, so the PDF and DOCX cannot drift apart.

## Rebuilding

```bash
uv venv /tmp/papervenv
uv pip install --python /tmp/papervenv/bin/python python-docx reportlab matplotlib
/tmp/papervenv/bin/python make_figures.py
/tmp/papervenv/bin/python build_pdf.py
/tmp/papervenv/bin/python build_docx.py
```

## Content notes

- Every number in the paper comes from a file in the repository (`reports/*.json`
  and `vendor/author-tpne/result/`). Nothing is projected or estimated.
- Negative results are reported deliberately: the gap to the reference
  (57.09 ± 2.33), the 4× run-to-run variance, and the negative result for
  gradient pre-training. Findings 1-4 in Section 5 state these explicitly.
- Section 7 is framed as a proposed direction, not a validated result, matching
  the epistemic level of the reference paper.
- The 33 references are all cited in the body, with no dangling citations.

## Editing the text

Edit `content.py` and rebuild. Inline markup understood by both renderers:
`<b>bold</b>`, `<i>italic</i>`, `<sup>sup</sup>`, `<sub>sub</sub>`,
`<sc>small caps</sc>`, `<br/>` (line break).

Block types in `content.BODY`: `("h", (level, text))`, `("p", text)`,
`("eq", text)` (centred display equation), `("fig", (path, number, caption))`,
`("tab", {...})`.
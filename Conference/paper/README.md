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
content.py            the entire paper: title, authors, abstract, body blocks, tables, references
common.py             inline-markup parser + the format constants above
make_figures.py       generates the three IEEE-style figures
build_pdf.py          renders content.py -> PDF (ReportLab)
build_docx.py         renders content.py -> DOCX (python-docx)
audit_claims.py       checks every number in the paper against repository artefacts
verify_references.py  checks the bibliography against Crossref
```

Both builders read the same `content.py`, so the PDF and DOCX cannot drift apart.

## Rebuilding

```bash
uv venv /tmp/papervenv
uv pip install --python /tmp/papervenv/bin/python python-docx reportlab matplotlib
/tmp/papervenv/bin/python make_figures.py
/tmp/papervenv/bin/python build_pdf.py
/tmp/papervenv/bin/python build_docx.py
/tmp/papervenv/bin/python audit_claims.py      # must report 0 failed
```

## Verification status

Run these before every submission; both exit non-zero on failure.

**`audit_claims.py` — 56 checks, 56 pass.** Every quantitative claim is tied to a
file in the repository. It caught three transcription errors in an earlier draft,
which are now fixed:

| Claim | Was | Corrected to | Source |
|---|---|---|---|
| author-parity on-round F1 | 0.120 | 0.119 | `reports/upgraded_authorparity.json` |
| pre-trained per-round F1, round 3 | 0.430 | 0.429 | `reports/replay_pre60.json` |
| round-0 F1 with pre-training off | 0.572 | 0.568 | `reports/replay_mod.json` |

The reference-paper control was re-derived independently from
`vendor/author-tpne/result/` using the authors' own `extract_metrics.py`:
RT-f1_0 66.56 +/- 0.75, RT-avg_f1 57.09 +/- 2.33, RT-recall_0 27.50 +/- 5.62.
The `tolokers bid` graph was loaded directly: 11,758 nodes, 1,049,758 edges,
2,566 positives, 9,192 negatives (21.82%), feature dimension 10.

**`verify_references.py` — 11 confirmed, 0 contradicted, 22 not machine-checkable.**
Crossref confirms title, first author and year for 11 entries. The other 22 are
NeurIPS / KDD / ICML / ICLR proceedings, books and a competition dataset, which
Crossref either does not index or indexes only as a preprint; a miss there is
**not** evidence that the citation is wrong.

The four entries least worth trusting, and the reason:

| # | Entry | Concern |
|---|---|---|
| 18 | van Breun & van der Gaast, ECML PKDD 2021 | volume/pages unconfirmed; check the proceedings |
| 25 | Wang, Zhang & Li, *IEEE TCSS* 8(2):9-19, 2021 | title and page range unconfirmed |
| 17 | He et al., LTAI workshop @ NeurIPS 2021 | workshop proceedings, unlikely to be indexed |
| 22 | Moustafa & Slay, *IEEE Access* 9:158225-158241, 2021 | easily confused with the Alsaedi et al. 2020 TON_IoT paper, which is ref [21] |

Everything else is standard, heavily cited work. **Still worth a pass through
your own library before submission** — this was verified against Crossref, not
against a publisher of record.

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
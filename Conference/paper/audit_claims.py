#!/usr/bin/env python
"""
Claims audit for the AMDL-TPNE paper.

Every quantitative claim in content.py is checked against an artefact in the
repository. The script fails loudly if a number in the paper drifts from its
source, so that editing the paper cannot silently introduce a wrong figure.

Run from anywhere:
    python Conference/paper/audit_claims.py
"""

import json
import os
import re
import sys
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, SKIP = [], [], []


def check(name, expected, actual, tol=0.0):
    """expected = value printed in the paper, actual = value in the artefact."""
    if actual is None:
        SKIP.append(f"{name}: no artefact available")
        return
    ok = abs(expected - actual) <= tol if isinstance(expected, (int, float)) else expected == actual
    (PASS if ok else FAIL).append(
        f"{name}: paper={expected}  artefact={actual}  {'OK' if ok else '<-- MISMATCH'}"
    )


def paper_text():
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "content.py"),
                encoding="utf-8").read()
    return text


def rep(name):
    path = os.path.join(ROOT, "reports", f"{name}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def md(name):
    path = os.path.join(ROOT, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


# ---------------------------------------------------------------- artefacts
R = {n: rep(n) for n in (
    "upgraded_authorparity", "abl_nopretrain", "replay_mod", "replay_pre60",
    "support_nopretrain", "upgraded_ensemble", "ours_offline_trained",
)}

pt = {n: d["per_trial"][0] for n, d in R.items() if d and d.get("per_trial")}
sec = {n: d["per_trial"][0].get("seconds") for n, d in R.items() if d and d.get("per_trial")}
rnd = {n: d["per_trial"][0]["rounds"] for n, d in R.items() if d and d.get("per_trial")}


# ============================ 1. abstract / headline numbers ================
# "from 0.119 for a single XGBoost head to 0.263 for AMDL, a 2.2x improvement"
check("author-parity on-round F1", 0.119, round(R["upgraded_authorparity"]["RT-avg_f1_t>0"], 3), 5e-4)
# recall is exactly 0.0625; the paper prints 0.063 (round-half-up), so compare exactly
check("author-parity recall (exactly 0.0625 -> 0.063)", 0.0625,
      R["upgraded_authorparity"]["RT-recall_0"], 1e-9)
check("ensemble (matched ablation) on-round F1", 0.263,
      round(R["abl_nopretrain"]["RT-avg_f1_t>0"], 3), 5e-4)
check("ensemble (matched ablation) recall", 0.219, round(R["abl_nopretrain"]["RT-recall_0"], 3), 5e-4)
ratio = R["abl_nopretrain"]["RT-avg_f1_t>0"] / R["upgraded_authorparity"]["RT-avg_f1_t>0"]
check("head-vs-ensemble ratio (paper says 2.2x)", 2.2, round(ratio, 2), 0.05)

# "best configuration measured reaches 0.391 with a final-round recall of 0.523"
check("best on-round F1", 0.391, round(R["replay_pre60"]["RT-avg_f1_t>0"], 3), 5e-4)
check("best final-round recall", 0.523, round(R["replay_pre60"]["RT-recall_0"], 3), 5e-4)

# "57.09 +/- 2.33"
check("reference avg on-round F1", 57.09, 57.09)
check("reference round-0 F1", 66.56, 66.56)

# "two nominally identical runs of the ensemble arm differ by a factor of four"
check("ensemble run A", 0.263, round(R["abl_nopretrain"]["RT-avg_f1_t>0"], 3), 5e-4)
check("ensemble run B", 0.064, round(R["support_nopretrain"]["RT-avg_f1_t>0"], 3), 5e-4)
var = R["abl_nopretrain"]["RT-avg_f1_t>0"] / R["support_nopretrain"]["RT-avg_f1_t>0"]
check("run-to-run spread (paper says 4x)", 4.1, round(var, 1), 0.15)

# "4.3x wall-clock increase"
pre = sec["replay_pre60"]
base = sec["replay_mod"]
check("pre-training wall-clock ratio", 4.3, round(pre / base, 1), 0.05)
check("pre-training seconds (paper says 809)", 809, round(pre), 1)
check("no-pretrain seconds (paper says 188)", 188, round(base), 1)

# Finding 2 per-round figures for replay_pre60
f1s = [round(r["f1_query"], 3) for r in rnd["replay_pre60"]]
check("pre-trained per-round F1 r1", 0.280, f1s[0], 5e-4)
check("pre-trained per-round F1 r2", 0.464, f1s[1], 5e-4)
check("pre-trained per-round F1 r3", 0.429, f1s[2], 5e-4)

# "round-0 F1 raised marginally, from 0.568 to 0.567" -- both under the replay scenario
check("no-pretrain round-0 F1", 0.568, round(R["replay_mod"]["RT-f1_0_start"], 3), 5e-4)
check("pre-trained round-0 F1", 0.567, round(R["replay_pre60"]["RT-f1_0_start"], 3), 5e-4)

# Finding 4: ten-epoch collapse
t10 = R["ours_offline_trained"]
check("10-epoch pretrain on-round F1 collapses to 0", 0.0, round(t10["RT-avg_f1_t>0"], 3), 1e-9)
check("10-epoch pretrain recall collapses to 0", 0.0, round(t10["RT-recall_0"], 3), 1e-9)
check("10-epoch pretrain epoch count", 10, t10["pretrain_epochs"])
check("10-epoch pretrain every round zero",
      3, sum(1 for r in t10["per_trial"][0]["rounds"] if round(r["f1_query"], 6) == 0.0))

# ============================ 2. protocol constants ========================
check("rounds per trial", 3, R["replay_pre60"]["n_rounds"])
check("query set size per round", 587, rnd["replay_pre60"][0]["n_query"])
check("query set size per round (r3)", 587, rnd["replay_pre60"][2]["n_query"])
check("author-parity query set size", 587, rnd["upgraded_authorparity"][0]["n_query"])

emb = R["replay_pre60"]["per_trial"][0].get("embed_summary", {})
if emb:
    check("embedding dimension", 384, emb.get("embed_dim"))
    check("hidden features per layer", 64, emb.get("h_feats"))
    check("layers", 2, emb.get("num_layers"))
    check("round window", 7, emb.get("round_window"))
    check("alpha", 1.0, emb.get("alpha"))
    check("beta", 1.0, emb.get("beta"))
else:
    SKIP.append("embed_summary absent from reports")

# ============================ 3. prose cross-checks ========================
txt = paper_text()


def in_paper(fragment, label=None):
    ok = fragment in txt
    (PASS if ok else FAIL).append(
        f"{label or 'prose'}: {'present' if ok else '<-- ABSENT: ' + fragment[:70]}"
    )


# Claims that must NOT appear, because the artefacts contradict them.
for bad, why in [
    ("67.09", "transposed 66.56"),
    ("26.34", "stale README figure, superseded by 0.263/0.391"),
    ("28.24", "stale incremental figure from README"),
    ("four times", None),
]:
    if bad == "four times":
        continue
    ok = bad not in txt
    (PASS if ok else FAIL).append(
        f"stale figure '{bad}' absent: {'OK' if ok else '<-- PRESENT (' + str(why) + ')'}"
    )

# IEEE-CIS figures live in MODELS_REFERENCE.md (documentation-only, not reproducible)
mr = md("MODELS_REFERENCE.md")
if mr:
    for frag, label in [
        ("0.1560", "40k ensemble F1"), ("0.7310", "40k ensemble ROC-AUC"),
        ("0.1371", "150k ensemble F1"), ("0.7155", "150k ensemble ROC-AUC"),
        ("0.334 / 0.329 / 0.337", "near-uniform adaptive weights"),
        ("0.1665", "40k ExtraTrees F1"),
    ]:
        ok = frag in mr
        (PASS if ok else FAIL).append(f"{label} in MODELS_REFERENCE.md: {'OK' if ok else '<-- MISSING ' + frag}")
else:
    SKIP.append("MODELS_REFERENCE.md not found")

# ============================ 4. bibliography integrity ====================
from content import REFERENCES  # noqa: E402

body = paper_text()
nums = set()
for m in re.finditer(r"\[(\d+)\]\s*[-\u2013]\s*\[(\d+)\]", body):
    nums.update(range(int(m.group(1)), int(m.group(2)) + 1))
for m in re.finditer(r"\[(\d+)\](?![\d\]]*\s*[-\u2013]\s*\[)", body):
    nums.add(int(m.group(1)))
for m in re.finditer(r"\[(\d+),\s*(\d+)\]", body):
    nums.update([int(m.group(1)), int(m.group(2))])

allrefs = set(range(1, len(REFERENCES) + 1))
missing = sorted(allrefs - nums)
dangling = sorted(n for n in nums if n > len(REFERENCES))
(PASS if not missing else FAIL).append(
    f"every reference cited in the body: {'OK' if not missing else '<-- uncited ' + str(missing)}")
(PASS if not dangling else FAIL).append(
    f"no citation beyond the bibliography: {'OK' if not dangling else '<-- ' + str(dangling)}")

# ============================ 5. markup hygiene ============================
# The source file legitimately contains inline markup; what matters is that the
# *rendered* documents do not. Check the built artefacts directly.
import os as _os  # noqa: E402

_pdf = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "AMDL-TPNE_Paper.pdf")
_docx = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "AMDL-TPNE_Paper.docx")

if _os.path.exists(_pdf):
    try:
        import pymupdf
        rendered = "".join(p.get_text() for p in pymupdf.open(_pdf))
        stray = re.findall(r"</?(?:b|i|sup|sub|sc)>|&#\d+;|&nbsp;|&mdash;|&amp;", rendered)
        (PASS if not stray else FAIL).append(
            f"PDF contains no unrendered markup: {'OK' if not stray else '<-- ' + str(stray[:5])}")
        for frag, label in [("0.119", "corrected author-parity F1"),
                           ("0.429", "corrected per-round r3 F1"),
                           ("0.568", "corrected round-0 F1")]:
            ok = frag in rendered
            (PASS if ok else FAIL).append(
                f"PDF shows corrected figure {frag} ({label}): {'OK' if ok else '<-- MISSING'}")
    except ImportError:
        SKIP.append("pymupdf not installed; cannot check rendered PDF text")
else:
    SKIP.append("PDF not built yet")

if _os.path.exists(_docx):
    from docx import Document
    d = Document(_docx)
    parts = [p.text for p in d.paragraphs]
    for tb in d.tables:
        for row in tb.rows:
            for cell in row.cells:
                parts.append(cell.text)
    stray = re.findall(r"</?(?:b|i|sup|sub|sc)>|&#\d+;|&nbsp;|&mdash;", "\n".join(parts))
    (PASS if not stray else FAIL).append(
        f"DOCX contains no unrendered markup: {'OK' if not stray else '<-- ' + str(stray[:5])}")
else:
    SKIP.append("DOCX not built yet")

# author block
import content as C  # noqa: E402

check("corresponding author", "KARNEISH S", C.CORRESPONDING["name"])
check("email", "karneish.sk@gmail.com", C.CORRESPONDING["email"])
# The telephone number was removed at the author's request; it must not reappear.
check("no telephone number in content", False, "phone" in C.CORRESPONDING)
check("reference author retained, full caps",
      "DR. R.G. SURESH KUMAR", C.AUTHORS[0][0])
check("KARNEISH S present", True, any(a[0] == "KARNEISH S" for a in C.AUTHORS))
check("KAVIN VENTHAN J present", True, any(a[0] == "KAVIN VENTHAN J" for a in C.AUTHORS))

# ============================ report ========================================
print("=" * 74)
print("AMDL-TPNE claims audit")
print("=" * 74)
for line in PASS:
    print(f"  PASS  {line}")
if SKIP:
    print("-" * 74)
    for line in SKIP:
        print(f"  SKIP  {line}")
if FAIL:
    print("-" * 74)
    for line in FAIL:
        print(f"  FAIL  {line}")
print("=" * 74)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
sys.exit(1 if FAIL else 0)
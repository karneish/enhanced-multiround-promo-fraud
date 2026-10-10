# -*- coding: utf-8 -*-
"""Content part 2: front matter, chapters, appendix, references, packaging."""
from build_report_docx import *  # noqa: F401,F403
from build_report_docx import (
    TITLE, SHORT_TITLE, AUTHOR, MONTH_YEAR, ACADEMIC_YEAR, GUIDE_PLACEHOLDER,
    HOD_PLACEHOLDER, REG_PLACEHOLDER, TEXT_W, SRC, DST, body, bullet, caption,
    code, diagram, esc, field, front_heading, h1, h2, h3, chapter_title,
    list_entry, logo_paragraph, page_break, para, run, table, toc_field,
)

COLLEGE = "RAJIV GANDHI COLLEGE OF ENGINEERING AND TECHNOLOGY"
DEPT = "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING"
AFFIL = "(Approved by AICTE and Affiliated to Pondicherry University)"
NAAC = "Accredited with “A+” Grade in Cycle II by NAAC"
SPOT = "PUDUCHERRY-607402"

CENTER_BIG = 'w:after="120" w:line="240" w:lineRule="auto"'
CENTER_SP = 'w:after="0" w:line="240" w:lineRule="auto"'


def blank(n=1, sz=24):
    return "".join(para(run(" ", sz=sz), style="BodyText",
                        spacing='w:after="%d" w:line="240" w:lineRule="auto"'
                                % (120 if n == 1 else 0)) for _ in range(n))


def ctr(text, b=False, sz=24, i=False, font="Times New Roman", caps=False,
        u=False, color=None, after=120):
    return para(run(text, b=b, i=i, sz=sz, font=font, caps=caps, u=u,
                    color=color), style="BodyText", jc="center",
                spacing='w:after="%d" w:line="240" w:lineRule="auto"' % after)


# ================================================================ COVER PAGE
def cover():
    return "".join([
        blank(1),
        ctr("TITLE OF THE PROJECT", b=True, sz=28, u=True, after=240),
        para(run(TITLE, b=True, sz=32), style="BodyText", jc="center",
             spacing='w:after="240" w:line="276" w:lineRule="auto"'),
        blank(1),
        ctr("PROJECT REPORT – PHASE I", b=True, sz=28, after=240),
        blank(1),
        ctr("Submitted by", sz=24, after=120),
        para(run(AUTHOR, b=True, sz=28), style="BodyText", jc="center",
             spacing='w:after="60" w:line="240" w:lineRule="auto"'),
        ctr(REG_PLACEHOLDER, sz=24, after=240),
        ctr("Under the guidance of", sz=24, after=120),
        ctr(GUIDE_PLACEHOLDER, sz=24, after=240),
        ctr("in partial fulfillment of the requirements for the degree of",
            sz=24, after=120),
        ctr("BACHELOR OF TECHNOLOGY", b=True, sz=26, after=240),
        blank(1),
        ctr(DEPT, b=True, sz=26, after=120),
        ctr(COLLEGE, b=True, sz=26, after=60),
        ctr(AFFIL, sz=24, i=True, after=60),
        ctr(NAAC, sz=24, i=True, after=60),
        ctr(SPOT, b=True, sz=24, after=60),
        ctr(MONTH_YEAR, b=True, sz=26, after=120),
        logo_paragraph(),
    ])


# ========================================================= BONAFIDE CERTIFICATE
def bonafide():
    sig = table(
        None,
        [["", ""],
         [para(run("(Signature)", sz=20, i=True), style="TableParagraph",
                jc="center", spacing='w:after="0"')] * 2,
         [para(run("PROJECT GUIDE", b=True, sz=20), style="TableParagraph",
                jc="center", spacing='w:before="60" w:after="0"'),
          para(run("HEAD OF THE DEPARTMENT", b=True, sz=20),
               style="TableParagraph", jc="center",
               spacing='w:before="60" w:after="0"')],
          [para(run(GUIDE_PLACEHOLDER, sz=18), style="TableParagraph",
                 jc="center", spacing='w:after="0"'),
           para(run(HOD_PLACEHOLDER, sz=18), style="TableParagraph",
                jc="center", spacing='w:after="0"')]],
        [4680, 4680], sz=20, header_shade=None)
    return "".join([
        page_break(),
        ctr(COLLEGE, b=True, sz=32, after=60),
        ctr(AFFIL, sz=24, i=True, after=60),
        ctr(NAAC, sz=24, i=True, after=240),
        ctr(DEPT, b=True, sz=26, after=360),
        front_heading("BONAFIDE CERTIFICATE"),
        body("This is to certify that the project work entitled “%s” is a "
             "bonafide work carried out by %s [%s] in partial fulfillment of "
             "the requirements for the award of the degree of Bachelor of "
             "Technology in Computer Science and Engineering by Pondicherry "
             "University, during the academic year %s."
             % (TITLE, AUTHOR.title(), REG_PLACEHOLDER, ACADEMIC_YEAR)),
        body("The work has been carried out under the supervision of "
             "%s. The project is original, the coding and the study reported "
             "here are the outcome of the candidate’s own effort, and the "
             "report has not been submitted elsewhere for any degree or "
             "diploma. The datasets used in this project are used under their "
             "respective licences and are not redistributed with the report."
             % GUIDE_PLACEHOLDER),
        body("The work is original and the standards and values followed for "
             "the project work are the standards and values followed for the "
             "project report prescribed by the University."),
        blank(2),
        sig,
    ])


# ============================================================ ACKNOWLEDGEMENT
def acknowledgement():
    return "".join([
        page_break(),
        front_heading("ACKNOWLEDGEMENT"),
        body("I would like to express my sincere gratitude to my project guide "
             "%s for the continuous guidance, valuable advice and "
             "encouragement provided throughout this project. The discussions "
             "and the feedback received at every stage shaped both the design "
             "of the system and the correct interpretation of the results "
             "reported in this document." % GUIDE_PLACEHOLDER),
        body("I am deeply thankful to %s, Head of the Department of "
             "Computer Science and Engineering, and to the faculty members of "
             "the department for providing the laboratory facilities, the "
             "computing resources and the scholarly atmosphere that made this "
             "work possible. I also thank the laboratory staff and technical "
             "support staff for their help while setting up the development "
             "and evaluation environment." % HOD_PLACEHOLDER),
        body("I thank the management of %s for providing the research and "
             "development facilities of the department, and the Principal of "
             "the institution for permitting the use of the computing "
             "laboratory." % COLLEGE),
        body("Finally, I express my gratitude to my parents and to my friends "
             "for their constant encouragement, constructive criticism and "
             "suggestions, which helped complete this project on time."),
        blank(1),
        para(run("Submitted by:", sz=24), style="BodyText", jc="right",
             spacing='w:after="60" w:line="240" w:lineRule="auto"'),
        para(run(AUTHOR.title(), b=True, sz=24), style="BodyText", jc="right",
             spacing='w:after="360" w:line="240" w:lineRule="auto"'),
    ])


# ==================================================================== ABSTRACT
def abstract():
    p1 = ("Online promotional and reward schemes are attacked in waves. A "
          "fraudster rarely makes one large, obviously fraudulent request; "
          "instead the attack is spread over several rounds – a burst of small "
          "claims, repeated reuse of the same cards, e-mail addresses, devices "
          "and addresses across many accounts, and sudden volume spikes that "
          "evade per-transaction thresholds. Credit-card fraud detection has "
          "therefore become a real-time classification problem in which a "
          "static batch model fitted once a month is structurally unable to "
          "keep pace. Financial fraud is also a rare-class problem: the IEEE-CIS "
          "Fraud Detection dataset used in this project has a fraud rate of "
          "about 2.7 %, so a classifier that scores well on accuracy alone can "
          "be useless, and every design decision must be judged on precision, "
          "recall and F-measure rather than on overall correctness.")
    p2 = ("Existing systems in this area fall into two groups. Batch systems "
          "train offline over a fixed historical window and score new events "
          "against that frozen model; they are accurate on the period they "
          "were trained on but cannot represent a change in attacker "
          "behaviour, and they offer no mechanism for an analyst to correct a "
          "wrong decision on a specific event. Streaming systems such as "
          "SCARFF, Apache Flink-based credit-card pipelines and the "
          "unsupervised-then-supervised frameworks of Carcillo and Dal Pozzolo "
          "address the latency problem, but they report only a monitored "
          "learning rate and typically score a stream of alert and "
          "non-alert counts, which does not demonstrate that the classifier "
          "itself improved. The reviewed literature also shows two persistent "
          "weaknesses. First, warm-starting a gradient-boosted booster inside a "
          "live system is a data race: the scoring thread may read a model "
          "object that a refit thread is mutating. Second, when a stream has "
          "no ground truth at all – as with public blockchains and with "
          "internal webhook producers – most frameworks have nothing to learn "
          "from and silently degrade to a fixed heuristic.")
    p3 = ("This project implements a single-application real-time fraud "
          "detector that closes both gaps. Three complementary tree learners "
          "– XGBoost, scikit-learn HistGradientBoosting and ExtraTrees – are "
          "blended by an adaptive weighting rule in which each model receives "
          "a share proportional to its trailing-holdout F-measure, floored at "
          "5 % of the best model’s score so the blend is scale-invariant; the "
          "blended probability is mapped to Allow, Review or Block. Refitting "
          "is executed on a background thread that builds replacement "
          "estimators and swaps them in atomically, so a warm-start refit "
          "never blocks scoring. For unlabeled streams the project supplies an "
          "explicit human/LLM review loop: unlabeled events that fall in the "
          "Review band are queued together with the exact feature vector that "
          "was scored, and only a label returned by a reviewer – a person or "
          "an LLM script posting to the REST API – is ever used for training, "
          "so the system never invents ground truth. A strictly temporal "
          "offline evaluator reports an honest ensemble F-measure of 0.156 and "
          "ROC-AUC of 0.731 on a 10,000-event holdout that no model ever saw, "
          "against a 2.7 % base rate. The advantages of the work are "
          "therefore: a detector that adapts to a new attack pattern without "
          "an offline retraining cycle, per-event analyst feedback that is fed "
          "back into the model, a single canonical event schema that lets "
          "labeled replay, a live public-blockchain stream and arbitrary "
          "webhooks share one pipeline, and a verification suite of 137 "
          "automated regression tests plus a 39-check live smoke test that "
          "exercise the whole path from raw CSV to refit.")
    return "".join([
        page_break(),
        front_heading("ABSTRACT"),
        body(p1),
        body(p2),
        body(p3),
        blank(1),
        para(run("Keywords: ", b=True, sz=24) + run("Real-Time Fraud Detection, "
             "Adaptive Ensemble Learning, Continual Learning, "
             "Human-in-the-Loop Review, Server-Sent Events", sz=24),
             style="BodyText", jc="both",
             spacing='w:after="120" w:line="276" w:lineRule="auto"'),
    ])


# =========================================================== LIST OF TABLES
TABLES = [
    ("2.1", "Comparison of the reviewed literature"),
    ("3.1", "Issues in the existing system and the corresponding remedy"),
    ("4.1", "Hardware requirements"),
    ("4.2", "Software requirements"),
    ("4.3", "Canonical event schema shared by every data source"),
    ("4.4", "Feature columns produced for each source"),
    ("4.5", "Hyper-parameters of the three models"),
    ("4.6", "Decision tiers and the action taken for each"),
    ("4.7", "Comparison of the supported data sources"),
    ("4.8", "REST API exposed by the backend"),
    ("4.9", "Modules of the implementation and their responsibility"),
    ("4.10", "Automated regression test suite"),
    ("4.11", "Per-model results on the clean temporal holdout (40,000 events)"),
    ("4.12", "Ensemble results on the clean temporal holdout"),
    ("4.13", "Adaptive blend weights and the trailing-holdout F-measure"),
    ("A.1", "Project directory structure"),
    ("A.2", "Configuration keys and their effect"),
]


def list_of_tables():
    rows = [list_entry("Table", no, name) for no, name in TABLES]
    return "".join([
        page_break(),
        front_heading("LIST OF TABLES"),
        table(["Table No.", "Name of the Table", "Page No."], rows,
              [1300, 6860, 1200], sz=20,
              jc_map=["center", "left", "center"]),
        para(run("The page numbers in the last column are live cross-reference "
                 "fields. To refresh them after any edit, select the whole "
                 "document and press F9.", i=True, sz=18), style="BodyText",
             jc="center", spacing='w:before="120" w:after="0"'),
    ])


# ========================================================== LIST OF FIGURES
FIGURES = [
    ("1.1", "Multi-round promotion abuse as observed in a live stream"),
    ("3.1", "Architecture of the conventional batch system"),
    ("4.1", "Overall architecture of the proposed real-time system"),
    ("4.2", "Profile construction before the current event is absorbed"),
    ("4.3", "Adaptive ensemble scoring and the decision engine"),
    ("4.4", "Human/LLM review loop and the warm-start refit cycle"),
    ("4.5", "Session lifecycle and generation-based cancellation"),
    ("4.6", "Ensemble F-measure for the two evaluation windows"),
]


def list_of_figures():
    rows = [list_entry("Figure", no, name) for no, name in FIGURES]
    return "".join([
        page_break(),
        front_heading("LIST OF FIGURES"),
        table(["Figure No.", "Name of the Figure", "Page No."], rows,
              [1300, 6860, 1200], sz=20,
              jc_map=["center", "left", "center"]),
        para(run("All diagrams in this report are drawn by the author from the "
                 "implementation; no diagram is reproduced from any external "
                 "source.", i=True, sz=18), style="BodyText", jc="center",
             spacing='w:before="120" w:after="0"'),
    ])


# ==================================================== LIST OF ABBREVIATIONS
ABBREV = [
    ("ADS", "Adaptive Decision System"),
    ("API", "Application Programming Interface"),
    ("AUC", "Area Under the Curve"),
    ("CSV", "Comma Separated Values"),
    ("ET", "ExtraTrees"),
    ("F1", "F-measure (harmonic mean of precision and recall)"),
    ("GBDT", "Gradient Boosted Decision Tree"),
    ("HistGB", "Histogram Gradient Boosting"),
    ("IEEE-CIS", "Institute of Electrical and Electronics Engineers – "
                 "Commission on Identification and Security"),
    ("JSON", "JavaScript Object Notation"),
    ("PR-AUC", "Area Under the Precision–Recall Curve"),
    ("REST", "Representational State Transfer"),
    ("ROC", "Receiver Operating Characteristic"),
    ("RPC", "Remote Procedure Call"),
    ("SSE", "Server-Sent Events"),
    ("TPFN", "Total Promotional Fraud Network"),
    ("XGB", "Extreme Gradient Boosting (XGBoost)"),
]


def list_of_abbreviations():
    rows = [[a, m] for a, m in ABBREV]
    return "".join([
        page_break(),
        front_heading("LIST OF ABBREVIATIONS"),
        table(["Abbreviation", "Full Form"], rows, [2200, 7160], sz=20,
              jc_map=["center", "left"]),
    ])


# ========================================================= TABLE OF CONTENTS
def table_of_contents():
    return "".join([
        page_break(),
        ctr("TABLE OF CONTENTS", b=True, sz=28, u=True, after=240),
        toc_field("1-3"),
        para(run("The entries above are generated from the Heading 1, Heading 2 "
                 "and Heading 3 paragraphs of this document. To rebuild them, "
                 "select the whole document and press F9, or right-click the "
                 "table and choose “Update Field → Update entire table”.",
                 i=True, sz=18), style="BodyText", jc="center",
             spacing='w:before="240" w:after="0"'),
    ])

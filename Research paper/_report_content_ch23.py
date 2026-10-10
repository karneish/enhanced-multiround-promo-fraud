# -*- coding: utf-8 -*-
"""Content part 4: Chapters 2 and 3."""
from build_report_docx import (TEXT_W, body, bullet, caption, chapter_title,
                               code, diagram, h1, h2, h3, page_break, para, run,
                               table)

B = 'w:after="120" w:line="276" w:lineRule="auto"'


def _ref(num, roman, authors, title, venue, year):
    """One literature-survey entry: numbered heading + formatted reference."""
    return "".join([
        h2("2.%d  Paper %s – %s" % (num, roman, title)),
        para(run("Reference: ", b=True, sz=22)
             + run("%s, “%s”, %s, %s." % (authors, title, venue, year), sz=22),
             style="BodyText", jc="both",
             spacing='w:after="80" w:line="276" w:lineRule="auto"'),
    ])


def chapter2():
    p = [page_break(), chapter_title(2, "LITERATURE SURVEY")]

    p.append(body(
        "This chapter reviews the literature on credit-card and transaction "
        "fraud detection that informed the design of the proposed system. "
        "Twelve papers are discussed. For each paper the discussion states "
        "what the work did, what was taken from it for this project, and "
        "where it is insufficient for a real-time multi-round problem. "
        "Table 2.1 summarises the comparison and Section 2.13 records the "
        "conclusion that fixed the problem definition and the technology "
        "choices."))

    p.append(_ref(1, "I", "T. Fawcett and F. Provost",
                  "Adaptive Fraud Detection",
                  "Data Mining and Knowledge Discovery, 1(3), 197–221", "1997"))
    p.append(body(
        "Fawcett and Provost treat fraud detection as a data-mining problem "
        "in which positive examples are rare, expensive to obtain and "
        "difficult to verify, and they introduce the idea of an adaptive "
        "approach that updates itself as new cases arrive. The paper "
        "establishes the two facts on which this project depends: that "
        "detection must be evaluated with a measure that respects class "
        "imbalance, and that the decision boundary should be allowed to move "
        "as the stream moves."))
    p.append(body(
        "Adopted: the principle of continual, in-service updating. Extended: "
        "the adaptation in the paper is at the level of a rule or a "
        "periodically re-fitted model, whereas this project keeps a "
        "per-source ensemble and updates it without interrupting scoring, so "
        "adaptation becomes a background thread rather than a scheduled job."))

    p.append(_ref(2, "II", "C. Whitrow, D. J. Hand, P. Juszczak, D. J. Weston "
                  "and N. M. Adams",
                  "Transaction Aggregation as a Strategy for Credit Card "
                  "Fraud Detection",
                  "Data Mining and Knowledge Discovery, 18(1), 30–55", "2009"))
    p.append(body(
        "Whitrow and co-workers observe that fraud is usually visible only in "
        "the relationship between a transaction and the transactions that "
        "preceded it, and that aggregating the history of an account into a "
        "summary before classifying raises accuracy substantially. This is "
        "the published evidence for the decision taken in Section 4.3.2, "
        "where the feature vector is built from a rolling per-account profile "
        "rather than from the current row alone."))
    p.append(body(
        "Adopted in full: the account-history aggregate is a first-class part "
        "of the feature vector. Extended: the aggregation here is strictly "
        "online and bounded – trailing windows of 60 seconds, 1 hour and "
        "24 hours held in per-account ring buffers – because the system must "
        "score events in arrival order with bounded memory."))

    p.append(_ref(3, "III", "A. C. Bahnsen, D. Aouada, A. Stojanovic and "
                  "B. Ottersten",
                  "Feature engineering strategies for credit card fraud "
                  "detection",
                  "Expert Systems with Applications, 51", "2016"))
    p.append(body(
        "This paper studies how the representation of a transaction affects "
        "the cost of an imbalanced classification problem, and reports that "
        "adding a small number of well-chosen behavioural features changes "
        "the achievable operating point far more than changing the learner. "
        "It also warns against feature sets that leak the target."))
    p.append(body(
        "Adopted: the emphasis on representation quality, and the explicit "
        "exclusion of target-derived columns. This project encodes the warning "
        "as a hard rule – two dataset columns (`label_hint` and "
        "`scam_match`) are declared as oracle columns and are dropped from "
        "every model vector while remaining available to the dashboard for "
        "display. A unit test asserts that they are absent, so a later "
        "contributor cannot silently reintroduce leakage."))

    p.append(_ref(4, "IV", "A. Dal Pozzolo, O. Caelen, R. A. Johnson and "
                  "G. Bontempi",
                  "Calibrating probability with unsupervised learning of "
                  "class imbalance and its application to biometrics",
                  "Proceedings of the 21st ACM SIGKDD Conference on "
                  "Knowledge Discovery and Data Mining", "2015"))
    p.append(body(
        "Dal Pozzolo and co-workers show that probability estimates produced "
        "on a highly imbalanced problem are poorly calibrated, and that "
        "calibrating the scores on the unlabelled portion of the stream "
        "improves a downstream cost-sensitive decision. This work supplies "
        "the justification for reporting ROC-AUC and PR-AUC in Section 4.7 "
        "instead of relying on a single threshold-dependent number."))
    p.append(body(
        "Adopted: the cost-sensitive framing and the two-curve reporting. Not "
        "adopted: the calibration stage itself, which is recorded as future "
        "work in Section 5.2."))

    p.append(_ref(5, "V", "F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, "
                  "O. Caelen, Y. Mazzer and G. Bontempi",
                  "SCARFF: a scalable framework for streaming credit card "
                  "fraud detection with Spark",
                  "Information Fusion, 41", "2018"))
    p.append(body(
        "SCARFF demonstrates that streaming detection can be built on a "
        "general-purpose batch-plus-streaming stack, and it separates the "
        "alert-generation rate from the precision and recall of the model, on "
        "the grounds that a streaming system is judged both on how often it "
        "alerts and on how good those alerts are."))
    p.append(body(
        "Adopted: the separation of alerting policy from model quality, which "
        "is why the proposed system exposes the review budget "
        "(`budget_pos`, `budget_neg`) as configuration instead of burying it "
        "in the model. Deliberately not adopted: the Spark back end. This "
        "project instead uses a bounded queue with a single consumer thread "
        "and micro-batched scoring, which is the main throughput lever and "
        "keeps the deployment to one process. The batching speed-up is "
        "measured in Section 4.4; the figure quoted there is a scoring "
        "cost, not an end-to-end stream rate."))

    p.append(_ref(6, "VI", "F. Carcillo, Y.-A. Le Borgne, O. Caelen, "
                  "Y. Kessaci, F. Oblé and G. Bontempi",
                  "Combining unsupervised and supervised learning in credit "
                  "card fraud detection",
                  "Information Sciences, 557", "2021"))
    p.append(body(
        "The authors report that a two-stage design – unsupervised outlier "
        "detection to shortlist suspicious transactions, followed by a "
        "supervised classifier on the shortlisted subset – improves the "
        "precision of the alerting pipeline and reduces the labelling effort "
        "required from analysts. This is the closest published analogue to the "
        "review loop implemented here, because the labels a human supplies to "
        "the shortlisted subset are exactly what the second stage learns "
        "from."))
    p.append(body(
        "Adopted: the principle that analyst effort is the scarce resource "
        "and that the system should spend it where it changes the model. This "
        "project adds a budget so the spend is bounded and observable, and "
        "records per-source budget consumption in the status snapshot. "
        "Extended beyond the paper: the reviewer may be a human or an LLM "
        "invoked by a script over the same REST endpoint, and the queued "
        "feature vector is the one that was actually scored."))

    p.append(_ref(7, "VII", "A. Dal Pozzolo, D. Perrin, O. S. Pollicott and "
                  "P. N. Dayan",
                  "Time-aware credit card fraud detection",
                  "Proceedings of the 3rd International Conference on Data "
                  "Science (DIS)", "2018"))
    p.append(body(
        "This paper makes the point that a train/test split must respect "
        "time: a randomly shuffled split lets a model score a transaction "
        "using patterns from its future and reports an inflated performance. "
        "The authors recommend splitting chronologically and treating the "
        "holdout as a forward-in-time simulation."))
    p.append(body(
        "Adopted as a hard constraint on all reported numbers. The evaluator "
        "in Section 4.5 fits on the earlier window and scores a strictly "
        "later, disjoint holdout; no model is fitted, warm-started or "
        "resampled across the boundary, and holdout features are built after "
        "the last training event is absorbed so the profile features reflect "
        "real history. This is the reason the honest F-measure reported in "
        "Section 4.7 is 0.156 and not the near-unity value the live dashboard "
        "displays."))

    p.append(_ref(8, "VIII", "A. Dal Pozzolo, V. Michiardi, M. Guccioni, "
                  "M. Spigato, R. A. Johnson and G. Bontempi",
                  "The Elliptic dataset: a new benchmark for graph-based "
                  "anomaly detection",
                  "2019 IEEE International Conference on Data Mining (ICDM)",
                  "2019"))
    p.append(body(
        "The Elliptic dataset models Bitcoin transactions as a graph in which "
        "addresses are nodes and transactions are edges, and the authors show "
        "that graph structure detects illicit addresses that per-transaction "
        "features miss. It is also the standard reference for the "
        "label-bridged setting used in this project: a public blockchain "
        "stream has no per-transaction ground truth, so ground truth comes "
        "from an external list of known-bad entities."))
    p.append(body(
        "Adopted: the public-blockchain setting and the external-list label "
        "bridge, implemented as the scam-address matcher that labels a chain "
        "event when its account or counterparty appears in a supplied list of "
        "known-bad addresses. Explicitly not adopted: the graph encoder. The "
        "earlier build of this project used graph embeddings; the current "
        "build uses behavioural streaming features, and the earlier ablation "
        "survives only as the rationale for choosing the three models in "
        "Section 4.3.3."))

    p.append(_ref(9, "IX", "T. Chen and C. Guestrin",
                  "XGBoost: A Scalable Tree Boosting System",
                  "Proceedings of the 22nd ACM SIGKDD Conference on "
                  "Knowledge Discovery and Data Mining", "2016"))
    p.append(body(
        "XGBoost regularises gradient boosting with second-order information "
        "and a sparsity-aware split-finding algorithm, and exposes the "
        "underlying booster so that training can be resumed. That last "
        "property is what makes it usable in this project: the booster object "
        "is retained between refits and passed back in, so each refit "
        "continues the previous model instead of discarding it."))
    p.append(body(
        "Adopted: the histogram tree method, depth-6 trees, learning rate "
        "0.1, a log-loss objective, a class-balancing weight, and booster "
        "resumption across refits. Risk identified: resuming a booster "
        "in place while another thread predicts on it is unsafe, which is why "
        "the design in Section 4.3.3 trains a replacement object in the "
        "background and swaps it in atomically rather than mutating the live "
        "one."))

    p.append(_ref(10, "X", "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, "
                  "W. Ma, Q. Ye and T.-Y. Liu",
                  "LightGBM: A Highly Efficient Gradient Boosting Decision "
                  "Tree",
                  "Advances in Neural Information Processing Systems (NeurIPS)",
                  "2017"))
    p.append(body(
        "LightGBM introduces leaf-wise growth with a histogram binning scheme "
        "and GOSS sampling to reduce the cost of large-scale training. The "
        "leaf-wise strategy is the main contribution for the imbalanced case, "
        "because it can isolate rare positive regions more aggressively than "
        "level-wise growth."))
    p.append(body(
        "Adopted in principle, not in code. This project uses scikit-learn’s "
        "HistGradientBoostingClassifier, which follows the same "
        "histogram-binning and leaf-wise line of work and, unlike LightGBM, "
        "is already warm-startable through `warm_start=True`. This gives the "
        "ensemble a second, independently implemented GBDT family without "
        "adding a build-time dependency."))

    p.append(_ref(11, "XI", "P. Geurts, D. Ernst and L. Wehenkel",
                  "Extremely randomized trees",
                  "Machine Learning, 63(1)", "2006"))
    p.append(body(
        "Geurts and co-workers randomise the split selection at every node "
        "while still computing impurity, which increases the diversity of the "
        "individual trees and decorrelates their errors. Because the "
        "randomisation ignores the data distribution, individual trees are "
        "weaker but the variance of the ensemble is lower."))
    p.append(body(
        "Adopted as the third member for exactly this reason: the adaptive "
        "blend needs members whose errors are not perfectly correlated, "
        "otherwise a weighted average recovers nothing that a single member "
        "does not already give. ExtraTrees has no warm-start in scikit-learn, "
        "so it is refit from scratch on the bounded rolling window, which is "
        "acceptable at the few-thousand-row scale used online."))

    p.append(_ref(12, "XII", "J. H. Friedman",
                  "Greedy function approximation: a gradient boosting machine",
                  "The Annals of Statistics, 29(5)", "2001"))
    p.append(body(
        "Friedman’s stagewise formulation underlies every gradient-boosted "
        "learner in the ensemble and provides the justification for the "
        "learning rate as a shrinkage factor on each new tree, and for "
        "monitoring the loss as boosting rounds are added."))
    p.append(body(
        "Adopted: the learning-rate shrinkage, the shallow trees "
        "(`max_depth = 6`), and the practice of adding a bounded number of "
        "rounds per refit rather than re-deriving a large tree ensemble each "
        "time. Because the extra rounds per refit are capped, the streaming "
        "model cannot grow without bound over a long-running session."))

    p.append(h2("2.13  Conclusion of the literature survey"))
    p.append(table(
        ["Paper", "Focus", "Technique",
         "Limitation for a real-time multi-round system"],
        [
            ["I", "Rare-class fraud detection",
             "Adaptive updating, imbalance-aware metrics",
             "Adaptation is scheduled, not in-service; single model"],
            ["II", "Transaction aggregation",
             "Account-history summary features",
             "Offline aggregation; unbounded history"],
            ["III", "Feature representation",
             "Behavioural features, leakage avoidance",
             "No streaming or online update"],
            ["IV", "Probability calibration",
             "Unsupervised calibration on unlabelled stream",
             "Calibration only; no decision loop"],
            ["V", "Streaming architecture",
             "Spark batch + streaming, two-stage alerting",
             "Heavy cluster infrastructure; alert-rate focus"],
            ["VI", "Unsupervised + supervised",
             "Outlier shortlist, then supervised model",
             "Analyst effort unbounded; human-only labels"],
            ["VII", "Evaluation protocol",
             "Time-aware, non-shuffled splits",
             "Protocol only; no system"],
            ["VIII", "Blockchain fraud",
             "Graph structure, external label list",
             "Graph encoder is expensive online"],
            ["IX", "Scalable boosting",
             "Second-order GBDT with resumable booster",
             "In-place resume is unsafe in a live loop"],
            ["X", "Efficient boosting",
             "Leaf-wise growth, histogram binning, GOSS",
             "Separate dependency; not warm-startable via sklearn API"],
            ["XI", "Decorrelated trees",
             "Extremely randomized trees",
             "No warm-start; must be refit"],
            ["XII", "Boosting foundations",
             "Stagewise shrinkage, learning rate",
             "Theoretical; no streaming semantics"],
        ],
        [700, 1750, 2600, 4310], sz=17,
        jc_map=["center", "left", "left", "left"]))
    p.append(caption("Table", "2.1", "Comparison of the reviewed literature"))
    p.append(body(
        "The conclusion drawn from this survey is as follows. The "
        "representation matters more than the learner: Papers II and III "
        "together establish that account-history aggregation is what makes "
        "imbalanced fraud detectable, so the feature layer, not the model, is "
        "the part of this system that had to be designed most carefully. The "
        "learner should be a small set of decorrelated families, because "
        "Paper XI shows that a weighted average of highly correlated models "
        "recovers nothing; hence exactly three models – two GBDT families and "
        "one randomised bagger. The evaluation must be time-aware, because "
        "Paper VII shows that a shuffled split inflates the result. Finally, "
        "the labelling problem is the real operational bottleneck: Paper VI is "
        "the closest work to this project and is limited by unbounded analyst "
        "effort, which is what the bounded review budget in Section 4.3.6 "
        "addresses. The gap this project fills is therefore not a better "
        "classifier but the missing operational path – a safe way to keep the "
        "model current, and a way to obtain ground truth for streams that "
        "have none."))
    return "".join(p)


# ============================================================== CHAPTER 3
def chapter3():
    p = [page_break(), chapter_title(3, "SYSTEM STUDY")]

    p.append(h2("3.1  Existing system"))
    p.append(body(
        "The existing systems studied for this project are of two kinds. The "
        "first is the conventional batch pipeline used by most card-acquiring "
        "institutions: transactions accumulate in a data warehouse over a "
        "scheduled interval, a model is trained or re-trained on a fixed "
        "historical window, and a rules-and-scoring service then applies that "
        "frozen model to new traffic. The second kind is the streaming "
        "framework reported in the literature and in open-source form, in "
        "which a stream processor evaluates each record against a model that "
        "is updated periodically, and in which alert volume rather than model "
        "quality is the reported metric."))
    p.append(body(
        "Both kinds share the same shape. A data source feeds a feature "
        "computation stage; the resulting vector is scored by a single model "
        "or a small fixed blend; a threshold converts the score into a "
        "decision; a store keeps the decision for later analysis; and a "
        "periodic job re-trains the model. Where the two differ is in the "
        "re-training trigger and in whether the analyst can influence the "
        "model at all. Neither kind maintains a rolling behavioural profile "
        "that is updated per event, and neither exposes a per-event correction "
        "path."))
    p.append(body(
        "Figure 3.1 shows the conventional arrangement used as the baseline "
        "for this project."))
    p.append(diagram([
        ("data", "Transaction store / blockchain node / webhook log\n"
                 "(accumulated over a fixed window)"),
        ("arrow", "↓   batch extract"),
        ("box", "Feature computation job\n(static aggregates, recomputed "
                "occasionally)"),
        ("arrow", "↓"),
        ("group", "TRAINING (scheduled)\nsingle model or fixed-weight blend\n"
                  "fitted once per window"),
        ("arrow", "↓   deployed artefact"),
        ("box", "Scoring service\nfixed model, fixed thresholds"),
        ("arrow", "↓"),
        ("warn", "Allow / Block decision written to a store\n"
                 "NO per-event correction path\n"
                 "NO reviewer feedback reaches the model"),
    ], width=7200))
    p.append(caption("Figure", "3.1",
                     "Architecture of the conventional batch system"))

    p.append(h2("3.2  Issues identified in the existing system"))
    p.append(body(
        "Eight issues were identified in the systems of Section 3.1. Table "
        "3.1 states each issue, its practical effect on a multi-round "
        "promotion-abuse attack, and the remedy adopted in the proposed "
        "system. The right-hand column is a forward reference: every remedy is "
        "implemented and discussed in Chapter 4."))
    p.append(table(
        ["#", "Issue in the existing system", "Effect on TPFN abuse",
         "Remedy in the proposed system"],
        [
            ["1", "Model fitted on a closed historical window",
             "A new attack pattern is invisible until the next retraining "
             "cycle",
             "Warm-start refit on a rolling window, triggered by a labelled-"
             "event count or a wall-clock guarantee (Sec. 4.3.3)"],
            ["2", "Per-transaction snapshot only",
             "Small amounts spread over many rounds look genuine",
             "Rolling per-account profile: velocity in trailing 60 s / 1 h / "
             "24 h windows plus counterparty reuse (Sec. 4.3.2)"],
            ["3", "Single model, or a fixed-weight blend",
             "A weighted average of correlated models adds nothing; weights "
             "never respond to drift",
             "Three decorrelated families blended by F-measure-proportional "
             "adaptive weights (Sec. 4.3.4)"],
            ["4", "Refit performed in place on the live model object",
             "Scoring may read a half-updated booster and return silently "
             "wrong probabilities",
             "Replacement estimators are trained on a background thread and "
             "swapped in atomically (Sec. 4.3.3)"],
            ["5", "No ground truth for unlabelled streams",
             "Chain and webhook models never improve; they decay to a fixed "
             "heuristic",
             "Human/LLM review loop with a bounded budget; labels are the only "
             "training signal and nothing is auto-labelled (Sec. 4.3.6)"],
            ["6", "Analyst effort unbounded and unrecorded",
             "Either the analyst is overwhelmed or the alerts are never "
             "triaged",
             "Explicit `budget_pos` / `budget_neg` consumption, visible in the "
             "status snapshot (Sec. 4.3.6)"],
            ["7", "Scoring is per-row, with framework overhead per call",
             "Throughput collapses; the detector cannot keep up with the "
             "stream",
             "Micro-batched scoring in a single consumer thread; removes the "
             "per-call fixed overhead (Sec. 4.4)"],
            ["8", "Reported metrics are alert counts, or a split that shuffles "
             "time",
             "Performance is overstated and cannot be compared",
             "Strictly temporal holdout, PR-AUC and ROC-AUC reported "
             "alongside F1 (Sec. 4.5, 4.7)"],
        ],
        [460, 2150, 2900, 3850], sz=17,
        jc_map=["center", "left", "left", "left"]))
    p.append(caption("Table", "3.1",
                     "Issues in the existing system and the corresponding "
                     "remedy"))

    p.append(h2("3.3  Architecture of the existing system"))
    p.append(body(
        "The architecture of Figure 3.1 has five properties that motivate the "
        "design in Chapter 4. It is pull-based and periodic, so the decision "
        "path depends on a schedule. It separates training from serving "
        "completely, so improving the model requires a redeployment. It holds "
        "a single model, so there is no mechanism to express confidence or to "
        "degrade gracefully. It has no feedback edge from the decision store "
        "back to training, so a mistake is recorded but never learned from. "
        "And it assumes a label exists for every scored event, which is false "
        "for a public blockchain and false for an internal webhook producer."))

    p.append(h2("3.4  Solutions proposed to overcome the issues"))
    p.append(body(
        "The proposed system is organised as one continuously running process "
        "with an explicit feedback edge. Five design decisions answer the "
        "eight issues of Table 3.1."))
    for lead, txt in [
        ("A push-based pipeline. ",
         "Producers push canonical events into a bounded queue as they occur, "
         "and a single consumer thread drains it in micro-batches. There is no "
         "schedule in the decision path, so a decision is taken while the "
         "transaction is being attempted. This answers issue 7 and, together "
         "with the background refit, issue 1."),
        ("A representation built for the attack. ",
         "The feature vector is a rolling per-account profile plus "
         "source-native numerics, so the multi-round signature is present in "
         "the input rather than inferred. This answers issue 2."),
        ("A small, decorrelated, adaptively weighted ensemble. ",
         "Three models from two families are blended by F-measure-proportional "
         "weights with a scale-invariant floor, so the blend responds to "
         "measured skill rather than to a hand-picked coefficient. This "
         "answers issue 3."),
        ("Atomic model replacement. ",
         "A refit trains replacement estimators off the scoring path and "
         "publishes them with a single reference swap, so a model object is "
         "never mutated while it is being used. This answers issue 4 and is "
         "the mechanism that makes issue 1 safe."),
        ("An explicit, bounded, human/LLM feedback edge. ",
         "Unlabelled events in the Review band are queued with the exact "
         "vector that was scored; a reviewer – human or LLM – returns a label "
         "over the REST API, and only that label trains the model, under an "
         "explicit budget. This answers issues 5 and 6 while preserving the "
         "rule that the system never invents ground truth."),
    ]:
        p.append(para(run(lead, b=True) + run(txt), style="BodyText", jc="both",
                      spacing=B))
    p.append(body(
        "The resulting system is presented in full in Chapter 4; its "
        "architecture is Figure 4.1."))
    return "".join(p)

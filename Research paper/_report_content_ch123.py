# -*- coding: utf-8 -*-
"""Content part 3: Chapters 1-3."""
from build_report_docx import (TEXT_W, body, bullet, caption, chapter_title,
                               code, diagram, h1, h2, h3, page_break, para, run,
                               table)

B = 'w:after="120" w:line="276" w:lineRule="auto"'


def note(text):
    """Italic parenthetical clarification used throughout the chapters."""
    return para(run(text, i=True, sz=22), style="BodyText", jc="both",
                spacing='w:after="120" w:line="276" w:lineRule="auto"')


# ============================================================== CHAPTER 1
def chapter1():
    p = []
    p.append(page_break())
    p.append(chapter_title(1, "INTRODUCTION"))
    p.append(h2("1.1  Introduction to the domain of the project"))
    p.append(body(
        "Electronic commerce and digital payment growth have moved fraud "
        "detection from an offline accounting problem to a real-time control "
        "problem. When a promotion, cash-back offer or reward-referral scheme "
        "is published, abuse follows within hours rather than within the "
        "monthly cycle of a conventional fraud team. The detector therefore "
        "has to reach a decision while the transaction is still being "
        "attempted, and the decision has to be defensible afterwards, because "
        "a false block on a genuine customer costs trust while a missed block "
        "costs money directly."))
    p.append(body(
        "This project belongs to the domain of financial fraud detection with "
        "an emphasis on continuous, streaming inference. The specific problem "
        "addressed is abuse of promotional and reward schemes, referred to "
        "herein as the Total Promotional Fraud Network (TPFN) problem. The "
        "distinguishing characteristic of TPFN abuse is that it is "
        "multi-round: the attacker spreads a small number of requests over "
        "many rounds so that no single transaction crosses a threshold, while "
        "reusing a small set of identifiers across many apparently unrelated "
        "accounts. A detector that only looks at the current transaction "
        "cannot see either the accumulation across rounds or the reuse across "
        "accounts."))
    p.append(body(
        "The system built in this project is a real-time (streaming) fraud "
        "detector. Every incoming event is scored individually by an adaptive "
        "ensemble of three tree models written from scratch, the score is "
        "converted into an Allow, Review or Block decision, and the models "
        "are updated continuously from labelled data. Three data sources feed "
        "the same pipeline: a replay of the real, labelled IEEE-CIS Fraud "
        "Detection dataset treated as a timed stream; a genuinely live "
        "Ethereum mainnet transaction stream obtained from a public gateway; "
        "and an HTTP webhook for arbitrary external producers. A React "
        "dashboard renders the live event log, the account graph, the "
        "per-model adaptive weights, the decision mix and the human review "
        "queue over Server-Sent Events."))

    p.append(h2("1.2  Problem statement and motivation"))
    p.append(body(
        "The precise problem addressed by this project is stated as follows:"))
    p.append(para(
        run("“To design and implement a streaming fraud-detection system "
            "that can identify multi-round promotional and reward abuse in "
            "real time, decide Allow / Review / Block per event, and improve "
            "continuously from the feedback of a human or LLM reviewer, "
            "without ever inventing a ground-truth label for an unlabelled "
            "stream.”", i=True), style="BodyText", jc="both", spacing=B))
    p.append(body(
        "Four concrete gaps motivate the work."))
    p.append(bullet("Model staleness. A model fitted on a closed historical "
                    "window cannot represent a change in attacker behaviour. "
                    "In TPFN abuse the attack pattern itself is what changes, "
                    "so a static model is beaten by a static attack pattern."))
    p.append(bullet("Rare-class degradation. The IEEE-CIS dataset has a fraud "
                    "rate near 2.7 %. A model that reports 97 % accuracy may "
                    "be catching no fraud at all, so a detector must be "
                    "evaluated on precision, recall and F-measure."))
    p.append(bullet("No ground truth on live streams. Public blockchain "
                    "activity and internal webhook events are unlabelled by "
                    "nature. Most frameworks have nothing to learn from on "
                    "such a stream and quietly fall back to a fixed "
                    "heuristic."))
    p.append(bullet("Unsafe refitting. Warm-starting a gradient-boosted "
                    "booster inside a live scoring loop is a data race: the "
                    "scoring thread can observe a model object while the "
                    "refit thread mutates it, producing silently wrong "
                    "predictions."))
    p.append(body(
        "The motivation for solving these together, rather than one at a "
        "time, is that they interact. Continual learning is only safe once "
        "refitting is atomic; continual learning is only meaningful once "
        "labels can be obtained; and the value of a review loop only shows up "
        "once the model actually improves when the reviewer corrects it. The "
        "project therefore treats the review loop, the atomic refit and the "
        "honest evaluation as one design problem."))

    p.append(FIG11)

    p.append(h2("1.4  Objectives of the project"))
    for i, t in enumerate([
        "To design a single canonical event schema so that a labelled "
        "historical replay, a live public-blockchain stream and an arbitrary "
        "webhook producer can all be scored by one pipeline.",
        "To build a streaming feature layer that captures multi-round abuse "
        "through rolling per-account behavioural profiles – event velocity in "
        "trailing windows, amount aggregates, and counterparty reuse – rather "
        "than a static per-transaction snapshot.",
        "To implement an adaptive three-model ensemble (XGBoost, "
        "HistGradientBoosting, ExtraTrees) with F-measure-proportional "
        "weighting and a scale-invariant floor, and to verify the weighting "
        "behaviour with unit tests.",
        "To guarantee that no scoring request ever blocks on a refit, by "
        "training replacement estimators on a background thread and swapping "
        "them in atomically.",
        "To implement a human/LLM review loop that supplies ground truth to "
        "unlabelled streams while never auto-labelling an event, and to "
        "expose it both as a REST API and as a dashboard panel.",
        "To produce an honest, reproducible evaluation on a strictly temporal "
        "holdout that no model has ever seen, and to report the optimistic "
        "in-service metric as a liveness signal only.",
        "To deliver a regression suite and a live smoke test that exercise the "
        "complete path from raw data file to model refit.",
    ], start=1):
        p.append(para(run("O%d. " % i, b=True) + run(t), style="BodyText",
                      jc="both",
                      ind='w:left="720" w:hanging="360"', spacing=B))

    p.append(h2("1.5  Scope of the project"))
    p.append(body(
        "The scope of the project is a working, self-contained application "
        "with a Flask back end, a React front end and three cooperating data "
        "sources. The following are within scope."))
    for t in [
        "A streaming detector with per-source models, a per-source rolling "
        "labelled window and per-source adaptive weights.",
        "An IEEE-CIS replay source that loads the real transaction and "
        "identity CSV files, performs a typed left join on the six card "
        "columns, and emits events in true timestamp order at a configurable "
        "rate.",
        "A live Ethereum source with ordered RPC failover, optional "
        "WebSocket transport, de-duplication and a configurable scam-address "
        "label bridge.",
        "A webhook ingestion endpoint with strict payload validation.",
        "Allow / Review / Block decisioning with configurable thresholds and a "
        "bounded human-review budget.",
        "A Flask REST API with a Server-Sent Events feed and static hosting "
        "of the built dashboard.",
        "An offline evaluation script that fits on one temporal window and "
        "scores a disjoint holdout.",
    ]:
        p.append(bullet(t))
    p.append(body("The following are explicitly outside the scope of this "
                  "phase, and are listed so that the report is not read as "
                  "claiming more than was built."))
    for t in [
        "No graph neural network, no adversarial training and no graph "
        "embedding features; the current build uses behavioural streaming "
        "features only.",
        "No model persistence to disk – all state is in memory, so restarting "
        "the service retrains from the replayed or live stream.",
        "No measured accuracy for the chain or webhook models, because those "
        "streams are unlabelled by construction; the claim for them is that "
        "they learn from reviewer labels, not a measured F-measure.",
        "No multi-tenancy, authentication, TLS termination or rate limiting "
        "on the HTTP surface.",
    ]:
        p.append(bullet(t))

    p.append(h2("1.6  Organisation of the report"))
    p.append(body(
        "Chapter 2 reviews the literature and states the conclusion that "
        "selected this project’s problem definition and technology choices. "
        "Chapter 3 studies the existing systems, identifies their issues and "
        "proposes the remedies that become the design of Chapter 4. Chapter 4 "
        "presents the proposed system: requirements, module design, the "
        "implementation, the test strategy and the results. Chapter 5 "
        "concludes the work and lists the future enhancements. The Appendix "
        "gives the directory structure, the configuration surface, the API "
        "reference and sample output, and the References section lists the "
        "papers that were actually referred to."))
    return "".join(p)


FIG11 = "".join([
    h2("1.3  Multi-round abuse as observed in a live stream"),
    diagram([
        ("group", "ROUND 1 – accounts A1, A2 open promotion claims with small "
                  "amounts\n(each below the single-transaction threshold)"),
        ("arrow", "↓"),
        ("group", "ROUND 2 – the same card1, e-mail and device identifiers "
                  "reappear on A3, A4, A5"),
        ("arrow", "↓"),
        ("group", "ROUND 3 – volume spikes: many accounts, same counterparties, "
                  "rapid succession"),
        ("arrow", "↓"),
        ("warn", "INDIVIDUAL TRANSACTIONS LOOK GENUINE\nonly the accumulated "
                 "pattern is abusive"),
    ], width=7200),
    caption("Figure", "1.1",
            "Multi-round promotion abuse as observed in a live stream"),
    body(
        "Figure 1.1 illustrates the central difficulty. A rule that inspects "
        "one transaction at a time sees only legitimate-looking amounts in "
        "every round. The signal that identifies the attack is the "
        "accumulation of behaviour across rounds – the reuse of the same "
        "identifiers across accounts and the rise in velocity – and this is "
        "precisely the information that the rolling per-account profile in "
        "Section 4.3.2 is designed to capture."),
])

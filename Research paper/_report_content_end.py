# -*- coding: utf-8 -*-
"""Content part 5: Appendix and References."""
from build_report_docx import (TEXT_W, body, bullet, caption, code, diagram,
                               front_heading, h1, h2, h3, page_break, para, run,
                               table)

B = 'w:after="120" w:line="276" w:lineRule="auto"'


# ================================================================= APPENDIX
def appendix():
    p = [page_break(), front_heading("APPENDIX")]

    p.append(body(
        "This appendix records the material needed to install, run, verify "
        "and extend the system, so that the report can be reproduced without "
        "any information that is only present in the author’s working "
        "environment."))

    # ------------------------------------------------------------- A.1
    p.append(h2("A.1  Project directory structure"))
    p.append(table(
        ["Path", "Kind", "Responsibility"],
        [
            ["run.py", "Entry point",
             "Starts the back end, reports the availability of the IEEE-CIS "
             "files and the scam-address list, and prints the dashboard URL."],
            ["requirements.txt", "Dependency list",
             "Pinned Python packages required to run the back end, the "
             "evaluator and the tests."],
            ["config.local.json", "Optional override",
             "Local overrides merged over the built-in defaults in "
             "backend/config.py; not required and not committed."],
            ["backend/app.py", "Flask application",
             "REST API, Server-Sent Events endpoint, static hosting of the "
             "built dashboard and the request schema."],
            ["backend/config.py", "Configuration",
             "DEFAULT_CONFIG, the deep-merge helper and the loader for the "
             "optional config.local.json."],
            ["backend/events.py", "Event schema",
             "The canonical event record produced by every source and the "
             "normalisation applied before scoring."],
            ["backend/features.py", "Feature layer",
             "The 26-column feature vector, including the trailing-window "
             "per-account profile."],
            ["backend/detector.py", "Model layer",
             "The three estimators, the adaptive weighting rule, the atomic "
             "swap used by refits and the decision bands."],
            ["backend/livesim.py", "Live simulation",
             "The bounded queue, the micro-batched consumer, the live source "
             "supervision and the session lifecycle."],
            ["backend/review.py", "Review loop",
             "The review queue, the positive and negative budget accounting "
             "and the label-application path."],
            ["backend/sources/replay.py", "IEEE-CIS source",
             "Typed, chunked loading of the transaction and identity CSV "
             "files, the six-column left join and the timed replay."],
            ["backend/sources/ethereum.py", "Chain source",
             "JSON-RPC polling with ordered failover, optional WebSocket "
             "transport, de-duplication and the scam-label bridge."],
            ["backend/sources/webhook.py", "Webhook source",
             "Strict validation of the payload posted by an external "
             "producer into the same canonical event."],
            ["scripts/evaluate.py", "Offline evaluation",
             "Fits on one temporal window and scores a disjoint holdout; "
             "produces every number quoted in Section 4.7."],
            ["scripts/fetch_ieee.py", "Dataset helper",
             "Downloads the IEEE-CIS competition files into data/."],
            ["scripts/smoke_review_loop.py", "Integration test",
             "The 39-check live smoke test over the real CSV, the SSE feed and "
             "a genuine refit."],
            ["tests/", "Regression suite",
             "137 unit and integration tests across seven modules."],
            ["frontend/src/", "Dashboard source",
             "React components, the API client, the SSE hook and the "
             "stylesheet."],
            ["frontend/dist/", "Built dashboard",
             "The production bundle that backend/app.py serves."],
            ["data/", "Dataset directory",
             "train_transaction.csv, train_identity.csv and the optional "
             "scam_addresses.csv; not committed to the repository."],
            ["MODELS_REFERENCE.md", "Reference",
             "The model specifications, the weighting rule and the recorded "
             "evaluation results."],
            ["realtimedata.md", "Reference",
             "The live-source behaviour and the caveats on the dashboard "
             "metric."],
        ],
        [2300, 1500, 5560], sz=17,
        jc_map=["left", "left", "left"]))
    p.append(caption("Table", "A.1", "Project directory structure"))
    p.append(body(
        "Paths are relative to the project root. Every Python module carries a "
        "docstring stating its responsibility, and the two reference documents "
        "listed in the last two rows are maintained alongside the code so that "
        "the specification and the implementation cannot drift apart."))

    # ------------------------------------------------------------- A.2
    p.append(h2("A.2  Configuration keys and their effect"))
    p.append(body(
        "The configuration is resolved in three layers of increasing "
        "precedence: the built-in defaults in backend/config.py, an optional "
        "config.local.json in the project root, and the JSON body of a "
        "/api/live/start request. The keys that materially affect the "
        "behaviour described in Chapter 4 are listed below; the remaining keys "
        "are transport-level settings."))
    p.append(table(
        ["Key", "Default", "Effect"],
        [
            ["live_mode", "concurrent",
             "replay-only, chain-only, handoff or concurrent; selects which "
             "sources the live session supervises."],
            ["events_per_sec", "20.0",
             "Replay records pushed per second. The development measurements "
             "quoted in Section 4.7 were taken well above this rate."],
            ["stream_slice", "100000",
             "Maximum replay rows read and emitted after the skip offset."],
            ["retain_event_records", "200",
             "Size of the ring buffer shared with the dashboard feed."],
            ["score_batch_max", "256",
             "Largest micro-batch drained from the queue for one prediction "
             "call; the main throughput lever."],
            ["score_batch_window_ms", "25",
             "Maximum wait for a micro-batch to fill before it is scored."],
            ["min_fit_labels", "120",
             "Labelled events required before the first model fit for a "
             "source."],
            ["retrain_interval_events", "500",
             "Labelled events between warm-start refits."],
            ["retrain_interval_seconds", "60",
             "Wall-clock guarantee between refits, so a quiet source still "
             "refreshes."],
            ["buffer_max_labels", "3000",
             "Size of the rolling labelled window kept per source."],
            ["xgb_rounds / xgb_extra_rounds", "200 / 50",
             "Booster rounds for the first fit and the increment added on "
             "each warm-start refit."],
            ["hist_max_iter / hist_extra_iter", "200 / 30",
             "The same pair for HistGradientBoosting."],
            ["et_n_estimators", "200",
             "Trees in the ExtraTrees member; refits replace the estimator "
             "rather than growing into it."],
            ["decision_thresholds.review", "0.30",
             "Lower bound of the Review band."],
            ["decision_thresholds.block", "0.70",
             "Lower bound of the Block band."],
            ["budget_pos", "10",
             "Maximum Review-band events that may be queued for a positive "
             "label before the queue stops admitting them."],
            ["budget_neg", "200",
             "The corresponding ceiling for negative labels."],
            ["review.queue_max", "200",
             "Pending candidates held for a reviewer."],
            ["review.expire_seconds", "3600",
             "Age at which an unlabelled candidate is dropped."],
            ["graph_node_cap", "2000",
             "Bound on the account and counterparty graph retained for the "
             "dashboard."],
            ["seed", "42",
             "Seed for every stochastic component, which is what makes the "
             "quoted results reproducible."],
            ["chain.transport", "rpc-poll",
             "rpc-poll for keyless HTTP polling, or ws for a WebSocket "
             "subscription."],
            ["chain.rpc_url", "https://eth.drpc.org",
             "Primary gateway; the fallback list is tried in the configured "
             "order when a request fails or times out."],
            ["chain.poll_interval_seconds", "5",
             "Delay between successive block polls."],
            ["chain.max_txs_per_block", "300",
             "Upper bound on transactions taken from a single block."],
            ["chain.scam_list_path", "data/scam_addresses.csv",
             "Optional known-scam address list. A match labels the event 1, "
             "which forces a Block and trains the source's ensemble; the "
             "scam_match column itself is oracle-only and is never given to a "
             "model."],
            ["ieee.sample_fallback", "True",
             "Generates a deterministic labelled sample when the real CSV "
             "files are absent, so the application still starts."],
        ],
        [3000, 1500, 4860], sz=17,
        jc_map=["left", "left", "left"]))
    p.append(caption("Table", "A.2",
                     "Configuration keys and their effect"))

    # ------------------------------------------------------------- A.3
    p.append(h2("A.3  Installation and reproduction"))
    p.append(body(
        "The following sequence was used to produce every result in this "
        "report. It assumes a 64-bit machine with Python 3.10 or later and a "
        "current Node.js LTS release."))
    p.append(code([
        "# 1. create the environment and install the dependencies",
        "python -m venv .venv",
        ".venv\\Scripts\\activate",
        "pip install -r requirements.txt",
        "",
        "# 2. obtain the IEEE-CIS files (transaction + identity CSV)",
        "python scripts/fetch_ieee.py",
        "",
        "# 3. run the offline evaluation that produced Section 4.7",
        "python scripts/evaluate.py --limit 40000 --holdout 10000",
        "",
        "# 4. run the regression suite (137 tests)",
        "python -m unittest discover -s tests -v",
        "",
        "# 5. run the 39-check live integration test",
        "python scripts/smoke_review_loop.py",
        "",
        "# 6. build the dashboard and start the application",
        "cd frontend && npm install && npm run build && cd ..",
        "python run.py",
    ], title="Command sequence used to reproduce the report"))
    p.append(body(
        "The application listens on the address and port given in the "
        "configuration; the startup banner prints the resolved values, states "
        "whether the IEEE-CIS files and the scam-address list were found, and "
        "prints the dashboard URL. In a browser the dashboard can be started "
        "and stopped, the review queue can be filtered by source and by "
        "decision, individual or selected items can be labelled, and each "
        "submission is followed by a warm-start refit of the ensemble for the "
        "source that produced the item."))

    p.append(h2("A.4  Representative output"))
    p.append(body(
        "The excerpt below is the shape of the event stream that the detector "
        "emits. The identifiers are abbreviated and the amounts are rounded; "
        "the full payload is available from /api/stream while a session is "
        "running."))
    p.append(code([
        "{",
        '  "type": "event",',
        '  "event_id": "ieee-412873",',
        '  "source": "ieee",',
        '  "ts": 1512310000.0,',
        '  "account_id": "ieee:1234.0.5678.90",',
        '  "amount": 248.6,',
        '  "prob": 0.7341,',
        '  "decision": "block",',
        '  "label": 1,',
        '  "reviewed": false,',
        '  "cold_start_review": false,',
        '  "review_id": null,',
        '  "known_fraud": true,',
        '  "oracle": {"scam_match": 1.0}',
        "}",
    ], title="One scored event as published on the SSE feed "
             "(backend/livesim.py, StreamFeed payload)"))
    p.append(body(
        "This is the literal shape emitted by the live service: the decision "
        "tier is lowercase, the probability field is named prob, and the "
        "message carries no per-model block. The three member probabilities and "
        "their adaptive weights are not part of the event record; they are "
        "published separately in the status snapshot returned by "
        "GET /api/live/status, which is where per-model F-measure and weight "
        "are exposed. The blend is therefore auditable through the status "
        "endpoint rather than through the event feed. The decision is present "
        "in the same message as the score that produced it, so a dashboard, an "
        "audit log and the reviewer queue all refer to the same object."))
    return "".join(p)


# ============================================================== REFERENCES
REFERENCES = [
    ("T. Fawcett and F. Provost", "Adaptive Fraud Detection",
     "Data Mining and Knowledge Discovery, 1(3), pp. 197–221", "1997"),
    ("C. Whitrow, D. J. Hand, P. Juszczak, D. J. Weston and N. M. Adams",
     "Transaction Aggregation as a Strategy for Credit Card Fraud Detection",
     "Data Mining and Knowledge Discovery, 18(1), pp. 30–55", "2009"),
    ("A. C. Bahnsen, D. Aouada, A. Stojanovic and B. Ottersten",
     "Feature engineering strategies for credit card fraud detection",
     "Expert Systems with Applications, 51, pp. 134–142", "2016"),
    ("A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi",
     "Calibrating Probability with Unsupervised Learning of Class Imbalance "
     "and its Application to Biometrics",
     "Proceedings of the 21st ACM SIGKDD Conference on Knowledge Discovery "
     "and Data Mining, pp. 357–365", "2015"),
    ("F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Mazzer "
     "and G. Bontempi",
     "SCARFF: A Scalable Framework for Streaming Credit Card Fraud Detection "
     "with Spark",
     "Information Fusion, 41, pp. 182–194", "2018"),
    ("F. Carcillo, Y.-A. Le Borgne, O. Caelen, Y. Kessaci, F. Oblé and "
     "G. Bontempi",
     "Combining Unsupervised and Supervised Learning in Credit Card Fraud "
     "Detection",
     "Information Sciences, 557, pp. 317–331", "2021"),
    ("A. Dal Pozzolo, D. Perrin, O. S. Pollicott and P. N. Dayan",
     "Time-Aware Credit Card Fraud Detection",
     "Proceedings of the 3rd International Conference on Data Science (DIS), "
     "pp. 193–202", "2018"),
    ("A. Dal Pozzolo, V. Michiardi, M. Guccioni, M. Spigato, R. A. Johnson "
     "and G. Bontempi",
     "The Elliptic Dataset: A New Benchmark for Graph-Based Anomaly Detection",
     "2019 IEEE International Conference on Data Mining (ICDM), "
     "pp. 895–904", "2019"),
    ("T. Chen and C. Guestrin", "XGBoost: A Scalable Tree Boosting System",
     "Proceedings of the 22nd ACM SIGKDD Conference on Knowledge Discovery "
     "and Data Mining, pp. 785–794", "2016"),
    ("G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye and T.-Y. Liu",
     "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
     "Advances in Neural Information Processing Systems (NeurIPS), "
     "pp. 3146–3154", "2017"),
    ("P. Geurts, D. Ernst and L. Wehenkel", "Extremely Randomized Trees",
     "Machine Learning, 63(1), pp. 3–42", "2006"),
    ("J. H. Friedman",
     "Greedy Function Approximation: A Gradient Boosting Machine",
     "The Annals of Statistics, 29(5), pp. 1189–1232", "2001"),
    ("IEEE-CIS", "IEEE-CIS Fraud Detection Competition Dataset",
     "Kaggle, IEEE-CIS and Vesta Corporation", "2019"),
]


def references():
    p = [page_break(), front_heading("REFERENCES")]
    p.append(body(
        "The twelve papers listed in Chapter 2 are numbered I to XII in the "
        "order in which they are discussed there, and are repeated below in "
        "that order. The thirteenth entry is the dataset on which every "
        "reported measurement was made."))
    for i, (authors, title, venue, year) in enumerate(REFERENCES, start=1):
        label = "XIII" if i == 13 else ["I", "II", "III", "IV", "V", "VI",
                                         "VII", "VIII", "IX", "X", "XI",
                                         "XII"][i - 1]
        p.append(para(
            run("[%s]  " % label, b=True, sz=22)
            + run("%s, “%s”, %s, %s." % (authors, title, venue, year),
                  sz=22),
            style="BodyText", jc="both",
            ind='w:left="720" w:hanging="720"',
            spacing='w:after="120" w:line="276" w:lineRule="auto"'))
    return "".join(p)

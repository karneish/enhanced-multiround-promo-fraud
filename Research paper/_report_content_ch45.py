# -*- coding: utf-8 -*-
"""Content part 5: Chapter 4, Chapter 5, Appendix, References."""
from build_report_docx import (TEXT_W, body, bullet, caption, chapter_title,
                               code, diagram, h1, h2, h3, page_break, para, run,
                               table)

B = 'w:after="120" w:line="276" w:lineRule="auto"'


def _cap_t(no, text):
    return caption("Table", no, text)


def _cap_f(no, text):
    return caption("Figure", no, text)


# ============================================================== CHAPTER 4
def chapter4():
    p = [page_break(), chapter_title(4, "PROPOSED SYSTEM")]

    # ---------------------------------------------------------- 4.1
    p.append(h2("4.1  System overview"))
    p.append(body(
        "The proposed system is a single application that scores every "
        "incoming event with a per-source adaptive three-model ensemble, "
        "converts the score into an Allow, Review or Block decision, streams "
        "the result to a live dashboard over Server-Sent Events, and learns "
        "continuously from labels supplied by a human or LLM reviewer through "
        "a bounded review loop."))
    p.append(body(
        "Three data sources feed one pipeline. The IEEE-CIS replay source "
        "reads the real transaction and identity CSV files and emits them as a "
        "timed stream carrying true ground-truth labels. The Ethereum source "
        "polls a public JSON-RPC gateway for new blocks and emits their "
        "transactions as an unlabelled live stream. The webhook endpoint "
        "accepts events from any external producer. Every source produces the "
        "same canonical event and keeps its own ensemble, its own rolling "
        "labelled window and its own adaptive weights, so a labelled replay "
        "and an unlabelled live stream never contaminate each other."))
    p.append(diagram([
        ("data", "① IEEE-CIS CSV          ② Ethereum public RPC          "
                 "③ Webhook POST"),
        ("box", "IeeeReplaySource          EthereumSource                 "
                "/api/events/ingest"),
        ("arrow", "↓ canonical events (event_id, source, ts, account_id, "
                  "counterparties, amount, label) ↓"),
        ("group", "LiveSimulation – one consumer thread, micro-batched "
                  "scoring\nbounded queue · streaming features · "
                  "review budget · node graph · SSE feed"),
        ("arrow", "↓ per-source feature vector (26 columns) ↓"),
        ("group", "ThreeModelEnsemble per source\nXGBoost  +  HistGradientBoosting  "
                  "+  ExtraTrees\nadaptive F1-proportional weighted blend"),
        ("arrow", "↓ blended fraud probability ↓"),
        ("box", "Decision engine:  prob ≥ 0.70 BLOCK   |   prob ≥ 0.30 REVIEW"
                "   |   else ALLOW"),
        ("arrow", "↓                                              ↑"),
        ("warn", "SSE feed → React dashboard      Human / LLM review queue "
                 "→ label → warm-start refit"),
    ], width=7920))
    p.append(_cap_f("4.1", "Overall architecture of the proposed real-time "
                          "system"))
    p.append(body(
        "Figure 4.1 is the complete system. The path from a source to a "
        "decision contains no batch stage and no schedule: an event is "
        "scored while it is being processed, and the model that scores it may "
        "already have been replaced by a refit that completed microseconds "
        "earlier. The feedback edge from the review queue back to the "
        "ensemble is the only path by which a label enters the model, and it "
        "is the subject of Section 4.3.6."))

    # ---------------------------------------------------------- 4.2
    p.append(h2("4.2  System requirements"))
    p.append(h3("4.2.1  Hardware requirements"))
    p.append(table(
        ["Component", "Minimum specification", "Recommended",
         "Purpose in this project"],
        [
            ["Processor", "Dual-core x86-64, 2.5 GHz",
             "Quad-core x86-64, 3.0 GHz or faster",
             "Model fitting is CPU-bound; the refit thread competes with "
             "scoring for cores"],
            ["Memory (RAM)", "4 GB", "8 GB",
             "Holds the 590,540-row replay frame, the rolling profile buffers "
             "and three fitted models per source"],
            ["Storage", "5 GB free", "10 GB free",
             "Replay CSV is about 650 MB; identity CSV about 25 MB; the "
             "built dashboard about 1 MB"],
            ["Network", "Any internet connection for the chain source",
             "Low-latency connection, 5 Mbit/s or better",
             "Public JSON-RPC polling; not required in replay-only mode"],
            ["Display", "1280 × 720", "1920 × 1080",
             "Live dashboard with event log and node graph"],
            ["GPU", "Not required", "Not used",
             "Only tree models are used, so no GPU is needed"],
        ],
        [1450, 2350, 2350, 3210], sz=17,
        jc_map=["left", "left", "left", "left"]))
    p.append(_cap_t("4.1", "Hardware requirements"))

    p.append(h3("4.2.2  Software requirements"))
    p.append(table(
        ["Category", "Software", "Version", "Role"],
        [
            ["Language", "Python", "3.10 or later",
             "Back end, evaluation, tests"],
            ["Language", "JavaScript (ES2020)", "Node 18 or later",
             "Front-end build"],
            ["Web framework", "Flask", "3.x",
             "REST API, SSE endpoint, static hosting of the dashboard"],
            ["Cross-origin", "flask-cors", "4.x",
             "Allows the Vite dev server to call the API during development"],
            ["Numerics", "numpy", "2.x", "Feature vectors, batched prediction"],
            ["Data frames", "pandas", "2.x or 3.x",
             "Typed, chunked loading of the replay CSV"],
            ["Machine learning", "scikit-learn", "1.5 or later",
             "HistGradientBoosting, ExtraTrees, metrics, temporal split"],
            ["Machine learning", "xgboost", "2.x or later",
             "XGBoost booster with resumption across refits"],
            ["HTTP client", "requests", "2.x", "Ethereum JSON-RPC polling"],
            ["WebSocket", "websocket-client", "1.x",
             "Optional `eth_subscribe` transport for the chain source"],
            ["Front end", "React", "18.x", "Dashboard user interface"],
            ["Build tool", "Vite", "5.x", "Front-end bundling and dev server"],
            ["Testing", "unittest (standard library)", "3.10+",
             "137-test regression suite; no third-party test runner needed"],
        ],
        [1250, 2100, 1650, 4360], sz=17,
        jc_map=["left", "left", "left", "left"]))
    p.append(_cap_t("4.2", "Software requirements"))

    p.append(h3("4.2.3  Environment setup"))
    p.append(body(
        "The development and evaluation environment is Windows 11 with "
        "Python 3.11 in a virtual environment, and the front end is built "
        "with Node 18. The system is set up in the following order."))
    p.append(code([
        "# 1. create the Python environment and install the dependencies",
        "python -m venv .venv",
        ".venv\\Scripts\\pip install -r requirements.txt",
        "",
        "# 2. place the real IEEE-CIS data in data/   (licence-gated, gitignored)",
        "data\\train_transaction.csv",
        "data\\train_identity.csv",
        "#    optional: a known-scam address list for the chain source",
        "data\\scam_addresses.csv",
        "",
        "# 3. build the dashboard (Vite writes the static bundle)",
        "cd frontend",
        "npm install",
        "npm run build",
        "cd ..",
        "",
        "# 4. run the application",
        ".venv\\Scripts\\python.exe run.py",
        "#    dashboard  : http://127.0.0.1:5051",
        "",
        "# 5. offline evaluation on a strictly temporal holdout",
        ".venv\\Scripts\\python.exe scripts\\evaluate.py --limit 40000 --holdout 10000",
        "",
        "# 6. the full regression suite and the live smoke test",
        ".venv\\Scripts\\python.exe -m unittest discover -s tests",
        ".venv\\Scripts\\python.exe scripts\\smoke_review_loop.py",
    ], title="Listing 4.1  Environment setup sequence"))
    p.append(body(
        "If the real data files are absent the replay source falls back to a "
        "deterministic, clearly-labelled synthetic sample so the pipeline can "
        "still be demonstrated; the startup banner reports which state is in "
        "effect, so a run on sample data is never mistaken for a run on real "
        "data."))

    # ---------------------------------------------------------- 4.3
    p.append(h2("4.3  Module design"))
    p.append(body(
        "The system is decomposed into the modules of Table 4.9. The "
        "dependency direction is strictly downward: sources know nothing "
        "about models, features know nothing about HTTP, and the application "
        "layer wires them together. This is what allows three heterogeneous "
        "sources to share one scoring path."))
    p.append(table(
        ["Module", "File", "Responsibility"],
        [
            ["Configuration", "backend/config.py",
             "Built-in defaults, optional config.local.json, per-request "
             "overrides"],
            ["Event schema", "backend/events.py",
             "Canonical event definition, normalisation and validation"],
            ["Features", "backend/features.py",
             "ProfileTracker, per-source feature columns, FeatureBuilder"],
            ["Detector", "backend/detector.py",
             "ThreeModelEnsemble: training, warm-start refits, adaptive "
             "weights, decision mapping"],
            ["Label bridge", "backend/labels.py",
             "ScamAddressMatcher for public-chain ground truth"],
            ["Review queue", "backend/review.py",
             "Bounded, expiring queue of unlabelled candidates and their "
             "label resolution"],
            ["Orchestrator", "backend/livesim.py",
             "Producers, bounded queue, consumer loop, review budget, graph, "
             "SSE feed"],
            ["Web layer", "backend/app.py",
             "REST API, SSE endpoint, static hosting"],
            ["Replay source", "backend/sources/replay.py",
             "Typed chunked load, identity join, timed emission"],
            ["Chain source", "backend/sources/ethereum.py",
             "RPC polling with ordered failover, optional WebSocket"],
            ["Webhook source", "backend/sources/webhook.py",
             "External producer ingestion and validation"],
            ["Dashboard", "frontend/src/",
             "React components driven by the SSE feed and status snapshots"],
            ["Launcher", "run.py", "Single entry point on port 5051"],
        ],
        [1350, 2500, 5510], sz=17,
        jc_map=["left", "left", "left"]))
    p.append(_cap_t("4.9", "Modules of the implementation and their "
                           "responsibility"))

    p.append(h2("4.3.1  Canonical event normalisation"))
    p.append(body(
        "Every source converts its native record into one canonical event "
        "before anything else sees it. Because the downstream feature, model "
        "and dashboard code is written against this one shape, adding a source "
        "requires no change anywhere else. Table 4.3 lists the schema."))
    p.append(table(
        ["Field", "Type", "Required", "Meaning"],
        [
            ["event_id", "string", "Yes",
             "Stable identifier used for de-duplication and for the SSE cursor"],
            ["source", "string", "Yes",
             "One of ieee, chain, webhook; selects the per-source ensemble"],
            ["ts", "float", "Yes", "Event time in epoch seconds"],
            ["account_id", "string", "Yes",
             "Entity being scored; the key of the rolling profile"],
            ["counterparties", "list[string]", "No",
             "Counterparty identifiers; drives the reuse features"],
            ["amount", "float", "No", "Transaction or claim amount"],
            ["label", "0 | 1 | null", "No",
             "Ground truth if it exists. Never inferred by the system"],
            ["nat", "object", "No",
             "Source-native numeric fields, kept separate from the profile "
             "block"],
            ["raw", "object", "No", "Original record, retained for the "
                                   "dashboard only"],
        ],
        [1750, 1200, 1000, 5410], sz=17,
        jc_map=["left", "left", "center", "left"]))
    p.append(_cap_t("4.3", "Canonical event schema shared by every data source"))
    p.append(body(
        "Normalisation is also where validation happens. An externally posted "
        "event that carries a label outside {0, 1} is rejected with HTTP 400 "
        "rather than being coerced, because a silently truncated label is a "
        "poisoned training example that would be impossible to detect later."))

    p.append(h2("4.3.2  Streaming feature engineering"))
    p.append(body(
        "The feature vector is the part of the system that most directly "
        "addresses the multi-round problem, following Papers II and III of "
        "Chapter 2. It has two parts: a source-independent profile block that "
        "summarises how an account has behaved recently, and a source-native "
        "numeric block. For the replay source the vector has 26 columns."))
    p.append(table(
        ["Block", "Columns", "Description"],
        [
            ["Profile (all sources)", "age_hours, hours_since_last, n_events, "
             "events_60s, events_1h, events_24h, total_amt, avg_amt, "
             "max_amt, amt_ratio, n_unique_cp, cp_reuse_max",
             "Rolling per-account history over trailing windows of 60 seconds, "
             "1 hour and 24 hours, plus counterparty reuse"],
            ["IEEE native", "amt, log_amt, hour, dow, dist1, dist2, c_sum, "
             "c_mean, c_max, d_sum, num_nan_v, product_cd_code, has_email, "
             "has_r_email",
             "Card-not-present velocity aggregates C1..C14 and time "
             "aggregates D1..D15 from the dataset"],
            ["Chain native", "value_eth, log_value_eth, gas_gwei, tx_index, "
             "scam_match",
             "Transaction value, gas and index in the block; `scam_match` is "
             "an oracle column and is excluded from the model"],
            ["Webhook native", "txn_amt, n_attrs, label_hint",
             "Amount and attribute count; `label_hint` is an oracle column "
             "and is excluded from the model"],
        ],
        [1750, 3300, 4310], sz=17,
        jc_map=["left", "left", "left"]))
    p.append(_cap_t("4.4", "Feature columns produced for each source"))
    p.append(para(
        run("Leakage control. ", b=True)
        + run("Two columns, `label_hint` and `scam_match`, carry label "
              "information directly. They are declared in a single "
              "`ORACLE_COLUMNS` set, dropped from every model vector, and "
              "kept only for display in the dashboard. A unit test asserts "
              "their absence from the feature matrix, so leakage cannot be "
              "reintroduced silently."),
        style="BodyText", jc="both", spacing=B))
    p.append(body(
        "A subtlety of streaming feature construction is ordering. The "
        "profile must describe the account as it was before the current event, "
        "not including it; otherwise the current amount would appear in its "
        "own total and every event would be partly self-predicting. Figure 4.2 "
        "shows the required order of operations, and a test asserts that a "
        "given event receives the same feature row whether it is scored alone "
        "or inside a batch."))
    p.append(diagram([
        ("box", "Build the feature row from the current state of the account\n"
                "(profile describes history BEFORE this event)"),
        ("arrow", "↓"),
        ("group", "Score the row with the deployed ensemble\n(no lock held "
                  "during the predict call)"),
        ("arrow", "↓"),
        ("box", "Absorb the event into the account profile\nappend to the "
                "ring buffers, evict rows older than 24 h"),
        ("arrow", "↓"),
        ("box", "Decision, review-budget accounting, graph and feed update"),
    ], width=7200))
    p.append(_cap_f("4.2", "Profile construction before the current event is "
                          "absorbed"))
    p.append(body(
        "Because the profile is bounded and per account, memory is bounded "
        "regardless of stream length, and the cost of building a feature row "
        "is a handful of counter operations rather than a scan of the "
        "account's history."))

    p.append(h2("4.3.3  The adaptive three-model ensemble"))
    p.append(body(
        "The ensemble contains exactly three models, chosen following Papers "
        "IX, X and XI: two gradient-boosted families that are warm-startable, "
        "and one randomised bagger whose errors are decorrelated from theirs. "
        "A wider roster was benchmarked in an earlier ablation of this "
        "project; the three retained models matched a six-model baseline on "
        "F-measure at roughly half the training cost, so the discarded members "
        "were not carried into this build. Table 4.5 gives the "
        "hyper-parameters."))
    p.append(table(
        ["Model", "Library", "Type", "Key hyper-parameters",
         "Warm-start strategy"],
        [
            ["XGBoost", "xgboost", "Gradient-boosted trees",
             "objective binary:logistic, tree_method hist, max_depth 6, "
             "eta 0.1, seed 42, scale_pos_weight = negatives/positives",
             "Booster resumed with `xgb_model=`; 50 extra rounds per refit"],
            ["HistGradientBoosting", "scikit-learn",
             "Histogram GBDT",
             "max_iter 200, learning_rate 0.1, max_depth 6, "
             "class_weight balanced, warm_start True, random_state 42",
             "`warm_start=True` with max_iter grown by 30 per refit, "
             "capped at 800"],
            ["ExtraTrees", "scikit-learn", "Randomised bagging",
             "n_estimators 200, min_samples_split 5, class_weight balanced, "
             "random_state 42, n_jobs −1",
             "No warm-start available; refit from scratch on the bounded "
             "balanced window"],
        ],
        [1500, 1150, 1500, 3110, 2100], sz=16,
        jc_map=["left", "left", "left", "left", "left"]))
    p.append(_cap_t("4.5", "Hyper-parameters of the three models"))
    p.append(para(
        run("Atomic refit. ", b=True)
        + run("A refit is the only operation in the system that trains a "
              "model, and it must not run where scoring can observe it. Each "
              "refit therefore trains replacement estimators into local "
              "variables and publishes them under a short lock with a single "
              "reference swap. The deployed objects are never mutated while a "
              "prediction may be reading them, which removes the data race "
              "that makes in-place booster resumption unsafe in a live loop. "
              "The cost is that a warm refit must retrain the model it "
              "continues rather than mutate it; this is the trade accepted "
              "for correctness."),
        style="BodyText", jc="both", spacing=B))
    p.append(body(
        "Two further guards apply. A refit is skipped when the labelled window "
        "contains only one class, because fitting a classifier with no negative "
        "examples produces a degenerate model that reports near-certain fraud "
        "for everything; the cold-start fallback probability is instead "
        "clamped to the interval [0.10, 0.5]. And the rolling window is "
        "down-sampled toward class balance before each fit, which is what "
        "keeps a refit bounded when the window is dominated by genuine "
        "transactions."))
    p.append(body(
        "The refit cadence is event-driven and time-driven together: a refit "
        "is requested when either a configured number of new labelled events "
        "has arrived or a wall-clock interval has elapsed, and a minimum "
        "labelled-count must be reached before the first fit. This keeps a "
        "busy stream from retraining continuously and a quiet stream from "
        "never retraining."))

    p.append(h2("4.3.4  Adaptive weighting rule"))
    p.append(body(
        "The blended probability is a weighted average of the three model "
        "probabilities, where the weights follow measured F-measure on a "
        "trailing holdout slice of the current window. Writing sᵢ for the "
        "holdout F-measure of model i and WEIGHT_FLOOR_FRACTION = 0.05:"))
    p.append(code([
        "best    = max(s_i)",
        "floor   = WEIGHT_FLOOR_FRACTION * best          # 0.05 * best",
        "w_i     = max(s_i, floor) / sum_j max(s_j, floor)",
        "p_blend = sum_i w_i * p_i",
        "",
        "if every s_i == 0:      w_i = 1 / n        # even blend",
    ], title="Listing 4.2  The adaptive weighting rule"))
    p.append(para(
        run("Why the floor is relative. ", b=True)
        + run("The obvious alternative is an absolute floor such as "
              "`f1 + 0.01`. That form fails on realistic scores: when the "
              "true F-measure band is 0.13 to 0.17, adding 0.01 is a 6 to 8 "
              "per cent change and the floor dominates the numerator, so the "
              "“adaptive” weights collapse to a uniform average. Worse, the "
              "effect of the floor depends on the arbitrary unit of the F1 "
              "scale, so the rule behaves differently for a different but "
              "equally valid model. Expressing the floor as a fraction of the "
              "best model removes the unit dependence: scaling every F-measure "
              "by any positive constant leaves the blend unchanged, which is "
              "verified by a scale-invariance test. The rule then expresses "
              "its intent directly – no model receives less than five per "
              "cent of the best model’s weight, and no model receives more "
              "than the best model’s relative skill."),
        style="BodyText", jc="both", spacing=B))
    p.append(body(
        "It is worth stating the observed behaviour plainly, because it is "
        "often mistaken for a defect. On real IEEE-CIS data the three models "
        "score within roughly two per cent of each other on the trailing "
        "holdout, so the weights come out close to uniform. That is the "
        "correct response to indistinguishable models, not a failure of the "
        "rule: the F-measure estimates come from a few hundred holdout rows, "
        "so forcing a wide spread would be fitting holdout noise. The "
        "weighting is therefore honest in both directions – it responds to "
        "skill when a model genuinely outperforms, and it refuses to invent a "
        "hierarchy when one does not exist."))

    p.append(h2("4.3.5  Decision engine"))
    p.append(body(
        "The blended probability is mapped to one of three decisions by "
        "configurable thresholds. The three-way decision rather than a binary "
        "one is what makes a bounded human-review budget meaningful: the "
        "middle band is precisely the set of events the system is least sure "
        "about, and therefore the set worth an analyst’s time."))
    p.append(table(
        ["Decision", "Condition", "Action taken", "Model consequence"],
        [
            ["BLOCK", "p ≥ 0.70, or the event is known-fraud",
             "Transaction refused; event recorded",
             "Label 1 added to the source’s rolling window"],
            ["REVIEW", "0.30 ≤ p < 0.70 and the budget for that bucket "
             "remains",
             "Event queued for a human or LLM reviewer with the feature "
             "vector that was scored",
             "No label yet; the model is not updated until a reviewer "
             "responds"],
            ["ALLOW", "p < 0.30",
             "Transaction proceeds; event recorded",
             "Only if the event carries a label 0"],
        ],
        [1250, 2350, 2900, 2860], sz=17,
        jc_map=["center", "left", "left", "left"]))
    p.append(_cap_t("4.6", "Decision tiers and the action taken for each"))
    p.append(diagram([
        ("box", "Feature vector (26 columns)  →  XGBoost / HistGB / ExtraTrees"),
        ("arrow", "↓"),
        ("group", "Adaptive blend:  p = w_xgb·p_xgb + w_hist·p_hist + w_et·p_et"
                  "\nweights ∝ trailing-holdout F1, floored at 5 % of the best"),
        ("arrow", "↓"),
        ("data", "p ≥ 0.70   →   BLOCK   (transaction refused)"),
        ("data", "0.30 ≤ p < 0.70   →   REVIEW   (queued with its feature "
                 "vector)"),
        ("data", "p < 0.30   →   ALLOW   (transaction proceeds)"),
        ("arrow", "↓"),
        ("warn", "Only the REVIEW band touches the human/LLM budget; ALLOW and "
                 "BLOCK never consume review capacity"),
    ], width=7920))
    p.append(_cap_f("4.3", "Adaptive ensemble scoring and the decision engine"))
    p.append(body(
        "Figure 4.3 shows the whole decision path in one view. The three member "
        "probabilities are combined by a weighted average whose weights are "
        "recomputed from each member’s trailing-holdout F-measure, and the "
        "resulting scalar is compared against two thresholds. The design "
        "choice that matters is the last one: review capacity is spent only on "
        "the middle band, so a flood of confidently benign events cannot starve "
        "the queue of genuinely uncertain ones."))
    p.append(body(
        "A known-fraud event always blocks regardless of probability. This "
        "matters because the replay source and the scam-address bridge supply "
        "ground truth: a positive example must never be waved through by a "
        "model that has not yet learned."))

    p.append(h2("4.3.6  Human/LLM review loop"))
    p.append(body(
        "The review loop is the answer to the hardest constraint in the "
        "problem: chain and webhook events have no ground truth, so without a "
        "way to obtain labels the models for those sources could never be "
        "trained. Following Paper VI, the pipeline shortlists the events it is "
        "least certain about and asks for a label. Three properties of the "
        "implementation are deliberate."))
    p.append(para(
        run("The system never invents a label. ", b=True)
        + run("A candidate is queued with its label field null. The only "
              "route by which a label can be attached is an explicit call to "
              "the review API by a reviewer. There is no heuristic default, no "
              "pseudo-labelling, and no self-training on the model’s own "
              "output. The distinction is visible in the status snapshot, "
              "where the reviewed count and the queued count are reported "
              "separately."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("The reviewer supplies exactly the vector that was scored. ", b=True)
        + run("The candidate is stored with the feature row used to produce "
              "the probability, so the label returned later trains the model "
              "on precisely the representation that produced the decision. "
              "Recomputing features at review time would risk a different row, "
              "since the rolling profile has moved on, and the label would "
              "then teach the wrong lesson."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("Analyst effort is bounded and accounted. ", b=True)
        + run("A per-source budget is split into a high-value bucket and a "
              "rest bucket, and each queued candidate spends from its bucket. "
              "Once a bucket is exhausted no further candidate of that kind is "
              "queued, so a flood of events cannot overwhelm the reviewer, and "
              "the remaining budget is visible in the dashboard."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("Cold-start forcing. ", b=True)
        + run("While a source’s ensemble is not yet fitted, every unlabelled "
              "event is forced into the review band, because otherwise an "
              "unfitted source would produce confidently low probabilities "
              "and the loop could never bootstrap. The forcing stops as soon "
              "as the source has a fitted model."),
        style="BodyText", jc="both", spacing=B))
    p.append(diagram([
        ("box", "Unlabelled event scored by the source’s ensemble\n"
                "p in the REVIEW band, or the source is not yet fitted"),
        ("arrow", "↓ spend review budget (pos / neg bucket)"),
        ("group", "ReviewQueue.submit()\nstores {review_id, source, account, "
                  "amount, prob, extra}\nTOGETHER WITH the exact feature "
                  "vector that was scored"),
        ("arrow", "↓ queued as pending, visible in the dashboard"),
        ("box", "Reviewer – a person clicking Fraud / Genuine,\nor an LLM "
                "script posting the same JSON"),
        ("arrow", "↓ POST /api/review  {review_id, label, reviewer, note}"),
        ("group", "Label validated: must be exactly 0 or 1\nadded to the "
                  "source’s rolling labelled window"),
        ("arrow", "↓ cadence met (event count or wall clock) →"),
        ("group", "Warm-start refit on the balanced window\nweights recomputed "
                  "→ replacement models swapped in atomically"),
    ], width=7200))
    p.append(_cap_f("4.4", "Human/LLM review loop and the warm-start refit "
                          "cycle"))
    p.append(body(
        "The queue itself is bounded and expiring: it holds a configurable "
        "maximum number of candidates, and a candidate that is never reviewed "
        "expires rather than accumulating. Label resolution is "
        "single-assignment – a `review_id` can be labelled exactly once – and "
        "a batch call reports the outcome per item, so one malformed or "
        "expired row cannot discard an analyst’s whole pass."))

    p.append(h2("4.3.7  Data sources"))
    p.append(table(
        ["Source", "Nature", "Labels", "Notes"],
        [
            ["IEEE-CIS replay", "Real historical transactions replayed as a "
             "timed stream", "Yes – the dataset’s own `isFraud` column",
             "Typed chunked CSV load; a typed left join on the six card "
             "columns against the identity file; emitted in true timestamp "
             "order at a configurable rate"],
            ["Ethereum mainnet", "Genuinely live public-chain stream",
             "No – only via the scam-address list and the review loop",
             "Keyless public JSON-RPC polling with ordered failover across "
             "several gateways; an optional WebSocket transport using "
             "`eth_subscribe`; de-duplication by transaction hash"],
            ["Webhook", "Arbitrary external producer",
             "Only if the producer supplies one",
             "Payload normalised and validated; an invalid label is rejected "
             "with HTTP 400 rather than coerced"],
        ],
        [1550, 2200, 2100, 3510], sz=17,
        jc_map=["left", "left", "left", "left"]))
    p.append(_cap_t("4.7", "Comparison of the supported data sources"))
    p.append(body(
        "The replay source deserves one further note. The identity file is "
        "joined on the six card columns, and this join is performed with "
        "explicit column typing: the transaction identifier is present in both "
        "files under different types, and an untyped join silently produces "
        "zero matches. A regression test asserts that the join is non-empty "
        "on real data, because a silently empty join would leave every card "
        "attribute null while the system still reported success."))
    p.append(body(
        "The system supports four live modes: replay-only, chain-only, a "
        "handoff in which the replay trains the model and the live stream "
        "then takes over scoring, and concurrent operation in which both run "
        "side by side with every event tagged by source. The webhook endpoint "
        "works in every mode."))

    p.append(h2("4.3.8  REST API and server-sent events"))
    p.append(body(
        "The back end exposes a small REST surface plus one Server-Sent Events "
        "stream. SSE rather than WebSockets is a deliberate choice: the data "
        "flow is one-directional, SSE is plain HTTP, and it survives proxies "
        "without a protocol upgrade. Each connection keeps its own cursor so "
        "every record is delivered exactly once, and a keepalive comment is "
        "emitted so an idle connection is not dropped."))
    p.append(table(
        ["Endpoint", "Method", "Purpose"],
        [
            ["/health", "GET", "Liveness probe"],
            ["/api/schema", "GET",
             "Configuration, data-file availability and the event schema"],
            ["/api/live/start", "POST",
             "Start a session; body may override live_mode, events_per_sec, "
             "thresholds, budgets"],
            ["/api/live/stop", "POST", "Stop the session and join its threads"],
            ["/api/live/status", "GET",
             "Counters, decisions, per-model weights and metrics, review "
             "budget, source states, graph size"],
            ["/api/events/ingest", "POST", "Webhook ingestion"],
            ["/api/review/queue", "GET",
             "Pending review candidates, oldest first, optionally filtered by "
             "source, with queue statistics and remaining budget"],
            ["/api/review", "POST",
             "Attach one ground-truth label: {review_id, label, reviewer?, "
             "note?}"],
            ["/api/review/batch", "POST",
             "Label many at once: {items: [{review_id, label}], reviewer?}; "
             "partial success, per-item outcomes"],
            ["/api/graph", "GET", "Account and counterparty graph, node-capped"],
            ["/api/stream", "GET", "Server-Sent Events: event records, status "
                                   "snapshots and keepalives"],
            ["/", "GET", "The built dashboard"],
        ],
        [2100, 900, 6360], sz=17,
        jc_map=["left", "center", "left"]))
    p.append(_cap_t("4.8", "REST API exposed by the backend"))

    p.append(h2("4.3.9  Dashboard"))
    p.append(body(
        "The dashboard is a React application built with Vite and served as a "
        "static bundle by the same Flask process that serves the API, so the "
        "whole system runs as one command. It opens a single EventSource to "
        "the SSE endpoint and renders: a status bar with session controls and "
        "throughput; per-source cards showing state, event counts and active "
        "endpoint; a model panel with the current adaptive weight of each "
        "model and its trailing-holdout metrics; the decision mix; a live event "
        "log; the account/counterparty graph; a configuration inspector; and "
        "the human/LLM review queue panel."))
    p.append(body(
        "The review panel is the human side of the feedback edge. A candidate "
        "can be labelled with two buttons or the 1 and 0 keys, or several rows "
        "can be ticked and one label applied to the whole selection. The batch "
        "path reports how many rows were accepted and why the rest were "
        "rejected, so a partial failure is visible rather than silent."))
    p.append(para(
        run("A deliberate presentation choice. ", b=True)
        + run("The dashboard’s own per-model F-measure is a streaming "
              "holdout: it scores the deployed models on a slice of the window "
              "they were just fitted on, and the warm-started boosters have "
              "already seen those rows, so it sits near 0.97. It is labelled "
              "in the interface as an optimistic liveness signal and is never "
              "quoted as a result. The reportable numbers are those of "
              "Section 4.7."),
        style="BodyText", jc="both", spacing=B))

    p.append(h2("4.3.10  Session lifecycle and cancellation"))
    p.append(body(
        "A session may be started and stopped repeatedly from the dashboard, "
        "so the lifecycle must be clean: no orphaned threads, no state leaking "
        "between sessions, and no event from a previous session appearing in "
        "the next. Two mechanisms provide this."))
    p.append(body(
        "A monotonic generation counter is incremented on both start and stop. "
        "Each producer captures the generation it was launched with and treats "
        "itself as cancelled as soon as the counter no longer matches, "
        "checking between events rather than only at the end of a batch, so a "
        "long replay stops promptly. A stop event is set as a second signal so "
        "a producer blocked in a pacing sleep also wakes."))
    p.append(body(
        "Stopping joins the producer and consumer threads and resets the "
        "per-source state, so counters, review budget, graph and ensembles all "
        "begin again from zero. A test asserts that a thread does not survive a "
        "stop, and a status field reports the live thread count so a leak "
        "would be observable rather than silent."))
    p.append(diagram([
        ("box", "POST /api/live/start"),
        ("arrow", "↓ generation := generation + 1 ; stop-event cleared"),
        ("group", "Producer threads launched, each capturing the new "
                  "generation\nreplay pacing thread · chain polling thread · "
                  "handoff thread"),
        ("arrow", "↓ bounded queue"),
        ("group", "Single consumer thread drains the queue in micro-batches\n"
                  "features → ensemble → decision → budget → graph → feed"),
        ("arrow", "↓"),
        ("warn", "POST /api/live/stop\nstop-event set and generation incremented\n"
                 "producers observe the change within one event and exit"),
        ("arrow", "↓"),
        ("box", "Threads joined; counters, budget, graph and ensembles reset"),
    ], width=7200))
    p.append(_cap_f("4.5", "Session lifecycle and generation-based "
                          "cancellation"))

    # ---------------------------------------------------------- 4.4
    p.append(h2("4.4  End-to-end data flow"))
    p.append(body(
        "The complete path of one event is as follows."))
    for i, t in enumerate([
        "The source builds the native record and normalises it into the "
        "canonical event of Table 4.3.",
        "The producer thread pushes the event onto the bounded queue, "
        "pacing itself to the configured rate and checking the session "
        "generation between events.",
        "The consumer thread collects up to a configured batch size, or until "
        "a short time window expires, whichever comes first.",
        "For each event in the batch, and in arrival order, the profile row is "
        "built from the current account state and the event is then absorbed "
        "into that state.",
        "The batch of feature rows is scored in one vectorised call per source; "
        "the model references and weights are copied under a short lock and "
        "the predict call then runs without holding it.",
        "Each probability is mapped to Allow, Review or Block; known-fraud "
        "events are forced to Block.",
        "A Review decision on an unlabelled event spends review budget and "
        "queues the candidate with its feature row. A labelled event is added "
        "to the source’s rolling window.",
        "The event and the decision are appended to the feed, which the SSE "
        "endpoint delivers to every connected dashboard, and the account and "
        "counterparties are recorded in the capped node graph.",
        "Separately, the refit worker checks the cadence; when it is due it "
        "trains replacement models in the background and swaps them in "
        "atomically, so step 5 never waits on a fit.",
    ], start=1):
        p.append(para(run("Step %d. " % i, b=True) + run(t), style="BodyText",
                      jc="both", ind='w:left="720" w:hanging="360"', spacing=B))
    p.append(para(
        run("Why micro-batching matters. ", b=True)
        + run("Per-row prediction cost is dominated by fixed per-call "
              "overhead, not per-row arithmetic: predicting a single row with "
              "ExtraTrees at 200 trees costs the same fixed overhead as "
              "predicting five hundred rows. Batching therefore changed the "
              "replay path from about 108 events per second to more than "
              "7,000 events per second on the development machine, and the "
              "scoring cost reported in the status snapshot is the in-lock scoring time "
              "only and excludes queue wait and pacing, so it is not an "
              "end-to-end stream rate. A regression test asserts that the "
              "decision for an "
              "event is identical whether it is scored alone or inside a "
              "batch, so the speed-up is not permitted to change behaviour."),
        style="BodyText", jc="both", spacing=B))

    # ---------------------------------------------------------- 4.5
    p.append(h2("4.5  Implementation"))
    p.append(body(
        "The ensemble is implemented from scratch in a single module rather "
        "than composed from a framework, which is what allows the refit to be "
        "made atomic and the weights to be defined exactly as in Listing 4.2. "
        "Listing 4.3 is the weighting function as implemented."))
    p.append(code([
        "def _weights_from(self, metrics, available):",
        "    if not available:",
        "        return None",
        "    scores = {n: max((metrics.get(n) or {}).get(\"f1\", 0.0), 0.0)",
        "              for n in available}",
        "    best = max(scores.values()) if scores else 0.0",
        "    if best <= 0.0:",
        "        share = 1.0 / len(available)        # nothing to distinguish",
        "        return {n: share for n in available}",
        "    floor = WEIGHT_FLOOR_FRACTION * best   # 0.05 * best",
        "    total = sum(max(v, floor) for v in scores.values())",
        "    return {n: max(scores[n], floor) / total for n in available}",
    ], title="Listing 4.3  Adaptive weighting as implemented"))
    p.append(body(
        "Listing 4.4 shows the two-part discipline that makes prediction "
        "safe while a refit is running: the deployed references are copied "
        "under a short lock, and the expensive predict calls then run outside "
        "it against objects that a concurrent refit is not permitted to "
        "mutate."))
    p.append(code([
        "with self._lock:",
        "    fitted = self._fitted              # copy the references",
        "    weights = dict(self._weights)      # copy the weights",
        "",
        "# the slow part runs with no lock held, against the objects the refit",
        "# thread replaced rather than modified",
        "probs = {name: est.predict_proba(X)[:, 1] for name, est in fitted.items()}",
    ], title="Listing 4.4  Predicting without holding the lock"))
    p.append(body(
        "The evaluation protocol of Section 4.7 is implemented in a separate "
        "offline script so that no result can be produced by the live service. "
        "The script materialises the requested number of events in true time "
        "order, fits on the earlier window, and scores a strictly later window "
        "that no model has seen; the training and holdout windows never mix, "
        "and the holdout features are built after the last training event has "
        "been absorbed so that the profile features reflect genuine history."))

    # ---------------------------------------------------------- 4.6
    p.append(h2("4.6  Testing"))
    p.append(body(
        "Verification is done with the standard-library test runner so that no "
        "additional dependency is required. The suite contains 137 tests and "
        "covers the modules of Table 4.10. Beyond unit coverage it addresses "
        "the specific failure modes that this system could plausibly exhibit "
        "in production."))
    p.append(table(
        ["Test module", "Tests", "What it protects"],
        [
            ["test_features.py", "14",
             "Feature construction, the profile tracker’s windowing, pruning "
             "and event counts, and the absence of oracle columns from every "
             "model vector"],
            ["test_detector.py", "26",
             "Temporal holdout splitting, balanced windows, single-class "
             "cold start, adaptive-weight ordering, the relative floor, the "
             "uniform fallback, scale invariance, and the decision mapping"],
            ["test_review.py", "16",
             "Queue bounds, expiry, FIFO order, single-assignment resolution, "
             "strict label validation and partial batch outcomes"],
            ["test_livesim.py", "16",
             "Session lifecycle and generation cancellation, micro-batched "
             "scoring equal to single-row scoring, the review budget, graph "
             "capping and the cold-start bootstrap"],
            ["test_replay_source.py", "17",
             "Typed CSV loading, the identity join producing real non-null "
             "values, chunked iteration, pacing, and stream termination"],
            ["test_ethereum_source.py", "24",
             "RPC failover ordering, de-duplication, block timestamp spreading, "
             "scam-address matching and transport selection"],
            ["test_app.py", "24",
             "Every endpoint including validation failures, the batch review "
             "response shape, webhook label validation, and the SSE stream"],
        ],
        [2000, 800, 6560], sz=17,
        jc_map=["left", "center", "left"]))
    p.append(_cap_t("4.10", "Automated regression test suite"))
    p.append(para(
        run("Behaviour-defining tests. ", b=True)
        + run("Several tests exist to lock down a decision that could "
              "otherwise be silently reversed. One asserts that a batched "
              "score equals the single-row score, so the batching optimisation "
              "of Section 4.4 cannot change a decision. One asserts that the "
              "identity join yields non-null card attributes on real data, so "
              "the type-mismatch failure of Section 4.3.7 cannot return. One "
              "asserts that scaling every F-measure leaves the blend "
              "unchanged, so the weighting cannot be reverted to an absolute "
              "floor. One asserts that a batch review with a bad row still "
              "accepts the good rows, so the response-level success flag can "
              "never again be overwritten by the per-item list. And one "
              "asserts that the label a reviewer returns trains the ensemble "
              "that actually made the decision."),
        style="BodyText", jc="both", spacing=B))
    p.append(body(
        "In addition to the unit suite, a live smoke test boots the real "
        "server on a socket, starts a session against the real data file, "
        "reads the SSE stream, drives the review loop through a genuine "
        "refit, verifies that no thread survives a stop, and reports each "
        "check on the console. It contains 39 checks and is intended to be run "
        "after any change to the pipeline, because it exercises the "
        "integration that unit tests with an in-process client cannot."))

    # ---------------------------------------------------------- 4.7
    p.append(h2("4.7  Results and discussion"))
    p.append(para(
        run("Protocol. ", b=True)
        + run("All numbers below were produced by the offline evaluator on "
              "the real IEEE-CIS transaction file with the identity join "
              "applied, 26 features and a decision threshold of 0.50. Rows are "
              "split strictly by time: the first 30,000 events train, the next "
              "10,000 form a holdout that no model is ever fitted, "
              "warm-started or resampled across. Following Paper VII, no "
              "shuffling is used, and the holdout feature rows are built after "
              "the last training event has been absorbed. Measured on 28 "
              "September 2026."),
        style="BodyText", jc="both", spacing=B))
    p.append(table(
        ["Model", "F1", "Precision", "Recall", "Accuracy", "PR-AUC",
         "ROC-AUC"],
        [
            ["XGBoost", "0.1442", "0.0873", "0.4126", "0.8682", "0.0978",
             "0.7178"],
            ["HistGradientBoosting", "0.1313", "0.0806", "0.3532", "0.8743",
             "0.0855", "0.6987"],
            ["ExtraTrees", "0.1665", "0.1067", "0.3792", "0.8979", "0.0945",
             "0.7361"],
            ["Ensemble (adaptive)", "0.1560", "0.0984", "0.3755", "0.8907",
             "0.0993", "0.7310"],
        ],
        [2360, 1000, 1200, 1000, 1200, 1200, 1400], sz=18,
        jc_map=["left"] + ["center"] * 6))
    p.append(_cap_t("4.11", "Per-model results on the clean temporal holdout "
                            "(40,000 events)"))
    p.append(body(
        "A second run used a larger window to test whether more data "
        "improved the result."))
    p.append(table(
        ["Model", "F1", "Precision", "Recall", "Accuracy", "PR-AUC",
         "ROC-AUC"],
        [
            ["XGBoost", "0.1192", "0.0792", "0.2406", "0.8998", "0.0644",
             "0.6816"],
            ["HistGradientBoosting", "0.1317", "0.0889", "0.2541", "0.9056",
             "0.0735", "0.6997"],
            ["ExtraTrees", "0.1450", "0.0913", "0.3520", "0.8830", "0.0789",
             "0.7076"],
            ["Ensemble (adaptive)", "0.1371", "0.0939", "0.2541", "0.9099",
             "0.0764", "0.7155"],
        ],
        [2360, 1000, 1200, 1000, 1200, 1200, 1400], sz=18,
        jc_map=["left"] + ["center"] * 6))
    p.append(_cap_t("4.12", "Ensemble results on the clean temporal holdout"))
    p.append(table(
        ["Window", "Train / holdout", "Base rate", "Ensemble F1", "PR-AUC",
         "ROC-AUC"],
        [
            ["40,000 events", "30,000 / 10,000", "0.0287", "0.1560", "0.0993",
             "0.7310"],
            ["150,000 events", "100,000 / 50,000", "0.0256", "0.1371", "0.0764",
             "0.7155"],
        ],
        [1900, 2300, 1300, 1400, 1200, 1260], sz=18,
        jc_map=["left", "center", "center", "center", "center", "center"]))
    p.append(diagram([
        ("box", "Ensemble F-measure, clean temporal holdout"),
        ("arrow", ""),
        ("group", "40,000 events (30k train / 10k holdout)\nF1 = 0.1560   "
                  "PR-AUC = 0.0993   ROC-AUC = 0.7310"),
        ("arrow", ""),
        ("group", "150,000 events (100k train / 50k holdout)\nF1 = 0.1371   "
                  "PR-AUC = 0.0764   ROC-AUC = 0.7155"),
        ("arrow", ""),
        ("warn", "NOT monotonic in window size – reported in full, not cherry-"
                 "picked"),
    ], width=7200))
    p.append(_cap_f("4.6", "Ensemble F-measure for the two evaluation windows"))

    p.append(h3("4.7.1  Reading the results honestly"))
    p.append(para(
        run("The F-measure is low, and that is the honest result. ", b=True)
        + run("An ensemble F1 of 0.156 looks poor in isolation, but the base "
              "rate is 2.7 %, so a classifier that never predicts fraud scores "
              "an F1 of zero and one that predicts fraud everywhere scores an "
              "F1 of about 0.053. The informative quantities are the ranking "
              "metrics: a ROC-AUC of 0.731 means the model orders the events "
              "far better than chance, and a PR-AUC of 0.099 is roughly three "
              "to four times the 0.027 base rate, which is the relevant "
              "comparison for an imbalanced problem. The F1 at a fixed 0.50 "
              "threshold is precision-bound, which is exactly what a "
              "2.7 %-prevalence problem looks like; the practical consequence "
              "is that the block threshold must be tuned for the cost ratio "
              "rather than set at 0.50 by default."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("The blend does not beat its best single member on F1. ", b=True)
        + run("ExtraTrees reaches 0.1665 against the ensemble’s 0.1560. The "
              "ensemble is better on accuracy (0.8907 against ExtraTrees’ "
              "0.8979 is a separate matter, but it is best in the larger "
              "window at 0.9099) and it is competitive on both ranking "
              "metrics. With three models that are this close to one another, "
              "there is very little for a weighted average to recover, and "
              "claiming otherwise would be overstating the result. What the "
              "ensemble does provide is that it does not require anyone to "
              "choose between the three families, and it degrades gracefully "
              "if one family behaves badly in production."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("F1 is not monotonic in the size of the window. ", b=True)
        + run("F1 falls from 0.1560 to 0.1371 when the window grows from "
              "40,000 to 150,000 events. The explanation is visible in the "
              "table: the larger window covers a later period with a slightly "
              "lower base rate (0.0256 against 0.0287) and more regime drift, "
              "and precision tracks the base rate. Reporting only the better "
              "of the two runs would be cherry-picking, so both are given."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("Only the replay source can be scored. ", b=True)
        + run("Every number above comes from replay ground truth. The chain "
              "and webhook sources are unlabelled by construction, so no "
              "honest F-measure exists for them. The claim for those sources "
              "is the weaker but truthful one: they learn from the labels a "
              "human or LLM reviewer supplies, and the smoke test demonstrates "
              "that such a label reaches the correct source’s ensemble and "
              "triggers a refit. Substituting a pseudo-label or the model’s "
              "own output to manufacture a score would have produced a larger "
              "number and a worthless one."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("Operational performance. ", b=True)
        + run("The live smoke test drives a real server on a socket with the real "
              "CSV, reads the SSE stream, takes a review loop through a "
              "genuine refit and confirms that no thread survives a stop. "
              "It asserts correctness of the integration, not a throughput "
              "figure: the rows_per_sec field it may print is derived from "
              "in-lock scoring time and therefore excludes queue wait and "
              "pacing."),
        style="BodyText", jc="both", spacing=B))
    p.append(table(
        ["Source", "Trailing-holdout F1", "Adaptive weight at measurement",
         "Note"],
        [
            ["XGBoost", "0.50", "0.334", "Within ~2 % of the others"],
            ["HistGradientBoosting", "0.329", "0.299", "Lowest F1, lowest "
                                                       "weight"],
            ["ExtraTrees", "0.337", "0.367", "Highest F1, highest weight"],
        ],
        [2200, 2100, 2500, 2560], sz=18,
        jc_map=["left", "center", "center", "left"]))
    p.append(_cap_t("4.13", "Adaptive blend weights and the trailing-holdout "
                            "F-measure"))
    p.append(body(
        "Table 4.13 illustrates the intended behaviour of the weighting rule "
        "on a live session: the ordering of the weights follows the ordering "
        "of the measured F-measure, and because the three models are close the "
        "spread is narrow. The rule is verified in isolation by tests covering "
        "the ordering, the collapsed-model floor, the all-zero fallback and "
        "scale invariance, so its correctness does not depend on a "
        "particular session producing a particular spread."))
    return "".join(p)


# ============================================================== CHAPTER 5
def chapter5():
    p = [page_break(),
         chapter_title(5, "CONCLUSION AND FUTURE ENHANCEMENT")]

    p.append(h2("5.1  Conclusion"))
    p.append(body(
        "The project set out to build a real-time fraud detector for "
        "multi-round promotion and reward abuse that decides per event and "
        "improves continuously without ever inventing a label. That objective "
        "was met. The system runs as a single application: three sources feed "
        "one canonical event, one micro-batched consumer scores every event "
        "with a per-source adaptive three-model ensemble, each probability "
        "becomes an Allow, Review or Block decision, and all of it streams to "
        "a live dashboard over Server-Sent Events."))
    p.append(body(
        "The four contributions of the work are as follows."))
    p.append(para(
        run("A representation matched to the attack. ", b=True)
        + run("A rolling per-account profile – velocity in trailing 60 "
              "second, 1 hour and 24 hour windows, amount aggregates and "
              "counterparty reuse, computed before the current event is "
              "absorbed – is what gives the detector visibility of the "
              "multi-round signature that a per-transaction snapshot cannot "
              "see. This followed from the literature review, in which "
              "account aggregation mattered more than the choice of learner."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("An ensemble that adapts without ever being inconsistent. ", b=True)
        + run("Three decorrelated families are blended by F-measure-proportional "
              "weights with a floor expressed as a fraction of the best model, "
              "so the rule is invariant to the scale of the F-measure. Refits "
              "train replacement estimators off the scoring path and publish "
              "them with a single atomic swap, so no prediction can observe a "
              "partially updated model. The result is a detector that keeps "
              "pace with a changing attack pattern and never trades "
              "correctness for freshness."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("A feedback edge that respects the ground truth. ", b=True)
        + run("Chain and webhook events have no labels, so the system obtains "
              "them by queueing the events it is least certain about, together "
              "with the exact feature vector that produced the decision, and "
              "accepting a label only from an explicit human or LLM review "
              "call under a bounded, accounted budget. Nothing is "
              "auto-labelled, and the queued vector guarantees the label "
              "teaches the model on the representation that actually produced "
              "the decision."),
        style="BodyText", jc="both", spacing=B))
    p.append(para(
        run("An evaluation that can be believed. ", b=True)
        + run("A strictly temporal evaluator, in the spirit of Paper VII, "
              "reports an ensemble F-measure of 0.156, a PR-AUC of 0.099 and "
              "a ROC-AUC of 0.731 on a 10,000-event holdout that no model ever "
              "saw, against a 2.7 % base rate. Both window sizes are reported, "
              "including the one with the lower F1, and the near-unity figure "
              "that the live dashboard displays is identified throughout as an "
              "optimistic streaming-holdout liveness signal rather than a "
              "result. A 137-test regression suite and a 39-check live smoke "
              "test make the behaviour reproducible."),
        style="BodyText", jc="both", spacing=B))
    p.append(body(
        "The honest limitation of the work should be stated alongside the "
        "claims. The reported F-measure is low in absolute terms; the "
        "ranking metrics show genuine signal, but a deployment would need its "
        "block threshold tuned for the cost ratio between a missed fraud and a "
        "falsely blocked genuine customer, and would need the precision side of "
        "the operating point improved before it could block automatically at "
        "scale. The chain and webhook models cannot yet be given a measured "
        "score at all, and the review loop is the intended route to the labels "
        "that would make one possible."))

    p.append(h2("5.2  Future enhancement"))
    for lead, txt in [
        ("Probability calibration. ",
         "Apply the unsupervised calibration of Paper IV to the blended score, "
         "using the unlabelled portion of the stream, so that the block "
         "threshold corresponds to a stable cost ratio instead of an "
         "arbitrary probability."),
        ("Cost-sensitive decision policy. ",
         "Replace the fixed thresholds with a policy driven by the monetary "
         "cost of a false positive against a false negative, so the system "
         "adapts its operating point to the economics of the promotion being "
         "protected."),
        ("Graph-based features. ",
         "Reintroduce structural features over the account–counterparty graph "
         "that the system already builds for the dashboard, using an "
         "incremental embedding rather than a full re-encode, so that shared "
         "counterparty structure can be scored in the same pass as the "
         "behavioural profile."),
        ("Model persistence. ",
         "Persist the fitted ensemble and the profile buffers so a restart "
         "resumes the model rather than retraining it from the beginning of "
         "the stream, which is currently the main cost of a redeployment."),
        ("Federated label exchange. ",
         "Publish aggregate, differentially-privatised label statistics from "
         "the review loop so that several deployments of the same promotion "
         "can share what they learn without sharing the underlying events."),
        ("Online concept-drift detection. ",
         "Add a drift statistic over the feature distribution, computed on "
         "the unlabelled stream, and use it to trigger an out-of-band refit "
         "earlier than the fixed event or clock cadence."),
        ("Authenticated, multi-tenant service. ",
         "Add authentication, per-tenant isolation and rate limiting to the "
         "HTTP surface so the service can be exposed beyond a single "
         "dashboard."),
        ("Brokered stream adapters. ",
         "Add adapters for mainstream event transports so the same canonical "
         "event can be fed by additional producers without touching the "
         "scoring path."),
    ]:
        p.append(para(run(lead, b=True) + run(txt), style="BodyText", jc="both",
                      spacing=B))
    return "".join(p)

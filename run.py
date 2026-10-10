"""Single entry point for the real-time adaptive 3-model detector.

Run from the repo root:

    .venv\\Scripts\\python.exe run.py            # start on http://127.0.0.1:5051

Optional `config.local.json` in the repo root overrides backend/config.py
defaults (see README.md).
"""
import importlib
import os
import sys

REQUIRED = ["flask", "flask_cors", "numpy", "pandas", "sklearn", "xgboost"]


def _check_packages():
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name.replace("-", "_"))
        except ImportError:
            missing.append(name)
    if missing:
        print("Missing required packages: " + ", ".join(missing))
        print("Install with:  .venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(2)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _check_packages()

    from backend.config import load_config
    from backend.app import create_app
    from backend.sources.replay import available_ieee_files

    cfg = load_config()
    app = create_app(cfg)
    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 5051))

    # report the real state of the data directory, not a generic reminder
    data_dir = (cfg.get("ieee") or {}).get("data_dir") or "data"
    files = available_ieee_files(data_dir)
    if files.get("train_transaction"):
        where = os.path.basename(files["train_transaction"])
        if files.get("train_identity"):
            where += " + train_identity.csv"
        replay_line = f"IEEE-CIS replay : {where}  ({files['train_transaction']})"
    elif files.get("final_fraud"):
        replay_line = f"IEEE-CIS replay : {os.path.basename(files['final_fraud'])} (merged)"
    elif (cfg.get("ieee") or {}).get("sample_fallback", True):
        replay_line = ("IEEE-CIS replay : no CSV under "
                       f"{data_dir}/ - using the deterministic sample (see data/README.md)")
    else:
        replay_line = (f"IEEE-CIS replay : NO CSV under {data_dir}/ and "
                       "ieee.sample_fallback is off - replay-only/handoff will error")

    print("=" * 64)
    print("  Real-time Adaptive 3-Model Detector (XGB + HistGB + ExtraTrees)")
    print(f"  Dashboard : http://{host}:{port}      (build frontend first)")
    print(f"  API schema: http://{host}:{port}/api/schema")
    print(f"  SSE feed  : http://{host}:{port}/api/stream")
    print("  Modes     : replay-only | chain-only | handoff | concurrent")
    print("-" * 64)
    print("  " + replay_line)
    scam = (cfg.get("chain") or {}).get("scam_list_path")
    print("  Scam list : " + (f"{scam} ({'found' if scam and os.path.isfile(scam) else 'not found'})"
                             if scam else "not configured (chain labels start empty)"))
    print("  Ethereum  : keyless public gateway with ordered RPC failover")
    print("=" * 64)

    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
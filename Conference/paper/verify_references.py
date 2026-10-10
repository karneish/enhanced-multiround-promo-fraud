#!/usr/bin/env python
"""
Verify every bibliography entry against Crossref.

Why Crossref and not a CS-specific index: this host is rate-limited (HTTP 429)
by OpenAlex, and Crossref answers reliably. Crossref's coverage of the
machine-learning *proceedings* (NeurIPS, KDD, ICML, ICLR) is poor, so an entry
with no Crossref match is reported as NOT INDEXED rather than as an error -
that is the expected outcome for a conference preprint, not evidence of a bad
citation.

Matching is deliberately strict: a candidate must contain essentially all of
our title tokens (forward containment) and must not be a much longer, different
work (reverse containment). Crossref's fuzzy search otherwise returns unrelated
papers that share a single word.

    python Conference/paper/verify_references.py
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".crossref_cache.json")
MAILTO = "paper-verify@example.org"

STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "with", "to", "from",
        "based", "via", "using", "into", "over", "is", "are", "toward", "towards"}


# ------------------------------------------------------------------ helpers
def load_cache():
    if os.path.exists(CACHE):
        try:
            c = json.load(open(CACHE))
            return {k: v for k, v in c.items() if not isinstance(v, dict)}
        except Exception:
            return {}
    return {}


def strip(s):
    return re.sub(r"<[^>]+>", "", s or "")


def ours_title(ref):
    """Quoted title, falling back to the whole entry for books/datasets."""
    m = re.search(r"“(.+?)”", ref)
    if m:
        return m.group(1).strip(" ,.")
    return strip(ref)[:160]


def ours_surname(ref):
    """First author's surname, tolerating the 'F. Lastname' IEEE form."""
    r = strip(ref)
    m = re.match(r"\s*(?:([A-Z]\.\s*(?:[A-Z]\.\s*)*)|"
                 r"([A-Z][\wÀ-ÿ'’\-]+))\s*(?:,|and\b)", r)
    if not m:
        return ""
    if m.group(1):
        m2 = re.match(r"([A-Z][\wÀ-ÿ'’\-]+)", r[m.end(1):].strip())
        return m2.group(1) if m2 else ""
    return m.group(2)


def ours_year(ref):
    ys = re.findall(r"(?:19|20)\d{2}", strip(ref))
    return int(ys[-1]) if ys else None


def toks(s):
    t = strip(s).lower().replace("“", " ").replace("”", " ")
    return {w for w in re.findall(r"[a-z0-9]{3,}", t) if w not in STOP}


def sim(ours, theirs):
    a, b = toks(ours), toks(theirs)
    if not a or not b:
        return 0.0, 0.0
    n = len(a & b)
    return n / len(a), n / len(b)


def search(title, surname, attempts=4):
    q = f"{surname} {title}" if surname else title
    url = ("https://api.crossref.org/works?rows=12&mailto=" + MAILTO
           + "&select=title,author,issued,container-title,volume,page,DOI,type"
           + "&query.bibliographic=" + urllib.parse.quote(q))
    delay = 3.0
    last = None
    for _ in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": "AMDL-TPNE-verify/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())["message"]["items"]
        except Exception as e:                        # noqa: BLE001
            last = e
            time.sleep(delay)
            delay *= 2
    raise last if last else RuntimeError("search failed")


def summarise(it):
    au = it.get("author") or []
    fam = au[0].get("family", "") if au else ""
    given = au[0].get("given", "") if au else ""
    ti = it.get("title") or [""]
    ct = it.get("container-title") or [""]
    yr = None
    dp = (it.get("issued") or {}).get("date-parts") or [[None]]
    if dp and dp[0]:
        yr = dp[0][0]
    return {
        "title": ti[0] if ti else "",
        "surname": fam,
        "given": given,
        "year": yr,
        "venue": ct[0] if ct else "",
        "vol": it.get("volume"), "page": it.get("page"),
        "doi": it.get("DOI"), "type": it.get("type"),
    }


# --------------------------------------------------------------------- main
def main():
    cache = load_cache()
    rows = []

    for i, ref in enumerate(C.REFERENCES, 1):
        t, sur, yr = ours_title(ref), ours_surname(ref), ours_year(ref)
        key = f"{sur}|{t}"

        if key in cache:
            hits = cache[key]
        else:
            try:
                hits = [summarise(h) for h in search(t, sur)]
                cache[key] = hits
                json.dump(cache, open(CACHE, "w"), indent=1)
            except Exception as e:                    # noqa: BLE001
                rows.append((i, ref, t, "ERROR", str(e), {}, sur, yr))
                continue
            time.sleep(0.5)

        cands = []
        for h in hits:
            f, b = sim(t, h["title"])
            cands.append((f, b, h))
        cands.sort(key=lambda c: c[0], reverse=True)

        strict = [c for c in cands if c[0] >= 0.90 and c[1] >= 0.50]
        status, note, best = "NOT INDEXED", "", {}

        if strict:
            f, b, best = strict[0]
            notes = []
            if sur and best["surname"] and best["surname"].lower() != sur.lower():
                status = "AUTHOR MISMATCH"
                notes.append(f"first author is '{best['surname']}', we printed '{sur}'")
            if yr and best["year"] and best["year"] != yr:
                status = "YEAR MISMATCH" if status == "OK" else status
                notes.append(f"Crossref year {best['year']}, we printed {yr}")
            status = status if status != "NOT INDEXED" else "OK"
            note = "; ".join(notes)
        elif cands and cands[0][0] >= 0.70:
            f, b, best = cands[0]
            status = "PARTIAL"
            note = f"best title match only {f:.0%} / reverse {b:.0%}"
        elif cands:
            f, b, best = cands[0]
            status = "NOT INDEXED"
            note = f"no close match (best {f:.0%}); likely a venue Crossref does not index"

        rows.append((i, ref, t, status, note, best, sur, yr))

    marks = {"OK": "  ok  ", "PARTIAL": "part  ", "NOT INDEXED": "nidx ",
             "AUTHOR MISMATCH": "AUTH! ", "YEAR MISMATCH": "YEAR! ", "ERROR": " ERR  "}
    counts = {}
    print("=" * 100)
    print("Crossref verification of the AMDL-TPNE bibliography")
    print("=" * 100)
    for i, ref, t, st, note, hit, sur, yr in rows:
        counts[st] = counts.get(st, 0) + 1
        print(f"[{i:2d}] {marks.get(st,'???? ')} {t[:72]}")
        if st != "OK":
            print(f"          {note}")
            if hit:
                print(f"          hit: {hit.get('title','')[:70]} | "
                      f"{hit.get('surname','')} | {hit.get('year')} | "
                      f"{hit.get('venue','')[:40]} | {hit.get('doi')}")
    print("-" * 100)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad = sum(v for k, v in counts.items() if k in ("AUTHOR MISMATCH", "YEAR MISMATCH", "ERROR"))
    print(f"\nconfirmed {counts.get('OK', 0)} | contradicted {bad} | "
          f"needs manual check {counts.get('PARTIAL', 0) + counts.get('NOT INDEXED', 0)} | "
          f"total {len(C.REFERENCES)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
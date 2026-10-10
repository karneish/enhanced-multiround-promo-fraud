#!/usr/bin/env python
"""
Verify every bibliography entry against OpenAlex.

OpenAlex is used rather than Crossref because Crossref indexes little of the
machine-learning venue literature (NeurIPS, KDD, ICML, ICLR, JMLR), and its
fuzzy title search returns unrelated papers that merely share a word.

Each entry is checked three ways:
  * does a work with (essentially) our title exist?
  * does its first author match the surname we printed?
  * does its year match the year we printed?

A mismatch is reported, never auto-corrected: a flagged entry needs a human to
decide whether the paper or the citation is at fault.

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
CACHE = os.path.join(HERE, ".openalex_cache.json")
MAILTO = "paper-verify@example.org"

STOP = {"the", "a", "an", "of", "for", "and", "in", "on", "with", "to", "from",
        "based", "via", "using", "into", "over", "is", "are"}


# ------------------------------------------------------------------ helpers
def load_cache():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE))
        except Exception:
            return {}
    return {}


def save_cache(c):
    json.dump(c, open(CACHE, "w"), indent=1)


def strip(s):
    return re.sub(r"<[^>]+>", "", s or "")


def ours_title(ref):
    """Pull the quoted title out of an IEEE entry."""
    q = ref.replace("<i>", "\u201c").replace("</i>", "\u201d")
    m = re.search(r"\u201c(.+?)\u201d", q)
    if m:
        return m.group(1).strip(" ,.")
    return strip(ref)[:160]


def ours_surname(ref):
    """First author's surname, tolerating the 'F. Lastname' IEEE form."""
    r = strip(ref)
    m = re.match(r"\s*(?:<[^>]+>\s*)*"
                 r"(?:([A-Z]\.\s*(?:[A-Z]\.\s*)*)|([A-Z][\w\u00c0-\u00de'\u2019\-]+))"
                 r"\s*(?:,|and\b)", r)
    if not m:
        return ""
    if m.group(1):                      # initials first: surname follows them
        rest = r[m.end(1):].strip()
        m2 = re.match(r"([A-Z][\w\u00c0-\u00de'\u2019\-]+)", rest)
        return m2.group(1) if m2 else ""
    return m.group(2)                     # surname first


def ours_year(ref):
    ys = re.findall(r"(?:19|20)\d{2}", strip(ref))
    return int(ys[-1]) if ys else None


def venue_keys(ref):
    """Venue tokens we printed, used to break ties between candidate records."""
    r = strip(ref).lower()
    keys = []
    for pat, key in [
        (r"sigkdd|knowledge discovery and data mining", "kdd"),
        (r"neural information processing systems|neurips|nips", "neurips"),
        (r"learning representations", "iclr"),
        (r"int\. conf\. machine learning|icml", "icml"),
        (r"advances in neural information", "neurips"),
        (r"ieee access", "ieee access"),
        (r"transactions on computational social systems", "tcss"),
        (r"computer aided verification", "cav"),
        (r"ieee transactions|ieee internet of things", "ieee trans"),
        (r"journal of machine learning research", "jmlr"),
        (r"statistics surveys", "statistics surveys"),
        (r"pattern recognition letters", "prl"),
        (r"communications surveys", "comst"),
        (r"sigkdd explorations", "sigkdd expl"),
    ]:
        if re.search(pat, r):
            keys.append(key)
    return keys


def venue_score(venue, want):
    v = (venue or "").lower()
    for w in want:
        if w in v:
            return 1.0
    if "arxiv" in v:
        return -0.5
    return 0.0


def toks(s):
    t = strip(s).lower().replace("\u201c", " ").replace("\u201d", " ")
    return {w for w in re.findall(r"[a-z0-9]{3,}", t) if w not in STOP}


def sim(ours, theirs):
    """Bidirectional token containment: catches both paraphrases and the case
    where Crossref/OpenAlex returns a different paper sharing some words."""
    a, b = toks(ours), toks(theirs)
    if not a:
        return 0.0, 0.0
    inter = len(a & b)
    return inter / len(a), inter / len(b)


# ------------------------------------------------------------------- search
def search(title):
    url = ("https://api.openalex.org/works?per_page=8&mailto=" + MAILTO
           + "&search=" + urllib.parse.quote(title))
    req = urllib.request.Request(url, headers={"User-Agent": "AMDL-TPNE-verify/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode()).get("results", [])


def summarise(w):
    au = w.get("authorships") or []
    name = au[0]["author"]["display_name"] if au else ""
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    return {
        "title": w.get("display_name") or "",
        "year": w.get("publication_year"),
        "author": name,
        "venue": venue,
        "doi": w.get("doi"),
    }


# --------------------------------------------------------------------- main
def main():
    cache = load_cache()
    rows = []

    for i, ref in enumerate(C.REFERENCES, 1):
        t, sur, yr = ours_title(ref), ours_surname(ref), ours_year(ref)

        if t not in cache:
            try:
                cache[t] = [summarise(w) for w in search(t)]
            except Exception as e:
                cache[t] = {"error": str(e)}
            time.sleep(0.6)
        hits = cache[t]

        if isinstance(hits, dict):
            rows.append((i, ref, t, "ERROR", hits.get("error", "?"), {}, sur, yr))
            continue
        if not hits:
            rows.append((i, ref, t, "NOT FOUND", "", {}, sur, yr))
            continue

        # Prefer a record whose venue matches the one we cited: OpenAlex often
        # returns an arXiv preprint of a conference paper, which carries the
        # preprint year rather than the publication year.
        want_venue = venue_keys(ref)
        cands = []
        for h in hits:
            ff, bb = sim(t, h["title"])
            cands.append((ff, bb, venue_score(h.get("venue") or "", want_venue), h))
        # Title match dominates; venue only breaks ties between equally-titled
        # records (typically a conference paper versus its arXiv preprint).
        cands.sort(key=lambda c: (round(c[0], 2) >= 0.90, c[0], c[2]), reverse=True)
        f, b, _v, best = cands[0] if cands else (0.0, 0.0, 0.0, {})

        exact = f >= 0.90 and b >= 0.55
        near = f >= 0.70
        status, note = "OK", ""

        if not near:
            status = "NOT FOUND"
            note = "no work in OpenAlex matches this title"
        else:
            got_sur = (best["author"] or "").split()[-1] if best["author"] else ""
            if not exact:
                status = "PARTIAL"
                note = f"closest title only {f:.0%} similar"
            if got_sur and sur and got_sur.lower().replace("\u00df", "ss") != sur.lower().replace("\u00df", "ss"):
                status = "AUTHOR MISMATCH" if status == "OK" else status
                note = (note + "; " if note else "") + \
                    f"first author is '{got_sur}', we printed '{sur}'"
            if best["year"] and yr and best["year"] != yr:
                status = "YEAR MISMATCH" if status == "OK" else status
                note = (note + "; " if note else "") + \
                    f"OpenAlex year {best['year']}, we printed {yr}"
        rows.append((i, ref, t, status, note, best or {}, sur, yr))

    save_cache(cache)

    marks = {"OK": "  ok  ", "PARTIAL": "part  ", "NOT FOUND": "NONE  ",
             "AUTHOR MISMATCH": "AUTH! ", "YEAR MISMATCH": "YEAR! ", "ERROR": " ERR  "}
    counts = {}
    print("=" * 100)
    print("OpenAlex verification of the AMDL-TPNE bibliography")
    print("=" * 100)
    for i, ref, t, st, note, hit, sur, yr in rows:
        counts[st] = counts.get(st, 0) + 1
        print(f"[{i:2d}] {marks.get(st,'????  ')} {t[:74]}")
        if st != "OK":
            print(f"           note : {note}")
            print(f"           hit  : {hit.get('title','')[:74]}")
            print(f"                  {hit.get('author','')} | {hit.get('year')} | "
                  f"{hit.get('venue','')[:46]} | {hit.get('doi')}")
    print("-" * 100)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    bad = sum(v for k, v in counts.items() if k in ("AUTHOR MISMATCH", "YEAR MISMATCH", "ERROR"))
    unver = counts.get("NOT FOUND", 0) + counts.get("PARTIAL", 0)
    print(f"\nconfirmed {counts.get('OK', 0)} | contradicted {bad} | "
          f"needs manual check {unver} | total {len(C.REFERENCES)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
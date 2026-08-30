"""
Validate paper.tex against the venue's submission rules.

Checks word counts (body and abstract), keyword count, anonymity, and that every
citation resolves. Run before every submission -- an anonymity slip or an
undefined \\cite is the kind of thing that is invisible in a PDF skim.

    python check_paper.py
"""

import re
import sys

TEX = "paper.tex"
BS = chr(92)
R = BS + BS                 # regex source for one literal backslash


def strip_tex(s: str) -> list[str]:
    s = re.sub(r"%.*", "", s)
    s = re.sub(R + r"[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = re.sub(r"[" + BS + r"{}$&_^~]", " ", s)
    return [x for x in s.split() if re.search(r"[A-Za-z0-9]", x)]


def main() -> None:
    raw = open(TEX, encoding="utf-8").read()
    # Strip comments before any content scan. The file's own header comment
    # tells the author not to add acknowledgements or funding statements, which
    # a naive scan reports as exactly the thing it warns against.
    t = re.sub(r"(?m)^\s*%.*$", "", raw)
    fails = []

    body = t[t.find(BS + "begin{document}"): t.find(BS + "begin{thebibliography}")]
    body = re.sub(R + r"begin\{table\}.*?" + R + r"end\{table\}", " ", body, flags=re.S)
    m = re.search(R + r"begin\{abstract\}(.*?)" + R + r"end\{abstract\}", t, re.S)
    body_only = re.sub(
        R + r"begin\{abstract\}.*?" + R + r"end\{abstract\}", " ", body, flags=re.S
    )

    n_body = len(strip_tex(body_only))
    n_abs = len(strip_tex(m.group(1))) if m else 0
    total = n_body + n_abs

    print("WORD COUNT  (references, tables and appendices excluded)")
    print("  body                : %d" % n_body)
    print("  abstract            : %d" % n_abs)
    print("  total               : %d      target 3,000-8,000" % total)
    if not 3000 <= total <= 8000:
        fails.append("word count %d outside 3,000-8,000" % total)
    if n_abs > 250:
        fails.append("abstract %d words, limit 250" % n_abs)

    kw = re.search(r"Keywords:\}(.*?)\n\s*\n", t, re.S)
    n_kw = len([k for k in kw.group(1).split(";") if k.strip()]) if kw else 0
    print("  keywords            : %d      target 4-6" % n_kw)
    if not 4 <= n_kw <= 6:
        fails.append("keyword count %d outside 4-6" % n_kw)

    print("\nANONYMITY")
    identifying = ["bayram", "aliyev", "valiyyaddin", "nitro5", "eliyev", "bayrii",
                   "annotator1", "annotator2", "github.com/"]
    hits = [s for s in identifying if s in t.lower()]
    print("  identifying strings : %s" % (", ".join(hits) if hits else "none"))
    if hits:
        fails.append("identifying strings present: %s" % hits)
    # ICRR requires an anonymised Funding declaration, so the presence of the
    # word is expected. What must not appear is a named funder or grant number.
    for label, pat, want in (("empty \\author", R + r"author\{\}", True),
                             ("acknowledgements", r"acknowledg", False),
                             ("named funder/grant no.",
                              r"grant (no|number|#)|funded by the|"
                              r"supported by (a |an |the )?[A-Z]", False)):
        got = bool(re.search(pat, t, re.I))
        print("  %-19s : %s" % (label, got))
        if got != want:
            fails.append("%s -> %s" % (label, got))

    print("\nCITATIONS")
    cited = set()
    for grp in re.findall(R + r"cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
        cited.update(k.strip() for k in grp.split(",") if k.strip())

    # Keys may come from a manual thebibliography or, as now, a .bib file.
    defined = set(re.findall(R + r"bibitem\[[^\]]*\]\{([^}]*)\}", t))
    try:
        bib = open("references.bib", encoding="utf-8").read()
        defined |= set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
        todos = len(re.findall(r"TODO", bib))
        if todos:
            print("  NOTE: %d TODO field(s) in references.bib "
                  "(missing page numbers)" % todos)
    except FileNotFoundError:
        pass
    print("  cited     : %d" % len(cited))
    print("  defined   : %d" % len(defined))
    print("  undefined : %s" % (sorted(cited - defined) or "none"))
    print("  uncited   : %s" % (sorted(defined - cited) or "none"))
    if cited - defined:
        fails.append("undefined citations: %s" % sorted(cited - defined))

    print("\nSECTIONS")
    for s in re.findall(R + r"section\{([^}]*)\}", t):
        print("  %s" % s)

    print()
    if fails:
        print("FAILED %d check(s):" % len(fails))
        for f in fails:
            print("  - %s" % f)
        sys.exit(1)
    print("all submission checks passed")


if __name__ == "__main__":
    main()

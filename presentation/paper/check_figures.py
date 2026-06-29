#!/usr/bin/env python
"""
check_figures.py
----------------
Fail the build if any figure referenced by main.tex is missing from
presentation/figs/. Keeps the figure<->experiment mapping honest: every
\\includegraphics{...} must resolve to a real file.

    python presentation/paper/check_figures.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.tex"
FIGS = (HERE / "figs").resolve()  # bundled, self-contained copy

INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def main() -> int:
    text = MAIN.read_text()
    refs = INCLUDE_RE.findall(text)
    if not refs:
        print("No \\includegraphics references found in main.tex.")
        return 1

    missing = []
    for name in refs:
        # \graphicspath points at ../figs, so names are relative to there.
        if not (FIGS / name).exists():
            missing.append(name)

    print(f"main.tex references {len(refs)} figures; figs dir = {FIGS}")
    if missing:
        print(f"MISSING {len(missing)}:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("OK: every referenced figure exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

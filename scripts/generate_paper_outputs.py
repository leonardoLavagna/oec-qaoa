"""Generate paper-facing figures and tables from validated workflow outputs."""

from __future__ import annotations

from pathlib import Path

from oec_qaoa.plotting import build_all_paper_outputs


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "figures"


def main() -> None:
    build_all_paper_outputs(RESULTS, OUTPUT)
    print(f"Paper-facing outputs written to {OUTPUT}")
    for path in sorted(OUTPUT.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
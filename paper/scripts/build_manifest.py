#!/usr/bin/env python3
"""Write or check SHA-256 hashes of the manuscript build outputs.

``--write`` records the hashes of the committed figures, generated macros, and
``main.pdf`` in ``build_manifest.json``, together with the plotting and TeX
versions that produced them. ``--check`` recomputes the hashes after a rebuild.

Figures and generated macros depend only on the pinned Python environment
(``requirements-figures.txt``), so a mismatch there is always an error.
``main.pdf`` also depends on the TeX distribution. A mismatch in it is an
error only when the recorded pdfTeX version is installed; otherwise it is
reported as expected drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_environment import environment_mismatches, pinned_versions  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
MANIFEST = PAPER / "build_manifest.json"
PYTHON_OUTPUTS = (
    [f"figures/fig1_overview.{ext}" for ext in ("pdf", "png")]
    + [
        f"figures/{stem}.{ext}"
        for stem in ("fig2_block_size", "fig3_policy_frontier", "fig4_diversity", "fig5_temperature_frontier")
        for ext in ("pdf", "png")
    ]
)
TEX_OUTPUTS = ["main.pdf"]


def generated_macros() -> list[str]:
    return sorted(str(path.relative_to(PAPER)) for path in (PAPER / "generated").glob("*.tex"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdftex_version() -> str:
    try:
        output = subprocess.run(["pdftex", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "not available"
    return output.splitlines()[0].strip()


def current_hashes(paths: list[str]) -> dict[str, str]:
    return {path: sha256(PAPER / path) if (PAPER / path).exists() else "missing" for path in paths}


def write() -> None:
    manifest = {
        "note": "Regenerate with `make manifest` after an intended change to the figures or text.",
        "environment": {
            "python_packages": pinned_versions(),
            "pdftex": pdftex_version(),
            "source_date_epoch": "set by the Makefile",
        },
        "python_outputs": current_hashes(PYTHON_OUTPUTS + generated_macros()),
        "tex_outputs": current_hashes(TEX_OUTPUTS),
    }
    if environment_mismatches():
        raise SystemExit("Refusing to write a manifest from an unpinned plotting environment.")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(PAPER)}")


def compare(expected: dict[str, str]) -> list[str]:
    actual = current_hashes(list(expected))
    return [path for path, digest in expected.items() if actual[path] != digest]


def check() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    status = 0
    unpinned = environment_mismatches()
    python_drift = compare(manifest["python_outputs"])
    if python_drift:
        status = 1
        print("FAIL: figures or generated macros differ from the manifest:\n  " + "\n  ".join(python_drift))
        if unpinned:
            print("The plotting environment is not the pinned one, which explains the difference:\n  "
                  + "\n  ".join(unpinned))
    else:
        print(f"OK: {len(manifest['python_outputs'])} figure and macro files are bit-identical.")

    tex_drift = compare(manifest["tex_outputs"])
    recorded_tex = manifest["environment"]["pdftex"]
    installed_tex = pdftex_version()
    if not tex_drift:
        print("OK: main.pdf is bit-identical.")
    elif python_drift:
        print("NOTE: main.pdf differs because the figures it embeds differ (see above).")
    elif installed_tex == recorded_tex:
        status = 1
        print("FAIL: main.pdf differs although the recorded pdfTeX version is installed.")
    else:
        print("NOTE: main.pdf differs. It is bit-identical only with the recorded TeX build\n"
              f"  recorded:  {recorded_tex}\n  installed: {installed_tex}\n"
              "  Compare page count and extracted text instead.")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write()
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()

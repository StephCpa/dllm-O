#!/usr/bin/env python3
"""Write or check the reproducibility manifest of the manuscript build.

``--write`` records the SHA-256 hashes of the committed figures, generated
macros, and ``main.pdf``. It also records the page count and per-page hashes of
the PDF's normalized text, and the environment that produced them. ``--check``
recomputes everything after a rebuild.

Exit codes of ``--check``:

* ``0`` PASS: every figure, macro, and ``main.pdf`` is bit-identical.
* ``1`` FAIL: a figure or macro differs or is missing; ``main.pdf`` differs
  although the recorded build environment is in use; or, in another TeX
  environment, the page count or the normalized text differs.
* ``3`` PDF NOT VERIFIED: figures and macros are bit-identical, but the TeX
  environment differs from the recorded one, so ``main.pdf`` cannot be checked
  bit for bit. Page count and normalized text match, but the layout still needs
  manual inspection. The same code is used when the text tools are missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_environment import environment_mismatches, pinned_versions  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
MANIFEST = PAPER / "build_manifest.json"

FIGURE_STEMS = (
    "fig1_overview",
    "fig2_block_size",
    "fig3_policy_frontier",
    "fig4_diversity",
    "fig5_temperature_frontier",
)
MACRO_NAMES = (
    "generation_accounting_rows",
    "paper_numbers",
    "postreview_majority_vote_rows",
    "postreview_math_difficulty_rows",
    "temperature_numbers",
)
PYTHON_OUTPUTS = tuple(
    [f"figures/{stem}.{ext}" for stem in FIGURE_STEMS for ext in ("pdf", "png")]
    + [f"generated/{name}.tex" for name in MACRO_NAMES]
)
PDF = "main.pdf"

# ACL review mode prints line numbers in the page margins (x < 26 pt and
# x > 565 pt on A4), while all content lies between 70 and 527 pt. Text is
# extracted from x = 60..535 pt only, so the line numbers, which move whenever
# the layout moves, never enter the comparison.
TEXT_BLOCK_X = (60, 535)
PAGE_HEIGHT = 842

PASS, FAIL, NOT_VERIFIED = 0, 1, 3


# ---------------------------------------------------------------- helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: str) -> str:
    target = PAPER / path
    return sha256_bytes(target.read_bytes()) if target.exists() else "missing"


def missing_artifacts(paths) -> list[str]:
    return [path for path in paths if not (PAPER / path).exists()]


def run(command: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None


def pdftex_version() -> str:
    result = run(["pdftex", "--version"])
    return result.stdout.splitlines()[0].strip() if result else "not available"


def pdftotext_version() -> str | None:
    try:
        result = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
    except OSError:
        return None
    lines = (result.stderr or result.stdout).splitlines()
    return lines[0].strip() if lines else None


def normalize_text(text: str) -> str:
    """Normalize extracted text so that only content differences remain.

    NFKC folds ligatures, and removing whitespace and hyphens neutralizes
    line breaks and TeX hyphenation. Minus signs (U+2212) and dashes are kept,
    so sign changes in numbers are still detected.
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s\-­‐‑]", "", text)


def pdf_page_count(pdf: Path) -> int | None:
    result = run(["pdfinfo", str(pdf)])
    if not result:
        return None
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def pdf_creation_date(pdf: Path) -> str | None:
    result = run(["pdfinfo", "-isodates", str(pdf)])
    if not result:
        return None
    match = re.search(r"^CreationDate:\s+(\S+)", result.stdout, flags=re.MULTILINE)
    return match.group(1) if match else None


def pdf_text_fingerprint(pdf: Path) -> dict | None:
    """Return the page count and per-page hashes of normalized text, or None."""
    pages = pdf_page_count(pdf)
    if pages is None or pdftotext_version() is None:
        return None
    left, right = TEXT_BLOCK_X
    page_hashes = []
    for page in range(1, pages + 1):
        result = run(
            ["pdftotext", "-q", "-enc", "UTF-8", "-r", "72", "-x", str(left), "-y", "0",
             "-W", str(right - left), "-H", str(PAGE_HEIGHT), "-f", str(page), "-l", str(page),
             str(pdf), "-"]
        )
        if result is None:
            return None
        page_hashes.append(sha256_bytes(normalize_text(result.stdout).encode("utf-8")))
    return {"pages": pages, "page_sha256": page_hashes}


def pdf_verdict(*, figures_ok: bool, hash_match: bool, same_build: bool,
                recorded_text: dict, current_text: dict | None) -> tuple[int, str]:
    """Classify main.pdf. Pure function so that every branch can be tested."""
    if not figures_ok:
        return FAIL, "main.pdf not evaluated: the figures it embeds differ (see above)."
    if hash_match:
        return PASS, "OK: main.pdf is bit-identical."
    if same_build:
        return FAIL, ("FAIL: main.pdf differs although the recorded TeX build and "
                      "SOURCE_DATE_EPOCH are in use.")
    if current_text is None:
        return NOT_VERIFIED, ("PDF NOT VERIFIED: main.pdf differs under a TeX build or SOURCE_DATE_EPOCH "
                              "other than the recorded one, and pdfinfo/pdftotext are unavailable for the "
                              "text comparison.")
    if current_text["pages"] != recorded_text["pages"]:
        return FAIL, (f"FAIL: main.pdf has {current_text['pages']} pages; the manifest records "
                      f"{recorded_text['pages']}.")
    differing = [
        index + 1
        for index, (current, recorded) in enumerate(zip(current_text["page_sha256"], recorded_text["page_sha256"]))
        if current != recorded
    ]
    if differing:
        return FAIL, ("FAIL: the normalized text of main.pdf differs on page(s) "
                      + ", ".join(map(str, differing)) + ".")
    return NOT_VERIFIED, ("PDF NOT VERIFIED BIT-FOR-BIT: main.pdf was built with a TeX build or "
                          "SOURCE_DATE_EPOCH other than the recorded one. Page count and normalized text "
                          "match the manifest; inspect the layout manually.")


# ------------------------------------------------------------------ write


def write() -> None:
    missing = missing_artifacts(PYTHON_OUTPUTS + (PDF,))
    if missing:
        raise SystemExit("Refusing to write a manifest; missing build outputs:\n  " + "\n  ".join(missing))
    if environment_mismatches():
        raise SystemExit("Refusing to write a manifest from an unpinned plotting environment.")
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch:
        raise SystemExit("Refusing to write a manifest: SOURCE_DATE_EPOCH is not set (run `make manifest`).")
    expected_date = datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    creation = pdf_creation_date(PAPER / PDF)
    if creation is None or not creation.startswith(expected_date):
        raise SystemExit(f"Refusing to write a manifest: main.pdf CreationDate {creation} does not match "
                         f"SOURCE_DATE_EPOCH {epoch}; rebuild with `make` first.")
    text = pdf_text_fingerprint(PAPER / PDF)
    if text is None:
        raise SystemExit("Refusing to write a manifest: pdfinfo and pdftotext are required.")
    manifest = {
        "note": "Regenerate with `make manifest` after an intended change to the figures or text.",
        "environment": {
            "python_packages": pinned_versions(),
            "pdftex": pdftex_version(),
            "pdftotext": pdftotext_version(),
            "source_date_epoch": epoch,
        },
        "python_outputs": {path: file_hash(path) for path in PYTHON_OUTPUTS},
        "tex_outputs": {PDF: file_hash(PDF)},
        "pdf_text": text,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(PAPER)}")


# ------------------------------------------------------------------ check


def check() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    status = PASS

    recorded = manifest["python_outputs"]
    absent = sorted(set(PYTHON_OUTPUTS) - set(recorded))
    if absent:
        print("FAIL: the manifest does not cover:\n  " + "\n  ".join(absent))
        status = FAIL
    drift = [path for path, digest in recorded.items() if file_hash(path) != digest]
    if drift:
        status = FAIL
        print("FAIL: figures or generated macros differ from the manifest or are missing:\n  "
              + "\n  ".join(drift))
        unpinned = environment_mismatches()
        if unpinned:
            print("The plotting environment is not the pinned one, which explains the difference:\n  "
                  + "\n  ".join(unpinned))
    elif not absent:
        print(f"OK: {len(recorded)} figure and macro files are bit-identical.")

    environment = manifest["environment"]
    installed_tex = pdftex_version()
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    same_build = installed_tex == environment["pdftex"] and epoch == environment["source_date_epoch"]
    hash_match = file_hash(PDF) == manifest["tex_outputs"][PDF]
    current_text = None if hash_match or same_build else pdf_text_fingerprint(PAPER / PDF)
    code, message = pdf_verdict(
        figures_ok=not drift and not absent,
        hash_match=hash_match,
        same_build=same_build,
        recorded_text=manifest["pdf_text"],
        current_text=current_text,
    )
    print(message)
    if not hash_match and not same_build:
        print(f"  recorded TeX:  {environment['pdftex']} (SOURCE_DATE_EPOCH={environment['source_date_epoch']})\n"
              f"  installed TeX: {installed_tex} (SOURCE_DATE_EPOCH={epoch})")
        recorded_poppler = environment.get("pdftotext")
        if code == FAIL and recorded_poppler != pdftotext_version():
            print(f"  Note: pdftotext also differs (recorded {recorded_poppler}, installed "
                  f"{pdftotext_version()}), which can change text extraction.")

    if code == FAIL or status == FAIL:
        status = FAIL
    elif code == NOT_VERIFIED:
        status = NOT_VERIFIED
    print({PASS: "STATUS: PASS", FAIL: "STATUS: FAIL", NOT_VERIFIED: "STATUS: PDF NOT VERIFIED"}[status])
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

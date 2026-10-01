# dLLM Order Transmission: ACL Working Materials

This repository is a curated, anonymized materials package for the current
ACL/ARR working manuscript on decoding schedules, repeated-sampling returns,
and order freedom in diffusion language models.

## Contents

- `paper/`: ACL-format LaTeX source, bibliography, generated numerical macros,
  figures, and the current PDF.
- `reports/`: audited result summaries, frozen-protocol records, CPU analyses,
  and manuscript-review notes.
- `protocols/`: prospective-study protocols, artifact schemas, and freeze files.
- `scripts/`: analysis scripts used to regenerate reported CPU summaries.
- `src/` and `tests/`: the auditable analysis and runner package plus tests.
- `CURRENT_PROGRESS.md`: current experimental and writing status.
- `REFERENCE_AUDIT.md`: citation-format and source-verification notes.

## Rebuild the manuscript

From the repository root:

```bash
cd paper
make
```

The build regenerates the audited manuscript-number macros from the committed
reports before running `latexmk`. The current source is an anonymous ACL/ARR
working draft; venue-specific CFP, page-limit, metadata, and artifact rules
must be rechecked before submission.

## Re-run CPU analyses

The analysis scripts are deterministic CPU-side audits over the committed JSON
reports. They do not require model weights or a GPU. Install the package and
run the relevant script from the repository root, for example:

```bash
python -m pip install -e .
python scripts/postreview_cpu_audits.py --help
python scripts/query_level_exploratory_audits.py --help
```

## Data and reproducibility boundary

This package includes the audited aggregate/result JSON needed to reproduce the
paper tables, figures, and CPU diagnostics. It intentionally excludes model
weights, private server paths, raw GPU generation traces, local dataset caches,
credentials, and large backup archives. The protocol and report files record
the corresponding artifact hashes and provenance where applicable.

The exploratory reports are explicitly labeled as post-outcome analyses. They
must not be presented as preregistered confirmatory evidence.

## Current scientific scope

The strongest supported claims concern the interaction between block-parallel
commit schedules, selection policies, sampling budget, and temperature in the
tested diffusion-language-model setting. Claims are intentionally scoped to
the evaluated model, tasks, decoding implementation, and metric definitions.

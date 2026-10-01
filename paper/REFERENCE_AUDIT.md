# Reference Audit for the ACL/ARR Draft

Audit date: 2026-09-28

## Automated checks

- The draft contains 29 bibliography entries and 29 distinct citation keys.
- Every citation key used in `main.tex` is present in `references.bib`.
- Every bibliography entry is cited; there are no uncited entries.
- All 29 DOI/URL targets resolved successfully during the accessibility check, including arXiv, OpenReview, PMLR, NeurIPS, ICLR, and ICML landing pages.
- Crossref title lookup matched the publisher DOI records for the three proceedings entries using `10.52202/...` identifiers (`Diffusion-LM`, `Simplified and Generalized Masked Diffusion`, and `Simple and Effective Masked Diffusion Language Models`).
- The ACL bibliography build completes without BibTeX warnings or undefined citations.

## Interpretation

The reference list is internally closed and mechanically valid. The current ACL style renders the entries as standard author--year citations; URLs and DOI fields are retained in the BibTeX source for provenance and are not forced to appear as visible links in the PDF.

The arXiv DOI namespace (`10.48550/arXiv...`) is retained where present, while the corresponding arXiv landing URLs were also checked. OpenReview links resolve through the current public landing-page redirect.

## Final pre-submission checks

Before committing to a venue, recheck the publication status and canonical metadata for 2025--2026 papers that may have moved from arXiv/OpenReview to a conference proceedings record. If a final proceedings version exists, replace the preprint-only record only when its authors, title, and venue metadata are confirmed to match. No such replacement was inferred from the automated accessibility check alone.

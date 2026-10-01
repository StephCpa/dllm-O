# ACL/ARR Working Manuscript

This directory is the ACL/ARR-format working copy of the decoding-policy
manuscript. It is the active submission-shaped source in this package.

## Build

```bash
make
```

The Makefile regenerates audited manuscript-number macros before running
`latexmk`. `main.tex` uses the official ACL review style and keeps the paper
anonymous for ARR review. Use `latexmk -C main.tex` before a clean rebuild when
switching style files or bibliography configuration.

## Submission constraints

- The current PDF keeps the main text within the ACL eight-page limit; references and the appendix begin after the main text.
- `Limitations` is a dedicated section, separate from `Discussion`.
- The AI Use Statement is retained as a separate disclosure section.
- The current copy is a working draft, not a venue-specific camera-ready package. Recheck the selected venue's current CFP, checklist, metadata, and artifact requirements before submission.

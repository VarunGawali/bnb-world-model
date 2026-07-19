# AAAI 2027 paper source

`anonymous-submission.tex` — the paper in AAAI 2027 anonymous-submission format,
with the abstract complete and the sections drafted. `references.bib` — the
bibliography.

## Required files (from the AAAI author kit)

This directory needs two files from the **AAAI 2027 author kit** — the same
download that contained the template/checklist PDFs:

    aaai2027.sty     % AAAI Press LaTeX style
    aaai2027.bst     % AAAI Press BibTeX style

Drop them next to `anonymous-submission.tex`. (If your kit ships `aaai2026.*`
instead, update the two filenames in the `.tex`/bibliographystyle accordingly.)

## Compile

    pdflatex anonymous-submission
    bibtex   anonymous-submission
    pdflatex anonymous-submission
    pdflatex anonymous-submission

Or upload the folder to Overleaf (add the two AAAI kit files first).

## Anonymity checklist before submitting (from the template)

- Author block stays **"Anonymous Submission"** — no names/affiliations.
- **Clear the PDF metadata** (e.g. `exiftool -all= anonymous-submission.pdf`).
- Anonymize the Code/Datasets URLs and any self-referential citations.
- No AAAI copyright footer on page 1 for the anonymous version.

## To finish for the full paper (July 28)

- Fill `Table~\ref{tab:main}` with real numbers once training completes.
- Report mean $\pm$ std over seeds / held-out set and Wilcoxon $p$-values
  (see `../REPRODUCIBILITY.md`, items 4.10--4.12).
- Add exact library versions (`pip freeze`) to the Reproducibility paragraph.

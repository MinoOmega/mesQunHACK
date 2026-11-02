# More Than A Hack 2027 — possession submission

This folder is the standalone hand-in package. It contains the reproducible estimator, its technical
description, setup instructions, and the generated possession CSV. Match data is intentionally not
included; point the command at the data bundle you received.

## Reproduce the submission

Python 3.8 or newer is required. Install the pinned runtime dependency:

```bash
python3 -m pip install -r requirements.txt
```

Run the estimator against a tactical data folder (or its parent containing `tactical/` and
`broadcast/`):

```bash
python3 possession.py \
  --data-dir /path/to/MoreThanAHack/tactical \
  --out submission.csv
```

For the supplied match this writes one row per tactical ATD/video frame. The estimator reads the
team mapping, pitch dimensions, and frame rate from the provided metadata. It requires the raw ATD
and Smart Tagging XML, but no video decoder and no homography.

The local `submission.csv` is the generated full-match output for the provided match. Running
the command with the same input bundle and default parameters regenerates it byte-for-byte.
Upload `submission.csv` through the competition's possession-file field; keep it separate from the
public code repository. The public repository should contain the source, documentation, and
reproduction instructions, not match data or generated match outputs.

## Files

- `possession.py` — implementation and command-line interface.
- `POSSESSION.md` — algorithm, asset choices, limitations, and rules compliance.
- `technical_report.tex` / `technical_report.pdf` — physicist-oriented explanation of the task and
  estimator.
- `submission.csv` — full-frame output to upload as the possession file.
- `test_possession.py` — synthetic invariant tests for transitions, restarts, and dead-ball rules.
- `requirements.txt` — pinned Python runtime dependency.

Run the state-model tests with `python3 -m unittest -v test_possession.py`.
Rebuild the PDF with `pdflatex -interaction=nonstopmode technical_report.tex` (run twice for final
PDF outlines).

Public labels were used beyond pipeline checking during development: earlier versions were scored
and inspected, and the 1.2-second persistence duration and dead-ball investigation were informed by
those results. This conflicts with the challenge's stated restriction. The final revision was not
rescored, but inherits that history; ask the organizers whether it is acceptable before submitting.
The details are documented in `technical_report.pdf`.

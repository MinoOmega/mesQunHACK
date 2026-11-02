# Technical document: tracking-first possession estimation

## Approach

The estimator labels each tactical video frame from the automatic FIFA/EPTS tracking and Metrica
Smart Tagging. It is deterministic, uses fixed heuristic thresholds, and has no learned model. The
code does not read video, hand annotations, or ground-truth labels at runtime.

### 1. Team and pitch metadata

The tracking XML maps each player track slot to one of the two teams and provides the pitch size and
frame rate. The first listed team maps to home (`H`), the second to away (`A`). The number of output
frames and their order are inferred from the raw tracking file.

### 2. Ball control evidence

For every frame with an observed ball, player-to-ball distances are computed in metres from the
normalized pitch coordinates. For each team, the nearest tracked player contributes evidence that
decays exponentially with distance. A nearest player farther than 4.5 metres contributes no control
evidence. Team phase tags can modestly reinforce direct evidence, but cannot create a team label on
their own when the ball is unobserved.

### 3. Temporal state and restarts

The team evidence is smoothed over a centered 0.28-second radius, separately inside uninterrupted
live segments so evidence cannot bleed across a dead-ball interval. A change of possession is
accepted only after a clear opposing-team advantage persists for 1.2 seconds. When that evidence
confirms a change, the label boundary is assigned to the start of the sustained evidence rather than
the end of the confirmation delay. Ambiguous contests and missing ball samples retain the current
team label. Before a team has been established, the system abstains with `X`.

Automatic tags are evidence, not frame labels. A `GOALS` interval marks a stoppage. A `SET PIECES`
interval is a phase that can include live play, so it is dead only from the tag's start until the
estimator detects a controlled restart. It detects that restart using three consecutive frames of
observed ball movement (provided speed or a short displacement-derived estimate) while the ball is
within 4.5 m of a player. When available, the set-piece `Team` tag adds a weak prior around that
restart. If tracked ball motion never confirms a restart, the tagged interval remains `D`. A
dead-ball interval resets the live possessor so the next restart begins a new possession.

For a `Kick Off (After Goal)` tagged `N/A`, the restart side is inferred as the opponent of the team
tagged in a `GOALS` interval ending within ten seconds of the kickoff tag. This follows the restart
rule and avoids assigning all such kickoffs to one hard-coded club.

Sustained ball-track runs outside the normalized pitch rectangle also emit `D`; isolated excursions
are ignored. These persistence rules are intended to reject brief touches, deflections, and tracking
flicker, while responding to sustained control by the opposing side.

## Assets used and omitted

- **Used:** tactical ATD metadata, raw player/ball tracking, and automatic Smart Tagging.
- **Video omitted:** the estimator does not require image decoding; tracking and tags provide its
  evidence while keeping execution lightweight and portable.
- **Homography omitted:** it maps normalized pitch coordinates into video pixels. No image pixels
  are inspected, so it supplies no independent signal to this estimator.
- **Broadcast omitted:** its timeline is separate and unsynchronized with tactical frames.

No human-labeled dataset or learned model was used.

## Portability and reproduction

The command accepts a Metrica-style tactical data directory, including the provided bundle layout.
It selects tactical files if both tactical and broadcast folders are present. It does not hard-code
team names, IDs, frame ranges, or output length. A new match in the same format can be run without
editing the source:

```bash
python3 -m pip install -r requirements.txt
python3 possession.py --data-dir /path/to/new-match/tactical --out new-match.csv
```

For the provided match, the generated output is `submission.csv`. Given the same files and default
parameters, the command reproduces that CSV exactly.

The synthetic state invariants can be checked without match data or labels:

```bash
python3 -m unittest -v test_possession.py
```

## Limitations

Nearest-player distance is an imperfect proxy for controlled possession. It can fail on aerial
balls, tackles, crowded rebounds, long passes, and tracking errors. A ball outside the normalized
pitch can also be an airborne localization error; requiring a sustained run reduces but cannot
eliminate that ambiguity. Smart Tagging is inferred and noisy; its intervals are treated as evidence,
not authoritative possession labels. Public labels were used during development beyond checking
format/alignment: earlier versions were scored, their possession/dead-ball summaries were examined,
the persistence duration was changed from 0.55 to 1.2 seconds after inspecting one result, and public
dead-ball prevalence informed later stoppage investigation. This conflicts with the repository's
instruction that the public sample is only for pipeline checks. The final revision was not rescored,
but inherits that history and cannot be claimed to be independently developed. Ask the organizers
whether this history is acceptable before submitting. No automated parameter sweep was performed.
Labeled data from other matches could support future documented model calibration, but was not used.

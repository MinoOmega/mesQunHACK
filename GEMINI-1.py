"""Physics-informed state machine for team possession estimation.

Replaces the naive nearest-player majority vote with:
  1. Spatial scaling and ball kinematics (velocity in m/s).
  2. Dead-ball (D) detection via pitch boundaries, stationary intervals, and gap durations.
  3. Inertial possession propagation across occluded ball frames and aerial passes.
  4. Run-length hysteresis to filter out deflections, rebounds, and brief touches.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1
DEAD = 2
LABELS = np.array(["H", "A", "D"])
FPS = 25.0
DT = 1.0 / FPS


def team_of_each_track(meta_path):
    """0 if the track slot belongs to the home team, 1 if away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) club names."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def get_runs(arr):
    """Return (values, starts, lengths) of contiguous constant segments in 1D array."""
    if len(arr) == 0:
        return np.array([]), np.array([]), np.array([])
    change_idx = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(arr)]))
    return arr[starts], starts, ends - starts


def extract_kinematics(raw_path, team_of_track, n_frames, pitch):
    """Parse raw tracking to extract ball coordinates (m) and closest player distances (m) per team."""
    length, width = pitch
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    dist_h = np.full(n_frames, np.inf, dtype=np.float32)
    dist_a = np.full(n_frames, np.inf, dtype=np.float32)
    n_players = np.zeros(n_frames, dtype=np.int16)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame_idx = int(parts[0]) - 1
            if frame_idx < 0 or frame_idx >= n_frames:
                continue

            # Count active players and store their positions temporarily if ball is present
            player_groups = parts[1].split(GROUP_SEP)
            ball = parts[2]
            has_ball = (ball[0] != MISSING)

            if has_ball:
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx = float(ball[:c1]) * length
                by = float(ball[c1 + 1:c2]) * width
                ball_x[frame_idx] = bx
                ball_y[frame_idx] = by

                best_h = np.inf
                best_a = np.inf
                active = 0
                for slot, group in enumerate(player_groups):
                    if group[0] == MISSING:
                        continue
                    active += 1
                    p_c1 = group.index(",")
                    p_c2 = group.index(",", p_c1 + 1)
                    dx = float(group[:p_c1]) * length - bx
                    dy = float(group[p_c1 + 1:p_c2]) * width - by
                    d2 = dx * dx + dy * dy
                    if team_of_track[slot] == HOME:
                        if d2 < best_h:
                            best_h = d2
                    else:
                        if d2 < best_a:
                            best_a = d2

                dist_h[frame_idx] = np.sqrt(best_h)
                dist_a[frame_idx] = np.sqrt(best_a)
                n_players[frame_idx] = active
            else:
                active = sum(1 for g in player_groups if g[0] != MISSING)
                n_players[frame_idx] = active

    # Compute ball speed (m/s) on visible frames
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    seen_idx = np.flatnonzero(~np.isnan(ball_x))
    if len(seen_idx) > 1:
        dx = np.diff(ball_x[seen_idx])
        dy = np.diff(ball_y[seen_idx])
        dt_frames = np.diff(seen_idx)
        # Direct step velocity where gap is small (<= 5 frames)
        valid_step = dt_frames <= 5
        step_speed = np.zeros_like(dx)
        step_speed[valid_step] = np.hypot(dx[valid_step], dy[valid_step]) / (dt_frames[valid_step] * DT)
        ball_speed[seen_idx[1:][valid_step]] = step_speed[valid_step]
        ball_speed[seen_idx[0]] = 0.0

    return ball_x, ball_y, ball_speed, dist_h, dist_a, n_players


def detect_dead_ball(ball_x, ball_y, ball_speed, dist_h, dist_a, pitch,
                     max_live_gap=75, min_dead_frames=30, stationary_frames=45):
    """Identify dead-ball (D) frames using geometry, kinematics, and tracking gap durations."""
    length, width = pitch
    n_frames = len(ball_x)
    seen = ~np.isnan(ball_x)
    is_dead = np.zeros(n_frames, dtype=bool)

    # 1. Ball outside pitch boundaries (with 0.5m tolerance for line width/optical noise)
    out_of_bounds = seen & (
        (ball_x < -0.5) | (ball_x > length + 0.5) |
        (ball_y < -0.5) | (ball_y > width + 0.5)
    )
    is_dead[out_of_bounds] = True

    # 2. Missing ball gaps: short gaps are in-flight/occluded; long gaps are dead ball
    vals, starts, lengths = get_runs(seen)
    for val, start, run_len in zip(vals, starts, lengths):
        if not val:  # Missing ball segment
            if run_len >= max_live_gap:
                is_dead[start:start + run_len] = True
            elif start > 0 and (start + run_len) < n_frames and run_len >= 20:
                # Check if ball teleported or moved at impossible speed during medium gap
                dx = ball_x[start + run_len] - ball_x[start - 1]
                dy = ball_y[start + run_len] - ball_y[start - 1]
                implied_v = np.hypot(dx, dy) / (run_len * DT)
                if implied_v > 38.0 or implied_v < 0.8:
                    is_dead[start:start + run_len] = True

    # 3. Stationary ball waiting for a set-piece restart
    min_dist = np.minimum(dist_h, dist_a)
    stationary = seen & (ball_speed < 0.35) & (min_dist > 1.2)
    s_vals, s_starts, s_lengths = get_runs(stationary)
    for val, start, run_len in zip(s_vals, s_starts, s_lengths):
        if val and run_len >= stationary_frames:
            is_dead[start:start + run_len] = True

    # 4. Morphological cleanup: bridge tiny live blips inside dead ball, remove tiny D blips
    d_vals, d_starts, d_lengths = get_runs(is_dead)
    for val, start, run_len in zip(d_vals, d_starts, d_lengths):
        if not val and run_len <= 15:
            is_dead[start:start + run_len] = True

    d_vals, d_starts, d_lengths = get_runs(is_dead)
    for val, start, run_len in zip(d_vals, d_starts, d_lengths):
        if val and run_len < min_dead_frames:
            is_dead[start:start + run_len] = False

    return is_dead


def infer_possession(ball_x, ball_y, ball_speed, dist_h, dist_a, n_players, pitch,
                     control_radius=2.4, duel_margin=0.35, max_control_speed=15.0,
                     min_possession_frames=35, max_live_gap=75):
    """State machine that assigns H, A, D, or X to every frame."""
    n_frames = len(ball_x)
    is_dead = detect_dead_ball(
        ball_x, ball_y, ball_speed, dist_h, dist_a, pitch, max_live_gap=max_live_gap
    )

    # Instantaneous control evidence on live frames
    raw_owner = np.full(n_frames, NOBODY, dtype=np.int8)
    seen = ~np.isnan(ball_x)
    slow_enough = np.isnan(ball_speed) | (ball_speed <= max_control_speed)
    controllable = (~is_dead) & seen & slow_enough

    h_close = controllable & (dist_h <= control_radius) & ((dist_a - dist_h) >= duel_margin)
    a_close = controllable & (dist_a <= control_radius) & ((dist_h - dist_a) >= duel_margin)

    # Fallback when ball speed is slightly higher or margin tighter, still within 1.5m
    h_tight = (~is_dead) & seen & (dist_h <= 1.5) & (dist_h < dist_a)
    a_tight = (~is_dead) & seen & (dist_a <= 1.5) & (dist_a < dist_h)

    raw_owner[h_close | h_tight] = HOME
    raw_owner[a_close | a_tight] = AWAY

    # Process each contiguous live-ball period independently (separated by Dead Ball 'D')
    labels = np.full(n_frames, "D", dtype="<U1")
    dead_vals, seg_starts, seg_lengths = get_runs(is_dead)

    for is_d, start, length in zip(dead_vals, seg_starts, seg_lengths):
        if is_d:
            continue
        end = start + length
        seg_owner = raw_owner[start:end].copy()

        valid_idx = np.flatnonzero(seg_owner != NOBODY)
        if len(valid_idx) == 0:
            # No clear controller during this live segment -> remain D
            continue

        # 1. Inertial propagation: fill forward from each controlled touch, backward at start
        first_valid = valid_idx[0]
        seg_owner[:first_valid] = seg_owner[first_valid]
        for i in range(len(valid_idx) - 1):
            idx_curr = valid_idx[i]
            idx_next = valid_idx[i + 1]
            seg_owner[idx_curr:idx_next] = seg_owner[idx_curr]
        seg_owner[valid_idx[-1]:] = seg_owner[valid_idx[-1]]

        # 2. Deflection & rebound filter: absorb isolated touches shorter than min_possession_frames
        # Iterate from shortest blips up to threshold to prevent fragmentation
        for threshold in (5, 12, 20, min_possession_frames):
            r_vals, r_starts, r_lengths = get_runs(seg_owner)
            if len(r_vals) <= 1:
                break
            for idx_run in range(len(r_vals)):
                if r_lengths[idx_run] < threshold:
                    s_r = r_starts[idx_run]
                    e_r = s_r + r_lengths[idx_run]
                    # Absorb into previous possession (or next if at the very start of play)
                    replacement = r_vals[idx_run - 1] if idx_run > 0 else r_vals[idx_run + 1]
                    seg_owner[s_r:e_r] = replacement

        labels[start:end] = np.where(seg_owner == HOME, "H", "A")

    # Mark pre-match / halftime / post-match frames with no players tracked as 'X'
    labels[n_players == 0] = "X"
    return labels


def write_submission(path, labels):
    """Write frame,label CSV."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=RESULTS_DIR, help="output directory")
    parser.add_argument("--min-poss", type=int, default=35,
                        help="minimum frames for a valid possession change (default: 35 = 1.4s)")
    parser.add_argument("--max-gap", type=int, default=75,
                        help="max missing-ball frames before declaring Dead Ball (default: 75 = 3.0s)")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("extracting kinematics from {} ...".format(os.path.basename(raw_path)))

    ball_x, ball_y, ball_speed, dist_h, dist_a, n_players = extract_kinematics(
        raw_path, team_of_each_track(meta_path), n_frames, pitch
    )

    labels = infer_possession(
        ball_x, ball_y, ball_speed, dist_h, dist_a, n_players, pitch,
        min_possession_frames=args.min_poss,
        max_live_gap=args.max_gap
    )

    out_path = write_submission(os.path.join(args.out, "physics_state_machine.csv"), labels)
    for k in ("H", "A", "D", "X"):
        print("  {}: {:>7} frames".format(k, int(np.sum(labels == k))))
    print("saved -> {}".format(os.path.relpath(out_path, ROOT)))
    print("\nValidate and score with:")
    print("  python evaluation/validate_submission.py {}".format(os.path.relpath(out_path, ROOT)))
    print("  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()

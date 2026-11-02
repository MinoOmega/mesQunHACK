"""Reusable Metrica tracking-first possession estimator.

Run from the package folder:
    python3 possession.py --data-dir /path/to/match-data/tactical --out submission.csv

Only NumPy and the Python standard library are required. No video, hand labels, or match-specific
team/frame constants are used.
"""

import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
HOME, AWAY, DEAD, UNKNOWN = 0, 1, 2, -1
OUTPUT_LABELS = np.array(["H", "A", "D"])
MAX_CONTROL_DISTANCE_METERS = 4.5
DISTANCE_DECAY_METERS = 1.8
PHASE_TEAM_BONUS = 0.25
RESTART_TEAM_BONUS = 0.35
EVIDENCE_FLOOR = 0.12
SWITCH_MARGIN = 0.12
RESTART_SPEED_MPS = 0.8
RESTART_CONFIRM_FRAMES = 3
RESTART_PRIOR_SECONDS = 1.2
OUTSIDE_CONFIRM_FRAMES = 5
PITCH_EDGE_TOLERANCE = 0.005


def _one_file(root, suffix):
    paths = glob.glob(os.path.join(root, "**", "*" + suffix), recursive=True)
    tactical = [path for path in paths if "tactical" in os.path.normpath(path).split(os.sep)]
    if tactical:
        paths = tactical
    if len(paths) != 1:
        raise ValueError("expected exactly one *{} under {}; found {}".format(suffix, root, len(paths)))
    return paths[0]


def _metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    players = list(root.iter("Player"))
    if len(teams) != 2 or not players:
        raise ValueError("{}: expected two teams and player track slots".format(path))
    names = [team.findtext("Name", default="") for team in teams]
    ids = [team.get("id") for team in teams]
    if any(player.get("teamId") not in ids for player in players):
        raise ValueError("{}: player track slot refers to an unknown team".format(path))
    if ids[0] == ids[1]:
        raise ValueError("{}: the two team identifiers must be distinct".format(path))
    slot_teams = np.asarray([0 if player.get("teamId") == ids[0] else 1 for player in players],
                            dtype=np.int8)
    fps = root.findtext(".//FrameRate")
    length, width = root.findtext(".//FieldSize/Width"), root.findtext(".//FieldSize/Height")
    pitch = (float(length) if length else 105.0, float(width) if width else 68.0)
    return names, slot_teams, float(fps) if fps else 25.0, pitch


def _read_tracking(path, slot_teams):
    with open(path) as handle:
        n_frames = sum(1 for _ in handle)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    player_xy = np.full((n_frames, len(slot_teams), 2), np.nan, dtype=np.float32)
    previous_id = None
    with open(path) as handle:
        for row, line in enumerate(handle):
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}:{}: expected three colon-separated fields".format(path, row + 1))
            frame_id = int(parts[0])
            if frame_id != row + 1:
                raise ValueError("{}:{}: expected ATD frame_id {}, found {}".format(
                    path, row + 1, row + 1, frame_id))
            if previous_id is not None and frame_id != previous_id + 1:
                raise ValueError("{}: tracking frame ids are not contiguous at {}".format(path, frame_id))
            previous_id = frame_id
            groups = parts[1].split(";")
            if len(groups) != len(slot_teams):
                raise ValueError("{}:{}: track slot count differs from metadata".format(path, row + 1))
            ball_fields = parts[2].split(",")
            if ball_fields[0] != "NaN":
                ball[row] = (float(ball_fields[0]), float(ball_fields[1]))
                if len(ball_fields) > 2 and ball_fields[2] != "NaN":
                    ball_speed[row] = float(ball_fields[2])
            for slot, group in enumerate(groups):
                fields = group.split(",")
                if fields[0] != "NaN":
                    player_xy[row, slot] = (float(fields[0]), float(fields[1]))
    if n_frames == 0:
        raise ValueError("{}: tracking file is empty".format(path))
    if row + 1 != n_frames:
        raise ValueError("{}: tracking changed while being read".format(path))
    return ball, player_xy, ball_speed


def _tag_evidence(path, n_frames, fps, team_names):
    priors = np.zeros((n_frames, 2), dtype=np.float32)
    goal_intervals = []
    raw_set_pieces = []
    aliases = {name.strip().casefold(): i for i, name in enumerate(team_names)}
    for instance in ET.parse(path).iter("instance"):
        start = int(round(float(instance.findtext("start")) * fps))
        end = int(round(float(instance.findtext("end")) * fps))
        lo, hi = max(0, start), min(n_frames, max(start + 1, end))
        if lo >= hi:
            continue
        code = (instance.findtext("code") or "").strip().casefold()
        tags = {item.findtext("group", default="").strip().casefold():
                item.findtext("text", default="").strip() for item in instance.findall("label")}
        team = aliases.get(tags.get("team", "").casefold())
        if code == "set pieces":
            raw_set_pieces.append((lo, hi, team, tags.get("type", "").casefold()))
        elif code == "goals":
            # Goal phases run from the scoring event through the dead-ball celebration, usually
            # ending at the tagged kickoff. Unlike ordinary phase tags, a goal is a stoppage.
            goal_intervals.append((lo, hi, team))
        elif team is not None:
            priors[lo:hi, team] = np.maximum(priors[lo:hi, team], PHASE_TEAM_BONUS)
    set_piece_events = []
    for start, end, restart_team, event_type in raw_set_pieces:
        if restart_team is None and event_type == "kick off (after goal)":
            preceding_goals = [(abs(goal_end - start), goal_team)
                               for _, goal_end, goal_team in goal_intervals
                               if goal_team in (HOME, AWAY) and abs(goal_end - start) <= fps * 10]
            if preceding_goals:
                restart_team = 1 - min(preceding_goals, key=lambda item: item[0])[1]
        set_piece_events.append((start, end, restart_team))
    return priors, set_piece_events, [(start, end) for start, end, _ in goal_intervals]


def _ball_motion_speed(ball_meters, provided_speed, fps):
    """Fill missing speed values using a short multi-frame displacement estimate."""
    speed = provided_speed.astype(np.float32, copy=True)
    observed = np.isfinite(ball_meters).all(axis=1)
    for lag in (4, 3, 2, 1):
        if len(speed) <= lag:
            continue
        valid = observed[lag:] & observed[:-lag]
        derived = np.full(len(speed) - lag, np.nan, dtype=np.float32)
        derived[valid] = (np.linalg.norm(ball_meters[lag:][valid] - ball_meters[:-lag][valid],
                                         axis=1) * fps / lag)
        missing = ~np.isfinite(speed[lag:]) & np.isfinite(derived)
        speed[lag:][missing] = derived[missing]
    return speed


def _confirmed_out_of_play(ball, confirm_frames=OUTSIDE_CONFIRM_FRAMES,
                           edge_tolerance=PITCH_EDGE_TOLERANCE):
    """Return sustained ball-track runs beyond the normalized pitch rectangle."""
    observed = np.isfinite(ball).all(axis=1)
    outside = observed & ((ball[:, 0] < -edge_tolerance) |
                          (ball[:, 0] > 1.0 + edge_tolerance) |
                          (ball[:, 1] < -edge_tolerance) |
                          (ball[:, 1] > 1.0 + edge_tolerance))
    confirmed = np.zeros(len(ball), dtype=bool)
    starts = np.flatnonzero(outside & ~np.r_[False, outside[:-1]])
    ends = np.flatnonzero(outside & ~np.r_[outside[1:], False]) + 1
    for start, end in zip(starts, ends):
        if end - start >= confirm_frames:
            confirmed[start:end] = True
    return confirmed


def _smooth_scores(scores, radius, dead_mask=None):
    if radius <= 0:
        return scores
    result = np.zeros_like(scores)
    live = np.ones(len(scores), dtype=bool) if dead_mask is None else ~dead_mask
    starts = np.flatnonzero(live & ~np.r_[False, live[:-1]])
    ends = np.flatnonzero(live & ~np.r_[live[1:], False]) + 1
    for start, end in zip(starts, ends):
        segment = scores[start:end]
        cumulative = np.vstack((np.zeros((1, 2), dtype=np.float64),
                                np.cumsum(segment, axis=0, dtype=np.float64)))
        index = np.arange(len(segment))
        low = np.maximum(0, index - radius)
        high = np.minimum(len(segment), index + radius + 1)
        result[start:end] = (cumulative[high] - cumulative[low]) / (high - low)[:, None]
    return result


def estimate(ball, players, ball_speed, slot_teams, team_priors, set_piece_events,
             goal_intervals,
             fps, pitch, max_control_distance=MAX_CONTROL_DISTANCE_METERS,
             switch_seconds=1.2, smoothing_seconds=0.28):
    n = len(ball)
    if ball.shape != (n, 2) or players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2:
        raise ValueError("ball and player coordinates must have shapes (frames, 2) and (frames, tracks, 2)")
    if (slot_teams.shape != (players.shape[1],) or
            not np.isin(slot_teams, (HOME, AWAY)).all() or
            not np.any(slot_teams == HOME) or not np.any(slot_teams == AWAY)):
        raise ValueError("slot_teams must identify every track as home (0) or away (1)")
    if ball_speed.shape != (n,) or team_priors.shape != (n, 2):
        raise ValueError("ball speed and team priors must have one row per frame")
    if len(pitch) != 2:
        raise ValueError("pitch must have length and width")
    numeric_params = np.asarray([fps, pitch[0], pitch[1], max_control_distance,
                                 switch_seconds, smoothing_seconds], dtype=np.float64)
    if (not np.isfinite(numeric_params).all() or fps <= 0 or min(pitch) <= 0 or
            max_control_distance <= 0):
        raise ValueError("fps, pitch dimensions, and control distance must be positive")
    if switch_seconds < 0 or smoothing_seconds < 0:
        raise ValueError("temporal durations cannot be negative")
    scale = np.asarray(pitch, dtype=np.float32)
    ball_m, players_m = ball * scale, players * scale
    distances = np.full((n, 2), np.inf, dtype=np.float32)
    for team in (HOME, AWAY):
        delta = players_m[:, slot_teams == team, :] - ball_m[:, None, :]
        per_player = np.sqrt(np.sum(delta * delta, axis=2))
        distances[:, team] = np.min(np.where(np.isfinite(per_player), per_player, np.inf), axis=1)

    observed = np.isfinite(ball_m).all(axis=1)
    scores = np.zeros((n, 2), dtype=np.float32)
    for team in (HOME, AWAY):
        distance = distances[:, team]
        scores[:, team] = np.where(observed & (distance < max_control_distance),
                                   np.exp(-distance / DISTANCE_DECAY_METERS), 0.0)
    direct_evidence = scores > 0
    scores += team_priors * direct_evidence
    motion_speed = _ball_motion_speed(ball_m, ball_speed, fps)
    controlled_ball = (observed & np.isfinite(motion_speed) &
                       (motion_speed >= RESTART_SPEED_MPS) &
                       (np.min(distances, axis=1) < max_control_distance))
    restart_priors = np.zeros((n, 2), dtype=np.float32)
    dead_mask = _confirmed_out_of_play(ball)
    for start, end in goal_intervals:
        dead_mask[start:end] = True
    prior_window = max(1, int(round(fps * RESTART_PRIOR_SECONDS)))
    for start, end, restart_team in set_piece_events:
        release = None
        last_start = end - RESTART_CONFIRM_FRAMES + 1
        for frame in range(start, max(start, last_start)):
            if controlled_ball[frame:frame + RESTART_CONFIRM_FRAMES].all():
                release = frame
                break
        if release is None:
            dead_mask[start:end] = True
            continue
        dead_mask[start:release] = True
        if restart_team in (HOME, AWAY):
            prior_end = min(end, release + prior_window)
            restart_priors[release:prior_end, restart_team] = RESTART_TEAM_BONUS
    scores += restart_priors * direct_evidence * controlled_ball[:, None]
    scores = _smooth_scores(scores, int(round(fps * smoothing_seconds)), dead_mask)

    labels = np.full(n, UNKNOWN, dtype=np.int8)
    current = pending_team = UNKNOWN
    pending_frames, pending_start = 0, None
    required = max(1, int(round(fps * switch_seconds)))
    for frame in range(n):
        if dead_mask[frame]:
            labels[frame] = DEAD
            current, pending_team, pending_frames, pending_start = UNKNOWN, UNKNOWN, 0, None
            continue
        home, away = scores[frame]
        if max(home, away) < EVIDENCE_FLOOR:
            labels[frame] = current
            pending_team, pending_frames, pending_start = UNKNOWN, 0, None
            continue
        candidate = HOME if home >= away else AWAY
        margin = abs(float(home - away))
        if current == UNKNOWN:
            current = candidate
        elif candidate == current:
            pending_team, pending_frames, pending_start = UNKNOWN, 0, None
        elif margin >= SWITCH_MARGIN:
            if pending_team == candidate:
                pending_frames += 1
            else:
                pending_team, pending_frames, pending_start = candidate, 1, frame
            if pending_frames >= required:
                labels[pending_start:frame + 1] = candidate
                current = candidate
                pending_team, pending_frames, pending_start = UNKNOWN, 0, None
        else:
            pending_team, pending_frames, pending_start = UNKNOWN, 0, None
        labels[frame] = current

    output = np.full(n, "X", dtype="<U1")
    valid = labels >= 0
    output[valid] = OUTPUT_LABELS[labels[valid]]
    return output


def write_submission(path, labels):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels.tolist()))


def build(data_dir, out, switch_seconds=1.2, smoothing_seconds=0.28):
    meta = _one_file(data_dir, "_atd.xml")
    raw = _one_file(data_dir, "_atd.txt")
    tagging = _one_file(data_dir, "_smart_tagging.xml")
    names, slot_teams, fps, pitch = _metadata(meta)
    ball, players, ball_speed = _read_tracking(raw, slot_teams)
    priors, set_piece_events, goal_intervals = _tag_evidence(tagging, len(ball), fps, names)
    labels = estimate(ball, players, ball_speed, slot_teams, priors, set_piece_events,
                      goal_intervals, fps, pitch,
                      switch_seconds=switch_seconds, smoothing_seconds=smoothing_seconds)
    write_submission(out, labels)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="folder containing, or above, one tactical data bundle")
    parser.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
    parser.add_argument("--switch-seconds", type=float, default=1.2,
                        help="sustained opposing-control duration (default: 1.2 s)")
    parser.add_argument("--smoothing-seconds", type=float, default=0.28,
                        help="centered evidence smoothing radius (default: 0.28 s)")
    args = parser.parse_args()
    result = build(args.data_dir, args.out, args.switch_seconds, args.smoothing_seconds)
    counts = {name: int(np.sum(result == name)) for name in ("H", "A", "D", "X")}
    print("wrote {} ({} frames): {}".format(args.out, len(result), counts))

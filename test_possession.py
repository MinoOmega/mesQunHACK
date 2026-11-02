"""Focused invariants for the possession state model; no match labels are involved."""

import os
import tempfile
import unittest

import numpy as np

import possession


class PossessionStateTests(unittest.TestCase):
    def estimate(self, home_distance, away_distance, *, ball=None, speed=None,
                 team_priors=None, events=None, goals=None, fps=10.0,
                 switch_seconds=1.2):
        n = len(home_distance)
        ball = np.tile([0.5, 0.5], (n, 1)).astype(np.float32) if ball is None else ball
        players = np.empty((n, 2, 2), dtype=np.float32)
        players[:, 0, :] = np.column_stack((0.5 + np.asarray(home_distance) / 100.0,
                                            np.full(n, 0.5)))
        players[:, 1, :] = np.column_stack((0.5 + np.asarray(away_distance) / 100.0,
                                            np.full(n, 0.5)))
        teams = np.asarray([possession.HOME, possession.AWAY], dtype=np.int8)
        speed = np.zeros(n, dtype=np.float32) if speed is None else speed
        phase = np.zeros((n, 2), dtype=np.float32) if team_priors is None else team_priors
        set_piece_events = [] if events is None else events
        goal_intervals = [] if goals is None else goals
        return possession.estimate(ball, players, speed, teams, phase, set_piece_events,
                                    goal_intervals,
                                    fps, (100.0, 100.0), smoothing_seconds=0,
                                    switch_seconds=switch_seconds)

    def test_short_opposing_touch_does_not_change_possession(self):
        home = np.r_[np.ones(20), np.full(7, 6.0), np.ones(18)]
        away = np.r_[np.full(20, 6.0), np.ones(7), np.full(18, 6.0)]
        labels = self.estimate(home, away)
        self.assertTrue(np.all(labels == "H"))

    def test_sustained_turnover_is_backdated_to_first_persistent_evidence(self):
        home = np.r_[np.ones(12), np.full(18, 6.0)]
        away = np.r_[np.full(12, 6.0), np.ones(18)]
        labels = self.estimate(home, away)
        self.assertTrue(np.all(labels[:12] == "H"))
        self.assertTrue(np.all(labels[12:] == "A"))

    def test_missing_ball_carries_established_live_possessor(self):
        ball = np.tile([0.5, 0.5], (20, 1)).astype(np.float32)
        ball[5:15] = np.nan
        labels = self.estimate(np.ones(20), np.full(20, 6.0), ball=ball)
        self.assertTrue(np.all(labels == "H"))

    def test_tagged_dead_ball_resets_the_possessor(self):
        home = np.r_[np.ones(5), np.full(15, 6.0)]
        away = np.r_[np.full(10, 6.0), np.ones(10)]
        labels = self.estimate(home, away, events=[(5, 10, None)])
        self.assertTrue(np.all(labels[:5] == "H"))
        self.assertTrue(np.all(labels[5:10] == "D"))
        self.assertTrue(np.all(labels[10:] == "A"))

    def test_team_tag_alone_cannot_invent_possession_without_ball(self):
        ball = np.full((8, 2), np.nan, dtype=np.float32)
        priors = np.zeros((8, 2), dtype=np.float32)
        priors[:, possession.HOME] = 1.0
        labels = self.estimate(np.ones(8), np.full(8, 6.0), ball=ball,
                               team_priors=priors)
        self.assertTrue(np.all(labels == "X"))

    def test_known_restart_team_can_seed_a_controlled_restart(self):
        speed = np.full(5, 1.2, dtype=np.float32)
        labels = self.estimate(np.ones(5), np.full(5, 2.0), speed=speed,
                               events=[(0, 5, possession.AWAY)])
        self.assertTrue(np.all(labels == "A"))

    def test_set_piece_phase_becomes_live_at_sustained_tracked_motion(self):
        n = 15
        ball = np.tile([0.5, 0.5], (n, 1)).astype(np.float32)
        ball[6:, 0] += np.arange(1, n - 5, dtype=np.float32) * 0.005
        speed = np.full(n, np.nan, dtype=np.float32)
        labels = self.estimate(np.ones(n), np.full(n, 6.0), ball=ball, speed=speed,
                               events=[(0, n, None)])
        self.assertTrue(np.all(labels[:6] == "D"))
        self.assertTrue(np.all(labels[6:] == "H"))

    def test_goal_phase_is_dead_until_tagged_restart(self):
        n = 12
        home = np.r_[np.ones(3), np.full(9, 6.0)]
        away = np.r_[np.full(3, 6.0), np.ones(9)]
        speed = np.zeros(n, dtype=np.float32)
        speed[9:] = 1.2
        labels = self.estimate(home, away, speed=speed, goals=[(2, 7)],
                               events=[(7, 12, None)])
        self.assertTrue(np.all(labels[:2] == "H"))
        self.assertTrue(np.all(labels[2:9] == "D"))
        self.assertTrue(np.all(labels[9:] == "A"))

    def test_sustained_ball_track_outside_pitch_marks_dead_ball(self):
        n = 20
        ball = np.tile([0.5, 0.5], (n, 1)).astype(np.float32)
        ball[3:5, 0] = 1.1  # two-frame tracking excursion is ignored
        ball[8:13, 0] = 1.1  # confirmed five-frame out-of-play run
        home = np.r_[np.ones(13), np.full(7, 6.0)]
        away = np.r_[np.full(13, 6.0), np.ones(7)]
        labels = self.estimate(home, away, ball=ball)
        self.assertTrue(np.all(labels[:8] == "H"))
        self.assertTrue(np.all(labels[8:13] == "D"))
        self.assertTrue(np.all(labels[13:] == "A"))

    def test_temporal_smoothing_does_not_leak_across_dead_ball(self):
        scores = np.asarray([[1, 0], [1, 0], [0, 0], [0, 1], [0, 1]], dtype=np.float32)
        dead = np.asarray([False, False, True, False, False])
        smoothed = possession._smooth_scores(scores, 1, dead)
        self.assertTrue(np.all(smoothed[:2, possession.HOME] == 1))
        self.assertTrue(np.all(smoothed[3:, possession.AWAY] == 1))

    def test_invalid_array_shapes_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "shapes"):
            possession.estimate(np.zeros((2, 3)), np.zeros((2, 2, 2)),
                                np.zeros(2), np.asarray([0, 1]), np.zeros((2, 2)),
                                [], [], 25, (105, 68))

    def test_tag_parser_separates_phase_team_goal_and_restart_signals(self):
        xml = """<root>
          <instance><start>0.6</start><end>0.8</end><code>SET PIECES</code>
            <label><group>Team</group><text>N/A</text></label>
            <label><group>Type</group><text>Kick Off (After Goal)</text></label></instance>
          <instance><start>0.4</start><end>0.5</end><code>GOALS</code>
            <label><group>Team</group><text>Home FC</text></label></instance>
          <instance><start>0.2</start><end>0.6</end><code>PROGRESSION</code>
            <label><group>Team</group><text>Away FC</text></label></instance>
        </root>"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as handle:
            handle.write(xml)
            path = handle.name
        try:
            priors, set_pieces, goals = possession._tag_evidence(
                path, 10, 10.0, ["Home FC", "Away FC"])
        finally:
            os.unlink(path)
        self.assertEqual(set_pieces, [(6, 8, possession.AWAY)])
        self.assertEqual(goals, [(4, 5)])
        self.assertTrue(np.all(priors[2:6, possession.AWAY] == possession.PHASE_TEAM_BONUS))
        self.assertTrue(np.all(priors[:, possession.HOME] == 0))

    def test_tracking_parser_rejects_nonzero_frame_origin(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
            handle.write("2:0.5,0.5,0.0;0.6,0.5,0.0:0.55,0.5,0.0\n")
            path = handle.name
        try:
            with self.assertRaisesRegex(ValueError, "expected ATD frame_id 1"):
                possession._read_tracking(path, np.asarray([0, 1], dtype=np.int8))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

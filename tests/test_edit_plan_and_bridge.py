import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

import opentimelineio as otio
from pydantic import ValidationError

from config.settings import probe_media
from layer2_decision.schemas import validate_edit_plan
from layer3_bridge.bridge import _build_timeline


class EditPlanValidationTests(unittest.TestCase):
    def test_valid_plan_is_sorted_and_normalized(self):
        plan = validate_edit_plan(
            {
                "keep_segments": [
                    {"start": 5, "end": 8, "reason": "second"},
                    {"start": 0, "end": 4, "reason": "first"},
                ],
                "remove_segments": [
                    {"start": 4, "end": 5, "reason": "pause"},
                ],
                "flag_zoom": [],
            },
            duration=8,
        )

        self.assertEqual([segment["start"] for segment in plan["keep_segments"]], [0.0, 5.0])

    def test_rejects_keep_remove_overlap(self):
        with self.assertRaisesRegex(ValidationError, "keep and remove segments overlap"):
            validate_edit_plan(
                {
                    "keep_segments": [{"start": 0, "end": 5, "reason": "keep"}],
                    "remove_segments": [{"start": 4, "end": 6, "reason": "cut"}],
                    "flag_zoom": [],
                },
                duration=10,
            )

    def test_rejects_segment_beyond_duration(self):
        with self.assertRaisesRegex(ValidationError, "ends after video duration"):
            validate_edit_plan(
                {
                    "keep_segments": [{"start": 0, "end": 11, "reason": "keep"}],
                    "remove_segments": [],
                    "flag_zoom": [],
                },
                duration=10,
            )

    def test_rejects_empty_keep_list(self):
        with self.assertRaisesRegex(ValidationError, "at least one keep segment"):
            validate_edit_plan(
                {"keep_segments": [], "remove_segments": [], "flag_zoom": []},
                duration=10,
            )


class TimelineExportTests(unittest.TestCase):
    def test_timeline_references_source_media_and_audio(self):
        source = "/tmp/AVEP source clip.mp4"
        timeline = _build_timeline(
            source,
            [{"start": 1.0, "end": 3.0, "reason": "clean take"}],
            fps=30.0,
            duration=10.0,
            include_audio=True,
        )

        self.assertEqual(len(timeline.tracks), 2)
        expected_url = Path(source).resolve().as_uri()
        for track in timeline.tracks:
            self.assertEqual(track[0].media_reference.target_url, expected_url)

        fcpxml = otio.adapters.write_to_string(timeline, adapter_name="fcpx_xml")
        self.assertIn(expected_url, fcpxml)
        self.assertIn('hasAudio="1"', fcpxml)
        self.assertNotIn("file:///tmp/clean%20take", fcpxml)


class MediaProbeTests(unittest.TestCase):
    def test_probe_uses_format_duration_and_r_frame_rate_fallbacks(self):
        response = {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "N/A",
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "30000/1001",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "12.5"},
        }
        completed = Mock(stdout=json.dumps(response))

        with patch("config.settings.subprocess.run", return_value=completed):
            media = probe_media("video.mp4")

        self.assertEqual(media["duration"], 12.5)
        self.assertEqual(media["fps"], 29.97)
        self.assertTrue(media["has_audio"])


if __name__ == "__main__":
    unittest.main()

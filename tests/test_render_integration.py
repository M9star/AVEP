import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.settings import probe_media
from layer4_execution import ffmpeg_render


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class RenderIntegrationTests(unittest.TestCase):
    def test_render_produces_verified_audio_video_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            intermediate = root / "intermediate"
            output = root / "output"
            intermediate.mkdir()
            output.mkdir()

            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100",
                    "-t", "2", "-c:v", "mpeg4", "-c:a", "aac", str(source),
                ],
                check=True,
            )

            edit_plan = intermediate / "edit_plan.json"
            edit_plan.write_text(json.dumps({
                "keep_segments": [
                    {"start": 0.0, "end": 0.8, "reason": "first"},
                    {"start": 1.2, "end": 2.0, "reason": "second"},
                ],
                "remove_segments": [
                    {"start": 0.8, "end": 1.2, "reason": "pause"},
                ],
                "flag_zoom": [],
            }))

            paths = {
                "output_dir": output,
                "edit_plan": edit_plan,
                "final_output": output / "final_cut.mp4",
            }
            with patch.object(ffmpeg_render, "get_paths", return_value=paths):
                rendered_path = ffmpeg_render.run(str(source))

            rendered = Path(rendered_path)
            metadata = probe_media(str(rendered))
            self.assertTrue(rendered.is_file())
            self.assertGreater(rendered.stat().st_size, 0)
            self.assertTrue(metadata["has_audio"])
            self.assertAlmostEqual(metadata["duration"], 1.6, delta=0.35)


if __name__ == "__main__":
    unittest.main()

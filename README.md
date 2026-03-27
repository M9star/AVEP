# 🎬 AVEP — Autonomous Video Editing Pipeline

A 4-layer agentic system that turns raw footage into a polished edit automatically.

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Add your API key to .env
nano .env

# 3. Drop a video in data/input/
cp ~/Desktop/myvideo.mp4 data/input/

# 4. Run the full pipeline
python pipeline.py --video data/input/myvideo.mp4
```

## Run Individual Layers

```bash
# Layer 1 only — transcription + silence detection
python pipeline.py --video data/input/myvideo.mp4 --layer 1

# Layer 2 only — decision agent (needs Layer 1 output)
python pipeline.py --video data/input/myvideo.mp4 --layer 2

# Layer 2 without LLM (heuristics only, free)
python pipeline.py --video data/input/myvideo.mp4 --layer 2 --skip-llm

# Layer 3 only — generate FCPXML/EDL
python pipeline.py --video data/input/myvideo.mp4 --layer 3

# Layer 4 only — render final cut
python pipeline.py --video data/input/myvideo.mp4 --layer 4
```

## Intermediate Files

All intermediate outputs land in `data/intermediate/`:
- `perception_output.json` — transcript + silence map
- `edit_plan.json` — LLM edit decisions
- `timeline.fcpxml` — import into DaVinci Resolve
- `timeline.edl` — import into any NLE

## GPU Support

| Platform       | Whisper Device | Render Codec         |
|----------------|----------------|----------------------|
| Apple Silicon  | mps            | h264_videotoolbox    |
| NVIDIA GPU     | cuda           | h264_nvenc           |
| CPU only       | cpu            | libx264              |

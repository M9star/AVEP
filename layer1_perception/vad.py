"""
Layer 1 — Voice Activity Detection
Detects silence regions using Silero-VAD.
"""
import torch


def detect_silence(audio_path: str, threshold: float = 0.5) -> list[dict]:
    """
    Returns silence segments:
    [{"start": 12.4, "end": 13.9}, ...]
    """
    print("  [VAD] Loading Silero-VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_ts, _, read_audio, *_) = utils

    wav = read_audio(audio_path, sampling_rate=16000)
    speech_ts = get_speech_ts(wav, model, sampling_rate=16000)

    # Convert speech timestamps → silence gaps
    duration = len(wav) / 16000
    silences = []
    prev_end = 0.0
    for ts in speech_ts:
        s_start = ts["start"] / 16000
        if s_start - prev_end > threshold:
            silences.append({"start": round(prev_end, 3), "end": round(s_start, 3)})
        prev_end = ts["end"] / 16000
    if duration - prev_end > threshold:
        silences.append({"start": round(prev_end, 3), "end": round(duration, 3)})

    print(f"  [VAD] Found {len(silences)} silence regions")
    return silences

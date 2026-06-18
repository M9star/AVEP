"""
Layer 1 — Noise Reduction
Runs DeepFilterNet to clean audio before transcription.
"""
import time
from pathlib import Path


def denoise(input_path: str, output_path: str, progress_cb=None) -> str:
    """
    Runs DeepFilterNet on input audio.
    Returns path to cleaned audio file.
    Falls back gracefully if model unavailable.
    """
    def report(msg, pct=None):
        if progress_cb:
            progress_cb(msg, pct)

    try:
        from df.enhance import enhance, init_df, load_audio, save_audio

        report("Loading DeepFilterNet model...", None)
        model, df_state, _ = init_df()

        report("Loading audio for denoising...", None)
        audio, _ = load_audio(input_path, sr=df_state.sr())
        duration_sec = audio.shape[-1] / df_state.sr()
        report(f"Denoising {duration_sec:.0f}s of audio (DeepFilterNet)...", 0)

        t0 = time.time()
        enhanced = enhance(model, df_state, audio)
        elapsed = time.time() - t0

        report(f"Denoised {duration_sec:.0f}s in {elapsed:.0f}s — saving...", 90)
        save_audio(output_path, enhanced, df_state.sr())
        report(f"Denoise complete ({elapsed:.0f}s)", 100)
        print(f"  [Denoise] Cleaned audio saved → {output_path}")
        return output_path
    except ImportError:
        report("DeepFilterNet not installed — skipping", None)
        print("  [Denoise] DeepFilterNet not installed — skipping noise reduction")
        return input_path

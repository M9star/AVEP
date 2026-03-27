"""
Layer 1 — Noise Reduction
Runs DeepFilterNet to clean audio before transcription.
"""
from pathlib import Path


def denoise(input_path: str, output_path: str) -> str:
    """
    Runs DeepFilterNet on input audio.
    Returns path to cleaned audio file.
    Falls back gracefully if model unavailable.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df()
        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        print(f"  [Denoise] Cleaned audio saved → {output_path}")
        return output_path
    except ImportError:
        print("  [Denoise] DeepFilterNet not installed — skipping noise reduction")
        return input_path

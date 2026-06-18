"""
Layer 2 — Context-Aware Transcript Correction
Uses LLM to fix misheard domain-specific words based on lecture context.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from config.settings import LLM_PROVIDER

load_dotenv()

PROMPT_PATH = Path(__file__).parent / "prompts" / "transcript_correction_prompt.txt"

CHUNK_SIZE = 500


def correct_transcript(words: list[dict], subject_hint: str = "") -> dict:
    """Send transcript in chunks to LLM for context-aware correction."""
    system_prompt = PROMPT_PATH.read_text()
    all_corrected = []
    all_summaries = {}

    chunks = [words[i:i + CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
    print(f"  [Corrector] Processing {len(words)} words in {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"  [Corrector] Chunk {i+1}/{len(chunks)}...")
        user_content = json.dumps({
            "words": chunk,
            "subject_hint": subject_hint,
        }, indent=2)

        result = _call_llm(system_prompt, user_content)
        all_corrected.extend(result.get("corrected_words", chunk))

        for s in result.get("corrections_summary", []):
            key = s["original"].lower()
            if key in all_summaries:
                all_summaries[key]["count"] += s.get("count", 1)
            else:
                all_summaries[key] = s

    corrections_made = sum(1 for w in all_corrected if w.get("corrected"))
    print(f"  [Corrector] Fixed {corrections_made} words across {len(all_summaries)} unique corrections")

    return {
        "corrected_words": all_corrected,
        "corrections_summary": list(all_summaries.values()),
        "total_corrections": corrections_made,
    }


def _call_llm(system: str, user: str) -> dict:
    if LLM_PROVIDER == "claude":
        return _call_claude(system, user)
    elif LLM_PROVIDER == "openai":
        return _call_openai(system, user)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system, user)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _call_claude(system: str, user: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def _call_openai(system: str, user: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _call_gemini(system: str, user: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(
        f"{system}\n\nTranscript data:\n{user}",
        generation_config={"response_mime_type": "application/json"},
    )
    return json.loads(resp.text)

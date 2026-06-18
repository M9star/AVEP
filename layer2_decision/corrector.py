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


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")


def _call_llm(system: str, user: str) -> dict:
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system, user)
    elif LLM_PROVIDER == "claude_code":
        return _call_claude_code(system, user)
    elif LLM_PROVIDER == "claude":
        return _call_claude(system, user)
    elif LLM_PROVIDER == "openai":
        return _call_openai(system, user)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system, user)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _call_ollama(system: str, user: str) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _call_claude_code(system: str, user: str) -> dict:
    """Call Claude via Claude Code CLI — uses existing auth, no API key needed."""
    import subprocess
    prompt = f"{system}\n\nTranscript data:\n{user}"

    prompt_file = Path(__file__).parent.parent / "data" / "intermediate" / ".llm_prompt.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(prompt)

    try:
        result = subprocess.run(
            ["claude", "-p", "--output-format", "json",
             f"Read the file {prompt_file} and follow the instructions in it. Return ONLY valid JSON, no prose."],
            capture_output=True, text=True, timeout=300,
        )
    finally:
        prompt_file.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Claude Code CLI failed: {result.stderr}")

    response = json.loads(result.stdout)
    text = response.get("result", "")

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


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
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{system}\n\nTranscript data:\n{user}",
        config={"response_mime_type": "application/json"},
    )
    return json.loads(resp.text)

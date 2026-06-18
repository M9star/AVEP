"""
Layer 2 — LLM Decision Agent
Sends transcript to Claude, Gemini, or GPT-4o and returns structured edit plan.
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
from config.settings import LLM_PROVIDER

load_dotenv()

PROMPT_PATH = Path(__file__).parent / "prompts" / "editor_prompt.txt"


def call_llm(perception_data: dict) -> dict:
    system_prompt = PROMPT_PATH.read_text()
    user_content   = json.dumps({
        "words":    perception_data["words"],
        "silences": perception_data["silences"],
    }, indent=2)

    if LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, user_content)
    elif LLM_PROVIDER == "claude_code":
        return _call_claude_code(system_prompt, user_content)
    elif LLM_PROVIDER == "claude":
        return _call_claude(system_prompt, user_content)
    elif LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, user_content)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_content)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")


def _call_ollama(system: str, user: str) -> dict:
    """Call a local Ollama model via OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    print(f"  [Agent] Calling Ollama ({OLLAMA_MODEL})...")
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
    """Call Claude via the Claude Code CLI — uses existing auth, no API key needed."""
    import subprocess
    print("  [Agent] Calling Claude via Claude Code CLI...")
    prompt = f"{system}\n\nTranscript data:\n{user}"

    # Write inside project so Claude Code has read access
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
    print("  [Agent] Calling Claude Sonnet 4.6...")
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keep_segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "number"},
                                    "end": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["start", "end", "reason"],
                                "additionalProperties": False,
                            },
                        },
                        "remove_segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "number"},
                                    "end": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["start", "end", "reason"],
                                "additionalProperties": False,
                            },
                        },
                        "flag_zoom": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "number"},
                                    "end": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["start", "end", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["keep_segments", "remove_segments", "flag_zoom"],
                    "additionalProperties": False,
                },
            }
        },
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def _call_openai(system: str, user: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("  [Agent] Calling GPT-4o...")
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _call_gemini(system: str, user: str) -> dict:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("  [Agent] Calling Gemini 2.0 Flash...")
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{system}\n\nTranscript data:\n{user}",
        config={"response_mime_type": "application/json"},
    )
    return json.loads(resp.text)

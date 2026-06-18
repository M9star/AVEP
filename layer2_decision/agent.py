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

    if LLM_PROVIDER == "claude":
        return _call_claude(system_prompt, user_content)
    elif LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, user_content)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_content)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


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
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro")
    print("  [Agent] Calling Gemini 1.5 Pro...")
    resp = model.generate_content(
        f"{system}\n\nTranscript data:\n{user}",
        generation_config={"response_mime_type": "application/json"},
    )
    return json.loads(resp.text)

import os

import requests
from dotenv import load_dotenv

load_dotenv()

NOTES_SYSTEM = (
    "You are a meeting notes assistant. Stay faithful to the transcript. "
    "Do not invent attendees, quotes, or decisions. If something is unclear, say so."
)

NOTES_INSTRUCTION = """Write markdown meeting notes from the transcript below.

Use these sections:
## Overview
## Notes by speaker
## Decisions
## Action items

Rules:
- If the transcript has [SPEAKER_0] / [SPEAKER_1] (or named) labels, organize notes under those labels.
- If a real name is used in the conversation, map it to that speaker.
- If there are no speaker labels, infer speakers only when the transcript makes it obvious; otherwise use "Unknown".
- Action items: owner and due date only when stated. Use `- [ ]` checkboxes.
- Be concise.

Transcript:
"""


def _env(name, default=""):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _split_chunks(text, size=8000):
    blocks = [part.strip() for part in text.split("\n") if part.strip()]
    chunks, current = [], []
    length = 0
    for block in blocks:
        extra = len(block) + 1
        if current and length + extra > size:
            chunks.append("\n".join(current))
            current, length = [block], extra
        else:
            current.append(block)
            length += extra
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def _openai_chat(messages, *, api_key, base_url, model, max_tokens=2500):
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _chat_openai(messages, model=None):
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")
    return _openai_chat(
        messages,
        api_key=api_key,
        base_url=_env("OPENAI_BASE_URL") or None,
        model=model or _env("OPENAI_MODEL", "gpt-4o-mini"),
    )


def _chat_groq(messages, model=None):
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Create a free key at console.groq.com")
    return _openai_chat(
        messages,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model=model or _env("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )


def _chat_freellmapi(messages, model=None):
    api_key = _env("FREELLMAPI_KEY")
    if not api_key:
        raise ValueError("FREELLMAPI_KEY is missing (unified key from the FreeLLMAPI dashboard).")
    return _openai_chat(
        messages,
        api_key=api_key,
        base_url=_env("FREELLMAPI_BASE_URL", "http://localhost:3001/v1"),
        model=model or _env("FREELLMAPI_MODEL", "auto"),
    )


def _chat_gemini(messages, model=None):
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")
    return _openai_chat(
        messages,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model=model or _env("GEMINI_MODEL", "gemini-2.5-flash"),
    )


def _chat_ollama(messages, model=None):
    host = _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = model or _env("OLLAMA_MODEL", "llama3.2:3b")
    response = requests.post(
        f"{host}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    return (payload.get("message") or {}).get("content", "").strip()


def _chat_anthropic(messages, model=None):
    api_key = _env("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is missing.")
    system = ""
    converted = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
            continue
        converted.append({"role": message["role"], "content": message["content"]})
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model or _env("ANTHROPIC_MODEL", "claude-sonnet-4-0"),
            "max_tokens": 2500,
            "system": system,
            "messages": converted,
        },
        timeout=600,
    )
    response.raise_for_status()
    content = response.json().get("content") or []
    return "".join(part.get("text", "") for part in content if part.get("type") == "text").strip()


def _dispatch(provider, messages):
    provider = provider.lower()
    if provider == "groq":
        return _chat_groq(messages)
    if provider == "openai":
        return _chat_openai(messages)
    if provider == "ollama":
        return _chat_ollama(messages)
    if provider == "freellmapi":
        return _chat_freellmapi(messages)
    if provider == "gemini":
        return _chat_gemini(messages)
    if provider == "anthropic":
        return _chat_anthropic(messages)
    raise ValueError(f"Unknown LLM provider: {provider}")


def generate_notes(transcript, provider=None):
    provider = (provider or _env("LLM_PROVIDER", "groq")).lower()
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    working = transcript
    if provider == "ollama" and len(transcript) > 12000:
        partials = []
        for index, chunk in enumerate(_split_chunks(transcript), start=1):
            print(f"Ollama: condensing transcript chunk {index}...")
            partials.append(
                _dispatch(
                    provider,
                    [
                        {"role": "system", "content": NOTES_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                "Condense this meeting transcript section. Keep speakers, "
                                f"decisions, and action items.\n\n{chunk}"
                            ),
                        },
                    ],
                )
            )
        working = "\n\n".join(partials)

    return _dispatch(
        provider,
        [
            {"role": "system", "content": NOTES_SYSTEM},
            {"role": "user", "content": NOTES_INSTRUCTION + working},
        ],
    )

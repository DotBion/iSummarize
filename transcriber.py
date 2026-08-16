import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

AUDIO_SUFFIXES = {".mp3", ".mp4", ".wav", ".m4a", ".aac", ".ogg", ".webm", ".flac"}
NO_DIARIZE_PROVIDERS = {"groq", "whisper"}


@dataclass
class Transcript:
    text: str
    has_speakers: bool


class DiarizeNotSupported(ValueError):
    pass


def _env(name, default=""):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def format_speaker_turns(turns):
    """turns: iterable of (speaker, text). Merge consecutive same-speaker lines."""
    lines = []
    current_speaker = None
    bucket = []
    for speaker, text in turns:
        text = (text or "").strip()
        if not text:
            continue
        speaker = speaker or "UNKNOWN"
        if current_speaker is None:
            current_speaker = speaker
        if speaker != current_speaker:
            lines.append(f"[{current_speaker}] {' '.join(bucket)}")
            bucket = [text]
            current_speaker = speaker
        else:
            bucket.append(text)
    if current_speaker is not None and bucket:
        lines.append(f"[{current_speaker}] {' '.join(bucket)}")
    return "\n".join(lines)


def _openai_client(api_key=None, base_url=None):
    from openai import OpenAI

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _transcribe_openai(audio_path, *, diarize, language):
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it in .env (platform.openai.com, not ChatGPT Plus).")

    model = _env("OPENAI_STT_MODEL")
    if not model:
        model = "gpt-4o-transcribe-diarize" if diarize else "gpt-transcribe"

    client = _openai_client(api_key=api_key, base_url=_env("OPENAI_BASE_URL") or None)
    with open(audio_path, "rb") as audio:
        if diarize or model == "gpt-4o-transcribe-diarize":
            result = client.audio.transcriptions.create(
                model=model or "gpt-4o-transcribe-diarize",
                file=audio,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
            segments = getattr(result, "segments", None) or result.get("segments", [])
            turns = []
            for segment in segments:
                if hasattr(segment, "model_dump"):
                    segment = segment.model_dump()
                turns.append((segment.get("speaker") or "UNKNOWN", segment.get("text") or ""))
            text = format_speaker_turns(turns) or getattr(result, "text", "") or ""
            return Transcript(text=text.strip(), has_speakers=bool(turns))

        kwargs = {"model": model, "file": audio}
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return Transcript(text=(text or "").strip(), has_speakers=False)


def _transcribe_groq(audio_path, *, language):
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Create a free key at console.groq.com")

    model = _env("GROQ_STT_MODEL", "whisper-large-v3")
    client = _openai_client(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    with open(audio_path, "rb") as audio:
        kwargs = {"model": model, "file": audio}
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return Transcript(text=(text or "").strip(), has_speakers=False)


def _transcribe_elevenlabs(audio_path, *, diarize, language, num_speakers=None):
    import requests

    api_key = _env("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is missing.")

    model_id = _env("STT_ELEVENLABS_MODEL", "scribe_v2")
    data = {
        "model_id": model_id,
        "diarize": "true" if diarize else "false",
        "timestamps_granularity": "word" if diarize else "none",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as audio:
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            files={"file": (Path(audio_path).name, audio)},
            data=data,
            timeout=600,
        )
    response.raise_for_status()
    payload = response.json()
    words = payload.get("words") or []
    if diarize and words:
        turns = []
        for word in words:
            if word.get("type") and word.get("type") != "word":
                continue
            turns.append((word.get("speaker_id") or "UNKNOWN", word.get("text") or ""))
        text = format_speaker_turns(turns) or (payload.get("text") or "")
        return Transcript(text=text.strip(), has_speakers=True)
    return Transcript(text=(payload.get("text") or "").strip(), has_speakers=False)


def _transcribe_whisper(audio_path, *, language):
    try:
        import whisper
    except ImportError as exc:
        raise ValueError(
            "Local Whisper is not installed. pip install openai-whisper  (and install ffmpeg)"
        ) from exc

    model_size = _env("WHISPER_MODEL", "base")
    model = whisper.load_model(model_size)
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(audio_path), **kwargs)
    return Transcript(text=(result.get("text") or "").strip(), has_speakers=False)


def transcribe(
    audio_path,
    provider=None,
    diarize=True,
    language=None,
    num_speakers=None,
):
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    provider = (provider or _env("STT_PROVIDER", "openai")).lower()
    language = language if language is not None else _env("LANGUAGE", "en")
    if language == "":
        language = None

    if diarize and provider in NO_DIARIZE_PROVIDERS:
        raise DiarizeNotSupported(
            f"{provider} cannot label speakers. Use --stt openai or --stt elevenlabs, or pass --no-diarize."
        )

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if provider in {"openai", "groq"} and size_mb > 24.5:
        raise ValueError(
            f"{audio_path.name} is {size_mb:.1f} MB; {provider} caps uploads around 25 MB. "
            "Compress to m4a/mp3 or split the file."
        )

    if provider == "openai":
        return _transcribe_openai(audio_path, diarize=diarize, language=language)
    if provider == "groq":
        return _transcribe_groq(audio_path, language=language)
    if provider == "elevenlabs":
        return _transcribe_elevenlabs(
            audio_path, diarize=diarize, language=language, num_speakers=num_speakers
        )
    if provider == "whisper":
        return _transcribe_whisper(audio_path, language=language)
    raise ValueError(f"Unknown STT provider: {provider}")

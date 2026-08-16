#!/usr/bin/env python3
"""Transcribe a meeting recording, then write speaker-aware notes.

Default: OpenAI diarize + Groq notes.

  python main.py recording.m4a
  python main.py --watch              # inbox/ or WATCH_DIR (iCloud/Google Drive folder)
  python main.py --transcript notes.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from summarizer import generate_notes
from transcriber import AUDIO_SUFFIXES, DiarizeNotSupported, transcribe

load_dotenv()

ROOT = Path(__file__).resolve().parent


def _env_bool(name, default=True):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env(name, default=""):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speech → transcript → speaker notes.",
        epilog=(
            "Minimal hands-off: point --watch at an iCloud or Google Drive folder, "
            "save recordings there from your phone, and leave this process running. "
            "Notes land in out/ (and sync back if that folder is in Drive/iCloud)."
        ),
    )
    parser.add_argument("audio", nargs="?", help="Audio file to process")
    parser.add_argument("--transcript", help="Skip STT; write notes from this text file")
    parser.add_argument(
        "--watch",
        nargs="?",
        const=True,
        default=False,
        help="Watch a drop folder (default: WATCH_DIR or ./inbox)",
    )
    parser.add_argument("--stt", default=_env("STT_PROVIDER", "openai"), help="openai|groq|elevenlabs|whisper")
    parser.add_argument("--llm", default=_env("LLM_PROVIDER", "groq"), help="groq|ollama|openai|gemini|anthropic|freellmapi")
    diarize = parser.add_mutually_exclusive_group()
    diarize.add_argument("--diarize", dest="diarize", action="store_true", default=None)
    diarize.add_argument("--no-diarize", dest="diarize", action="store_false")
    parser.add_argument("--language", default=_env("LANGUAGE", "en") or None)
    parser.add_argument("--speakers", type=int, default=None, help="Hint for ElevenLabs diarize")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write transcript + notes (default: OUTPUT_DIR or ~/Documents/MeetingNotes)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave originals in place (required for the Voice Recorder iCloud folder)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing notes")
    parser.add_argument("--poll", type=float, default=5.0, help="Watch poll interval in seconds")
    return parser.parse_args()


def default_notes_dir():
    configured = _env("OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Documents" / "MeetingNotes"


def resolve_output_dir(args, fallback=None):
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    if _env("OUTPUT_DIR"):
        return Path(_env("OUTPUT_DIR")).expanduser()
    if fallback is not None:
        return Path(fallback)
    return default_notes_dir()


def resolve_diarize(flag):
    if flag is not None:
        return flag
    return _env_bool("STT_DIARIZE", True)


def wait_until_stable(path, settle=3.0, timeout=600):
    """Wait until Drive/iCloud finishes copying the file."""
    deadline = time.time() + timeout
    last = (-1, -1)
    stable_since = None
    while time.time() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(0.5)
            continue
        marker = (stat.st_size, stat.st_mtime_ns)
        if marker == last and stat.st_size > 0:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= settle:
                return
        else:
            last = marker
            stable_since = None
        time.sleep(0.5)
    raise TimeoutError(f"{path.name} never finished copying")


INDEX_NAME = ".transcribed.json"


def index_path(output_dir):
    return Path(output_dir) / INDEX_NAME


def load_index(output_dir):
    path = index_path(output_dir)
    if not path.exists():
        return {"files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}
    if not isinstance(data, dict) or "files" not in data:
        return {"files": {}}
    return data


def save_index(output_dir, index):
    path = index_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def fingerprint(audio_path):
    stat = Path(audio_path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def index_key(audio_path):
    return str(Path(audio_path).resolve())


def is_already_transcribed(index, audio_path, output_dir, force=False):
    if force:
        return False
    audio_path = Path(audio_path)
    fp = fingerprint(audio_path)
    rec = index.get("files", {}).get(index_key(audio_path))
    notes_path = output_dir / f"{audio_path.stem}_notes.md"
    transcript_path = output_dir / f"{audio_path.stem}_transcript.txt"
    if rec:
        notes = Path(rec["notes"]) if rec.get("notes") else notes_path
        transcript = Path(rec["transcript"]) if rec.get("transcript") else transcript_path
        return (
            rec.get("size") == fp["size"]
            and rec.get("mtime_ns") == fp["mtime_ns"]
            and notes.exists()
            and transcript.exists()
        )
    return notes_path.exists() and transcript_path.exists()


def mark_transcribed(index, audio_path, transcript_path, notes_path, output_dir):
    audio_path = Path(audio_path)
    fp = fingerprint(audio_path)
    index.setdefault("files", {})[index_key(audio_path)] = {
        "name": audio_path.name,
        "size": fp["size"],
        "mtime_ns": fp["mtime_ns"],
        "transcript": str(Path(transcript_path)),
        "notes": str(Path(notes_path)),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_index(output_dir, index)


def process_transcript_text(text, *, stem, output_dir, llm, force):
    notes_path = output_dir / f"{stem}_notes.md"
    if notes_path.exists() and not force:
        print(f"Skip notes (exists): {notes_path}")
        return notes_path
    print(f"Notes via {llm}...")
    notes = generate_notes(text, provider=llm)
    write_text(notes_path, notes)
    print(f"Wrote {notes_path}")
    return notes_path


def process_audio(audio_path, *, args, output_dir, index=None):
    audio_path = Path(audio_path)
    stem = audio_path.stem
    transcript_path = output_dir / f"{stem}_transcript.txt"
    notes_path = output_dir / f"{stem}_notes.md"
    diarize = resolve_diarize(args.diarize)
    if index is None:
        index = load_index(output_dir)

    if is_already_transcribed(index, audio_path, output_dir, force=args.force):
        mark_transcribed(index, audio_path, transcript_path, notes_path, output_dir)
        return transcript_path, notes_path, False

    if transcript_path.exists() and not args.force:
        print(f"Reusing {transcript_path}")
        text = transcript_path.read_text(encoding="utf-8")
    else:
        print(f"Transcribe {audio_path.name} via {args.stt} (diarize={diarize})...")
        result = transcribe(
            audio_path,
            provider=args.stt,
            diarize=diarize,
            language=args.language,
            num_speakers=args.speakers,
        )
        text = result.text
        write_text(transcript_path, text)
        print(f"Wrote {transcript_path}" + (" (speakers)" if result.has_speakers else ""))

    process_transcript_text(text, stem=stem, output_dir=output_dir, llm=args.llm, force=args.force)
    mark_transcribed(index, audio_path, transcript_path, notes_path, output_dir)
    return transcript_path, notes_path, True


def is_ts_recording(path):
    """Only Voice Recorder files named ts… / TS… (ignore older recordings)."""
    return path.stem.lower().startswith("ts")


def watch_dir_from_args(args):
    if args.watch is True:
        return Path(_env("WATCH_DIR") or ROOT / "inbox")
    return Path(args.watch)


def iter_inbox_audio(folder):
    skip_dirs = {"out", "processed", "failed"}
    candidates = [folder]
    documents = folder / "Documents"
    if documents.is_dir():
        candidates.append(documents)
    for base in candidates:
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for path in sorted(entries):
            if path.is_dir():
                continue
            if path.name.startswith(".") or path.name.startswith("~$"):
                continue
            if path.parent.name in skip_dirs:
                continue
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            if not is_ts_recording(path):
                continue
            yield path


def watch_loop(args):
    inbox = watch_dir_from_args(args)
    keep = args.keep or is_app_icloud_container(inbox)
    output_dir = resolve_output_dir(args)
    failed = output_dir / "failed"
    if keep:
        processed = None
    else:
        inbox.mkdir(parents=True, exist_ok=True)
        processed = inbox / "processed"
        processed.mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)

    print(f"Watching {inbox}")
    print(f"Notes → {output_dir}")
    print("Only files whose name starts with ts / TS are processed.")
    print(f"Already-transcribed index: {index_path(output_dir)}")
    if keep:
        print("Keeping originals in place (Voice Recorder iCloud folder).")
    seen_failed = {}
    index = load_index(output_dir)

    while True:
        for audio_path in iter_inbox_audio(inbox):
            key = (str(audio_path), audio_path.stat().st_mtime_ns if audio_path.exists() else 0)
            if seen_failed.get(audio_path.name) == key:
                continue
            if is_already_transcribed(index, audio_path, output_dir, force=args.force):
                if index_key(audio_path) not in index.get("files", {}):
                    mark_transcribed(
                        index,
                        audio_path,
                        output_dir / f"{audio_path.stem}_transcript.txt",
                        output_dir / f"{audio_path.stem}_notes.md",
                        output_dir,
                    )
                continue
            try:
                wait_until_stable(audio_path)
                _, _, did_work = process_audio(
                    audio_path, args=args, output_dir=output_dir, index=index
                )
                if did_work:
                    print(f"Done: {audio_path.name}")
                if processed is not None:
                    target = processed / audio_path.name
                    if target.exists():
                        target = processed / f"{audio_path.stem}-{int(time.time())}{audio_path.suffix}"
                    shutil.move(str(audio_path), str(target))
                    print(f"Moved original → {target}")
            except DiarizeNotSupported as exc:
                print(exc, file=sys.stderr)
                return 2
            except Exception as exc:
                print(f"Failed {audio_path.name}: {exc}", file=sys.stderr)
                seen_failed[audio_path.name] = key
                try:
                    log = failed / f"{audio_path.stem}.error.txt"
                    log.write_text(str(exc), encoding="utf-8")
                except OSError:
                    pass
        time.sleep(args.poll)


def main():
    args = parse_args()
    if args.watch:
        return watch_loop(args) or 0

    if args.transcript:
        text_path = Path(args.transcript)
        text = text_path.read_text(encoding="utf-8")
        output_dir = resolve_output_dir(args, fallback=text_path.parent)
        process_transcript_text(
            text, stem=text_path.stem.replace("_transcript", ""), output_dir=output_dir, llm=args.llm, force=args.force
        )
        return 0

    if not args.audio:
        print("Pass an audio file, --transcript, or --watch.", file=sys.stderr)
        return 2

    audio_path = Path(args.audio)
    output_dir = resolve_output_dir(args, fallback=audio_path.parent)
    try:
        process_audio(audio_path, args=args, output_dir=output_dir)
    except DiarizeNotSupported as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0)

# iSummarize
# Meeting transcriber + speaker notes
Batch pipeline for voice-recorder files: **speech → transcript → speaker-aware notes**. Not realtime.
Default path:
- **Speech to text:** OpenAI `gpt-4o-transcribe-diarize` (speaker labels). ChatGPT Plus does **not** include this; you need API billing at [platform.openai.com](https://platform.openai.com).
- **Notes:** Groq Llama (free tier).
## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Fill in `.env` (never commit `.env`):
```bash
OPENAI_API_KEY=...          # platform.openai.com
GROQ_API_KEY=...            # console.groq.com
STT_PROVIDER=openai
STT_DIARIZE=true
LLM_PROVIDER=groq
WATCH_DIR=/Users/YOU/Library/Mobile Documents/iCloud~com~TapMediaLtd~VoiceRecorderFREE
OUTPUT_DIR=/Users/YOU/Documents/MeetingNotes
```
`WATCH_DIR` is the Voice Recorder app’s iCloud folder. Recordings stay there; this tool does not move them.
`OUTPUT_DIR` is where `{name}_transcript.txt` and `{name}_notes.md` go. Use a folder under iCloud Drive if you want the notes on your iPhone (Files app).
## One file
Export **m4a** from Voice Recorder. Names must start with `ts` or `TS` for watch mode.
```bash
python main.py /path/to/ts-meeting.m4a
python main.py /path/to/ts-meeting.m4a --output-dir ~/Documents/MeetingNotes
python main.py --no-diarize --stt groq ts-meeting.m4a   # free STT, no speakers
python main.py --transcript existing.txt                # notes only
```
## Watch folder (hands-off)
```bash
python main.py --watch
```
Leave that process running. It will:
- Watch `WATCH_DIR` (and `Documents/` inside it)
- Process only files named `ts…` / `TS…`
- Skip files already in `OUTPUT_DIR/.transcribed.json` (same path, size, mtime)
- Re-run if you save over a file and it changes
- Write notes to `OUTPUT_DIR`, not into this repo and not into Voice Recorder
In the phone app, turn on iCloud. Save new meetings as **m4a** with a `ts` prefix.
## Formats
Use **m4a**. Avoid wav/aiff/caf (too large for the ~25 MB API cap). mp3 is a fine backup.
## Other providers
| Flag | Values |
|---|---|
| `--stt` | `openai` (default), `groq`, `elevenlabs`, `whisper` (local, uses disk) |
| `--llm` | `groq` (default), `ollama`, `openai`, `gemini`, `anthropic`, `freellmapi` |
| `--diarize` / `--no-diarize` | Speaker labels (OpenAI / ElevenLabs only) |
| `--keep` | Leave originals in place (automatic for Voice Recorder iCloud) |
| `--force` | Redo even if already transcribed |
`groq` and local `whisper` cannot label speakers. Use `--no-diarize` with those.
## Layout
```
main.py           CLI and watcher
transcriber.py    OpenAI / Groq / ElevenLabs / optional Whisper
summarizer.py     Groq / Ollama / OpenAI / others
.env.example      Copy to .env
plan.md           Design notes
```

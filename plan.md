# Meeting transcriber + speaker notes

Simple batch pipeline for voice-recorder files: **speech → transcript → speaker-aware notes**. Not realtime.

## Best bet right now

Use **OpenAI for speakers, Groq for notes.** Nothing heavy on the M1 8GB.

| Step | Service | Why |
|---|---|---|
| Transcribe + diarize | OpenAI `gpt-4o-transcribe-diarize` | You already have the key. Real speaker labels. No SSD. ~tens of cents per hour of audio |
| Notes | Groq `llama-3.3-70b-versatile` | Free hosted tier, better notes than a 3B Ollama model on 8GB |
| Command | `python main.py recording.m4a --diarize --llm groq` | One shot, then two text files |

Do **not** start with local Whisper, Colab CLI, FreeLLMAPI, or ElevenLabs. Those are fallbacks.

| If this happens | Then |
|---|---|
| You will not spend anything on STT | `--stt groq --no-diarize --llm groq` — free, no speakers |
| Groq is down / you want notes offline | `--llm ollama` with `llama3.2:3b` |
| You want WhisperX speakers with no OpenAI bill | Colab CLI + T4, later — not v1 |
| You want one proxy for many free LLMs | FreeLLMAPI, later |

v1 implements OpenAI (diarize + plain), Groq (STT + notes), Ollama notes. Everything else stays optional in config, not the happy path.

## Goal

Given an audio file (`.mp3`, `.wav`, `.m4a`, `.mp4`):

1. Transcribe it.
2. Write meeting notes organized by speaker: overview, what each person said, decisions, action items.
3. Save both files next to the audio (or in an output folder).

One command:

```bash
python main.py recording.m4a
```

Hands-off drop folder (recommended instead of an app):

```bash
python main.py --watch
# or: python main.py --watch "/path/to/iCloud or Google Drive/MeetingInbox"
```

## Ingestion (minimal tapping)

Do **not** build a Mac/iPhone dashboard in v1. Opening an app is more work than dropping a file. ChatGPT Plus also does not run this pipeline.

The low-interference path is a **watched folder** that already syncs:

1. Create `MeetingInbox` in iCloud Drive or Google Drive (Desktop app / Drive for Desktop).
2. Run `python main.py --watch "/path/to/MeetingInbox"` on the Mac (leave it running, or add a Login Item later).
3. After a meeting, save/share the recording into that folder (Files app, Voice Memos share, USB copy from the recorder, AirDrop).
4. The watcher waits until the upload finishes, transcribes, writes notes to `MeetingInbox/out/`, moves the audio to `MeetingInbox/processed/`.
5. Notes appear on the iPhone in the same Drive/iCloud folder. No extra app.

A dedicated recorder still needs **one** copy/sync step unless you record on the phone into that folder. That is the remaining human step; a dashboard would not remove it.

## Out of scope (for now)

- Live / streaming captions
- Zoom/Meet bots
- A GUI / iPhone app
- Local speaker diarization (WhisperX / pyannote) — too heavy for an M1 8GB
- Downloading local Whisper models unless you explicitly opt in with `--stt whisper`
- Running Whisper and Ollama at the same time

## How it works

```
audio file
    │
    ▼
transcriber (openai | groq | elevenlabs | whisper)
    │
    ▼
transcript with optional [SPEAKER_0] labels
    │
    ▼
notes generator (ollama | freellmapi | groq | gemini | openai | anthropic)
    │
    ▼
recording_transcript.txt
recording_notes.md
```

Audio is uploaded for transcription; only the small text files are written locally. Notes still run after transcription finishes.

## Cost: is the API free?

**OpenAI transcription is not free.** New accounts sometimes get a small one-time credit; after that you pay per minute. Roughly:

- `gpt-4o-transcribe-diarize` — paid, speaker labels (default for meetings)
- `gpt-transcribe` — paid, no speakers, cheaper than old Whisper
- A 1-hour meeting is on the order of **tens of cents**, not dollars

There is **no unlimited free hosted API that also labels speakers**. Diarization is the paid part.

| Task | Actually free (ongoing) | Speakers | SSD? |
|---|---|---|---|
| Transcribe + diarize | None that is unlimited | Yes | No (OpenAI / ElevenLabs / Deepgram credits) |
| Transcribe, no speakers | **Groq Whisper** (rate-limited free tier, no card) | No | No |
| Transcribe, no speakers | Local Whisper | No | Yes |
| Notes / summary | **Ollama** on this Mac | n/a | Small 3B model only |
| Notes / summary | **FreeLLMAPI** → Groq/Gemini/etc. free tiers | n/a | No (proxy is local; models are hosted) |
| Notes / summary | **Groq** or **Gemini** chat APIs directly | n/a | No |

Signup credits (not forever-free): Deepgram (~$200), AssemblyAI (~$50). Those *do* diarize, then you pay.

**Practical split for this project**

- Meetings where you care who spoke: `--diarize` → OpenAI (paid, tiny)
- Just need the words, $0, no SSD: `--stt groq --no-diarize`
- Notes always $0 unless you opt into a paid LLM: Ollama, **FreeLLMAPI**, or Groq/Gemini keys

## Free hosted inference (not your Mac)

There is no Heroku-for-Whisper that is unlimited, always-on, **and** does speaker labels. What exists:

| Host | What it runs | Speakers | Good as an API? |
|---|---|---|---|
| **Groq** | Whisper large-v3 + Llama/etc. | No | Yes — best free host. One key, OpenAI-compatible. ~hours of audio/day on the free tier |
| **Google AI Studio (Gemini)** | Notes; can also ingest short audio | Weak / no real diarize | Yes for notes |
| **Cloudflare Workers AI** | Whisper + small LLMs | No | Yes if you already use Cloudflare; daily neuron cap |
| **Hugging Face Spaces / ZeroGPU** | Demos of Whisper/WhisperX | Maybe, in a demo UI | Poor as a backend: queues, sleeps, no SLA |
| **Google Colab** | Anything, including WhisperX + speakers | Yes | Not a service — a notebook you click each time |
| Hugging Face Inference Endpoints | Whisper + pyannote | Yes | **Paid** GPU, not free |
| FreeLLMAPI | Router only | n/a | Talks *to* Groq/Gemini; does not host models |

So: **Groq is the free model host** for this project. Same idea as “don’t use the SSD”: their machines run Whisper and the notes model. You still create a free key.

Diarize stays paid (OpenAI / ElevenLabs) or a Colab one-off. Do not plan on HF Spaces as the production transcriber.

## Speech to text

Both modes are supported. `--diarize` (default) vs `--no-diarize`.

Do **not** default to local Whisper. On an M1 8GB that means a 100MB–3GB model on disk, RAM pressure, and SSD swap.

| Mode | Flag | Provider / model | Speakers | Cost |
|---|---|---|---|---|
| Diarize (default) | `--diarize` | OpenAI `gpt-4o-transcribe-diarize` | Yes | Paid |
| Diarize | `--stt elevenlabs --diarize` | ElevenLabs `scribe_v2` | Yes | Paid |
| Plain transcribe | `--no-diarize` | OpenAI `gpt-transcribe` | No | Paid, cheaper |
| Plain transcribe, free | `--stt groq --no-diarize` | Groq `whisper-large-v3` | No | Free tier |
| Offline fallback | `--stt whisper` | Local Whisper `base` | No | Free, uses SSD |

If you pass `--diarize` with Groq or local Whisper, the CLI should error: those backends cannot label speakers. Use OpenAI or ElevenLabs, or drop `--diarize`.

Local `whisper` is opt-in. `openai-whisper` is not a required install.

| Provider | When to use | Speakers |
|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` already in `.env` | Yes if `--diarize`, no if `--no-diarize` |
| `groq` | Free hosted Whisper, no SSD | No |
| `elevenlabs` | `ELEVENLABS_API_KEY` | Yes |
| `whisper` | Offline / privacy | No |

Speaker-labeled APIs format the transcript as:

```
[SPEAKER_0] Let's start with the budget.
[SPEAKER_1] I can send the spreadsheet today.
```

## Notes / summary

The notes model always gets the full transcript and is asked for:

- Short overview
- Notes **by speaker** (use labels if present; otherwise infer names from the conversation)
- Decisions
- Action items with owner if known

| Provider | When to use | Default model | Cost |
|---|---|---|---|
| `groq` (default) | Free hosted LLM, better than 3B local | `llama-3.3-70b-versatile` | Free tier |
| `ollama` | Notes fully offline | `llama3.2:3b` | Free |
| `freellmapi` | Stack free Groq/Gemini/etc. behind one local `/v1` | `auto` | Free tiers (keys go in FreeLLMAPI) |
| `gemini` | Talk to Gemini directly | Gemini Flash | Free tier |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | Paid |
| `openai` + `OPENAI_BASE_URL` | Any other OpenAI-compat gateway | set `OPENAI_MODEL` | Varies |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-0` | Paid |

### FreeLLMAPI ([tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi))

**Yes, use it for notes.** It is a local OpenAI-compatible proxy (`http://localhost:3001/v1`) that routes chat across free tiers (Groq, Gemini, Mistral, OpenRouter, …) with failover when one hits a rate limit.

It is **not** a speech-to-text service. Its `/v1/audio/speech` path is text-to-speech. Diarized transcription still goes to OpenAI / ElevenLabs / Groq Whisper as above.

How we wire it (no extra SDK):

```python
OpenAI(base_url="http://localhost:3001/v1", api_key=FREELLMAPI_KEY)
# model="auto"
```

**Free tier still needs keys.** Groq, Gemini, etc. do not let you call them with no credential. You sign up (usually no credit card), they give you a **free API key**, and that key is rate-limited rather than billed. FreeLLMAPI does not invent access; it stores those keys and rotates across them.

Two different keys:

1. **Provider keys** (Groq `gsk_…`, Gemini `AIza…`) — free to create, pasted into the FreeLLMAPI dashboard. Without at least one of these, the proxy has nothing to call.
2. **Unified `freellmapi-…` token** — only authenticates our script to your local proxy. It is not a vendor key and costs nothing.

The only notes path with **no API key at all** is Ollama on this Mac.

You run FreeLLMAPI separately (desktop app or Docker), paste Groq/Gemini keys into its dashboard, copy the unified `freellmapi-…` key into our `.env`. Personal/experimentation use; no SLA; quality dips when free quotas reset.

That replaces a pile of `LLM_PROVIDER=groq|gemini|…` keys in *this* repo with one gateway. Direct Groq/Gemini providers stay as simpler options if you do not want another process.

Ollama must already be running (`ollama serve`) with the model pulled:

```bash
ollama pull llama3.2:3b
```

Long meetings: if the transcript is large, chunk it (summarize sections, then combine). Needed for the 3B local model; less important for paid APIs.

## Config

`.env` (never commit this):

```bash
# speech to text: openai | groq | elevenlabs | whisper
STT_PROVIDER=openai
STT_DIARIZE=true
LANGUAGE=en

# used automatically: diarize → gpt-4o-transcribe-diarize, else gpt-transcribe
OPENAI_STT_MODEL=

# groq: free Whisper (no speakers) + free notes
GROQ_API_KEY=
GROQ_STT_MODEL=whisper-large-v3
GROQ_MODEL=llama-3.3-70b-versatile

# local whisper fallback only
WHISPER_MODEL=base

# notes: ollama | freellmapi | groq | gemini | openai | anthropic
LLM_PROVIDER=groq
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# FreeLLMAPI local proxy (run separately: http://localhost:3001)
FREELLMAPI_KEY=
FREELLMAPI_BASE_URL=http://localhost:3001/v1
FREELLMAPI_MODEL=auto

OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-0

ELEVENLABS_API_KEY=
```

CLI flags override env:

```bash
python main.py AUDIO --diarize
python main.py AUDIO --no-diarize
python main.py AUDIO --stt groq --no-diarize --llm groq
python main.py AUDIO --llm freellmapi
```

Notes-only mode if a transcript already exists:

```bash
python main.py --transcript recording_transcript.txt
```

## File layout

```
summarizer/
  main.py           # CLI: transcribe then notes
  transcriber.py    # OpenAI (diarize + plain) / Groq / ElevenLabs / optional local Whisper
  summarizer.py     # Ollama / FreeLLMAPI / Groq / Gemini / OpenAI / Anthropic
  requirements.txt
  .env.example
  .env              # local keys, gitignored
  plan.md           # this file
```

Keep the existing two modules. Fix the broken class indentation in `transcriber.py`. Remove the hardcoded demo run at the bottom of `summarizer.py`.

## CLI shape

```bash
python main.py AUDIO                              # openai diarize + groq notes
python main.py AUDIO --no-diarize                 # openai gpt-transcribe, no speakers
python main.py AUDIO --stt groq --no-diarize      # free hosted Whisper, no SSD
python main.py AUDIO --stt groq --no-diarize --llm groq
python main.py AUDIO --llm freellmapi              # notes via local FreeLLMAPI proxy
python main.py AUDIO --stt elevenlabs --diarize --llm openai
python main.py AUDIO --stt whisper --llm ollama   # local fallback; uses SSD
python main.py --transcript notes_input.txt --llm ollama
python main.py AUDIO --output-dir ./out
python main.py --watch
python main.py --watch ~/Library/CloudStorage/…/MeetingInbox
```

Prints progress, writes:

- `{name}_transcript.txt`
- `{name}_notes.md`

## Implementation order

1. `.env.example`, `.gitignore`, `requirements.txt`
2. Rewrite `transcriber.py`: OpenAI diarize **and** `gpt-transcribe`, Groq Whisper, ElevenLabs Scribe, optional local Whisper; shared transcript formatter
3. Rewrite `summarizer.py`: Ollama / FreeLLMAPI / Groq / Gemini / OpenAI / Anthropic + speaker-notes prompt
4. Add `main.py` to wire the two steps and write files
5. Smoke-check: notes from a sample transcript via Ollama; a short OpenAI STT call only if an audio file is handy

## Dependencies

- `python-dotenv`
- `requests` (Ollama + ElevenLabs)
- `openai` (OpenAI STT + chat; Groq/Gemini/OpenRouter via base URL)
- `anthropic` (optional extra; only required if that provider is used)
- `openai-whisper` + `ffmpeg` **only if** you opt into `--stt whisper`

## Hardware notes (M1 8GB)

- Default path does **not** load a speech model on the Mac: upload audio → OpenAI diarize → Groq for notes
- `$0` hosted notes: `--llm freellmapi` (or Groq/Gemini directly). `$0` STT without speakers: Groq Whisper (`--no-diarize`)
- That avoids Whisper weights on the SSD and avoids swap while transcribing
- You still write two small text files locally; that is fine
- Audio leaves the machine for any hosted STT. Use `--stt whisper` only when you need offline/privacy
- Do not default to local Whisper `small`/`medium` or 7B+ chat models

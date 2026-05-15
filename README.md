# Conference Video Transcriber

Local transcription of conference videos with automatic speaker diarization. Runs entirely on your Mac — no data leaves your machine.

## What it does

- Extracts audio from video files (any format ffmpeg supports)
- Identifies and separates different speakers (pyannote.audio)
- Transcribes speech to text (faster-whisper, Whisper large-v3)
- Outputs SRT subtitle files with speaker labels
- Provides a simple browser UI (Gradio) for drag-and-drop usage

## Requirements (one-time)

### 1. Get a HuggingFace token

The speaker diarization model requires a free HuggingFace account:

1. Create an account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens and create a **read** token
3. Accept the license for these two models (click "Agree" on each page):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## How to start

Double-click **`Start Transcriber.command`** in Finder.

On the first run it will:
- Check that Python 3 and ffmpeg are installed (installs ffmpeg via Homebrew if missing)
- Create a Python virtual environment and install all dependencies

After that the browser opens automatically at **http://127.0.0.1:7860**.

## Usage

1. Upload a video (drag and drop)
2. Paste your HuggingFace token
3. Choose model size and language
4. Optionally set the expected number of speakers
5. Click **Transcribe**

The SRT file is saved next to your original video file.

## Settings guide

| Setting | Recommendation |
|---------|---------------|
| **Model** | `large-v3` for best quality, `medium` for faster processing |
| **Language** | `auto` works well, or pick `en`/`el`/etc. if you know |
| **Speakers** | `0` for auto-detect, or specify if you know the count |

## Output format

```
1
00:00:05,120 --> 00:00:08,340
[Speaker 00] Welcome everyone to the conference.

2
00:00:09,100 --> 00:00:12,560
[Speaker 01] Thank you. Let me start with the overview.
```

## Troubleshooting

**"No module named torch"** — Delete the `venv` folder and double-click `Start Transcriber.command` again to rebuild it.

**Diarization fails with 401** — Your HuggingFace token is invalid or you haven't accepted the model licenses (see setup step above).

**Slow transcription** — Use `medium` or `small` model. `large-v3` needs ~3 GB RAM and is slower.

**ffmpeg not found** — `brew install ffmpeg`

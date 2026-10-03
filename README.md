# Automated Video Dubbing System

This project implements the Idealabs Digital assignment: give it a YouTube URL (or a local video), and it creates an English-dubbed MP4 while copying the original video stream without re-encoding.

## Architecture

`yt-dlp` downloads the source. FFmpeg extracts mono 16 kHz audio. `faster-whisper` transcribes speech with timestamps. Each segment is translated for natural spoken English using Argos Translate when installed, or an OpenAI-compatible chat model when `OPENAI_API_KEY` is present. `edge-tts` synthesizes English speech. FFmpeg time-fits the clips, concatenates them, then muxes the new audio with the untouched video frames (`-c:v copy`). A JSON manifest preserves every source/translation timestamp for review.

## Setup

Install FFmpeg and add it to PATH, then:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

For non-English input choose one translation backend: set `OPENAI_API_KEY`, or install Argos Translate and the source-language -> English package. The first faster-whisper run downloads its model. `small` is a good quality/speed default; use `--model medium` for accuracy or `--model base` for faster iteration.

## Run

```bash
python -m app.main "https://www.youtube.com/watch?v=VIDEO_ID" -o outputs/dubbed.mp4
```

Useful options: `--language hi` avoids language auto-detection, `--voice en-US-GuyNeural` changes the English voice, `--keep-intermediate` keeps source audio/TTS clips, and `--work-dir` chooses a job directory. The command prints progress and writes `outputs/dubbed.json` beside the output.

## Evaluation / long videos

Run the same command once for a roughly 30-minute video and once for a roughly 2-hour video. Record wall-clock time and hardware in the submission notes. The pipeline processes Whisper segments incrementally, so memory usage is flatter than whole-file processing. Long runs still need disk space for the source, extracted audio, and temporary TTS clips.

## Quality notes

- Visuals are copied with `-c:v copy`; only the audio stream is replaced.
- Segment boundaries preserve original timing. TTS is trimmed/padded to each source segment window.
- The default voice is natural English neural TTS, not voice cloning. Diarization and cloning are stretch goals.
- Review the generated manifest and watch the output before submission; model and source quality affect translation and TTS.

## Tests

```bash
pytest -q
```

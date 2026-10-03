"""Automated YouTube video dubbing CLI."""
from __future__ import annotations
import argparse, asyncio, json, os, re, shutil, subprocess, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class Segment:
    start: float
    end: float
    text: str
    translation: str = ""
    @property
    def duration(self) -> float:
        return max(0.05, self.end - self.start)

def log(message: str) -> None:
    print(f"[dub] {message}", flush=True)

def run(command: list[str], *, capture: bool = False):
    if command and command[0] == "ffmpeg" and shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            command = [imageio_ffmpeg.get_ffmpeg_exe(), *command[1:]]
        except ImportError:
            pass
    try: return subprocess.run(command, check=True, text=True, capture_output=capture)
    except FileNotFoundError as exc: raise RuntimeError(f"Required executable '{command[0]}' was not found. Install FFmpeg and ensure it is on PATH.") from exc
    except subprocess.CalledProcessError as exc: raise RuntimeError((exc.stderr or exc.stdout or "").strip()) from exc

def download_video(source: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True); local = Path(source).expanduser()
    if local.exists(): shutil.copy2(local, destination); return destination
    if not source.startswith(("http://", "https://")): raise ValueError("source must be a YouTube URL or an existing local video path")
    try: import yt_dlp
    except ImportError as exc: raise RuntimeError("Install yt-dlp first: pip install yt-dlp") from exc
    log("downloading source video")
    options={"format":"134+140/136+140/best","outtmpl":str(destination.with_suffix(".%(ext)s")),"merge_output_format":"mp4","noplaylist":True,"quiet":True,"no_warnings":True}
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            options["ffmpeg_location"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    with yt_dlp.YoutubeDL(options) as ydl: ydl.download([source])
    candidates=sorted(destination.parent.glob(destination.stem+".*"))
    if not candidates: raise RuntimeError("yt-dlp completed but no downloaded video was found")
    if candidates[0] != destination: candidates[0].replace(destination)
    return destination

def extract_audio(video: Path, wav: Path) -> None:
    log("extracting audio"); run(["ffmpeg","-y","-i",str(video),"-vn","-ac","1","-ar","16000",str(wav)])

def transcribe(audio: Path, model_name: str, language: str):
    try: from faster_whisper import WhisperModel
    except ImportError as exc: raise RuntimeError("Install faster-whisper first: pip install faster-whisper") from exc
    log(f"transcribing with Whisper model '{model_name}'"); model=WhisperModel(model_name,device="auto",compute_type="int8")
    raw,info=model.transcribe(str(audio),vad_filter=True,**({} if language=="auto" else {"language":language}))
    segments=[Segment(float(s.start),float(s.end),s.text.strip()) for s in raw if s.text.strip()]
    if not segments: raise RuntimeError("transcription returned no speech")
    return getattr(info,"language",language),segments

def translate_text(text: str, source_language: str) -> str:
    if source_language.lower().startswith("en"): return text
    try:
        import argostranslate.translate as argos; translated=argos.translate(text,source_language,"en")
        if translated and translated != text: return translated
    except (ImportError, OSError): pass
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_language, target="en").translate(text)
        if translated: return translated
    except Exception: pass
    try:
        import requests
        response=requests.get("https://api.mymemory.translated.net/get",params={"q":text,"langpair":f"{source_language}|en"},timeout=30)
        translated=response.json().get("responseData",{}).get("translatedText","")
        if translated: return translated
    except Exception: pass
    api_key=os.getenv("OPENAI_API_KEY")
    if api_key:
        import requests
        response=requests.post(os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")+"/chat/completions",headers={"Authorization":f"Bearer {api_key}"},json={"model":os.getenv("OPENAI_TRANSLATION_MODEL","gpt-4o-mini"),"temperature":0.1,"messages":[{"role":"system","content":"Translate the user text to natural spoken English. Return only the translation."},{"role":"user","content":text}]},timeout=60); response.raise_for_status(); return response.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"No translator configured for detected language '{source_language}'. Install argostranslate or set OPENAI_API_KEY.")

def translate_segments(segments: list[Segment], source_language: str) -> None:
    log(f"translating {len(segments)} speech segments to English")
    for number,segment in enumerate(segments,1): segment.translation=translate_text(segment.text,source_language); print(f"[dub] translated {number}/{len(segments)}",flush=True)

async def _synthesize(text: str, output: Path, voice: str) -> None:
    import edge_tts; await edge_tts.Communicate(text,voice).save(str(output))

def synthesize_segment(text: str, output: Path, voice: str) -> None:
    try: import edge_tts
    except ImportError as exc: raise RuntimeError("Install edge-tts first: pip install edge-tts") from exc
    asyncio.run(_synthesize(text,output,voice))

def fit_audio(input_audio: Path, output_audio: Path, duration: float) -> None:
    run(["ffmpeg","-y","-i",str(input_audio),"-filter:a",f"apad,atrim=duration={duration:.3f},asetpts=N/SR/TB",str(output_audio)])

def build_dub_audio(segments: list[Segment], work_dir: Path, voice: str) -> Path:
    log("synthesizing English speech"); fitted=[]
    for index,segment in enumerate(segments,1):
        raw,fitted_audio=work_dir/f"tts_{index:05d}.mp3",work_dir/f"fit_{index:05d}.wav"; synthesize_segment(segment.translation,raw,voice); fit_audio(raw,fitted_audio,segment.duration); fitted.append(fitted_audio); print(f"[dub] synthesized {index}/{len(segments)}",flush=True)
    concat=work_dir/"audio_concat.txt"; concat.write_text("\n".join(f"file '{p.as_posix()}'" for p in fitted),encoding="utf-8"); dubbed=work_dir/"dubbed_audio.wav"; run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(concat),"-ac","2",str(dubbed)]); return dubbed

def build_timed_audio_from_existing(segments: list[Segment], fitted: list[Path], output: Path) -> None:
    inputs=[]
    filters=[]
    labels=[]
    for index,(segment,audio) in enumerate(zip(segments,fitted)):
        inputs += ["-i", str(audio)]
        delay=max(0,int(segment.start*1000))
        label=f"a{index}"
        filters.append(f"[{index}:a]adelay={delay}|{delay}[{label}]")
        labels.append(f"[{label}]")
    filters.append("".join(labels)+f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0,apad,atrim=duration={segments[-1].end:.3f},asetpts=N/SR/TB[aout]")
    run(["ffmpeg","-y",*inputs,"-filter_complex",";".join(filters),"-map","[aout]","-ac","2",str(output)])
def mux_video(video: Path, audio: Path, output: Path) -> None:
    log("muxing dubbed audio with original video frames"); output.parent.mkdir(parents=True,exist_ok=True); run(["ffmpeg","-y","-i",str(video),"-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output)])

def write_manifest(path: Path, source: str, detected_language: str, segments: Iterable[Segment], output: Path) -> None:
    path.write_text(json.dumps({"source":source,"detected_language":detected_language,"target_language":"en","output":str(output),"segments":[asdict(s) for s in segments]},indent=2,ensure_ascii=False),encoding="utf-8")

def parse_vtt(path: Path) -> list[Segment]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig"))
    segments = []
    for block in blocks:
        match = re.search(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[.,]\d{3})", block)
        if not match:
            continue
        def seconds(value: str) -> float:
            h, m, rest = value.replace(",", ".").split(":")
            return int(h) * 3600 + int(m) * 60 + float(rest)
        text = " ".join(line.strip() for line in block.splitlines()[1:] if line.strip() and "-->" not in line)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            segments.append(Segment(seconds(match.group(1)), seconds(match.group(2)), text))
    return segments

def coalesce_segments(segments: list[Segment], max_chars: int = 450, max_duration: float = 20.0) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and len(merged[-1].text) + len(segment.text) + 1 <= max_chars and segment.end - merged[-1].start <= max_duration:
            merged[-1].end = segment.end
            merged[-1].text += " " + segment.text
        else:
            merged.append(Segment(segment.start, segment.end, segment.text))
    return merged
def download_auto_captions(source: str, work_dir: Path, language: str) -> Path | None:
    try:
        import yt_dlp
        target = work_dir / "captions.%(ext)s"
        options = {"skip_download": True, "writeautomaticsub": True, "subtitleslangs": [language], "subtitlesformat": "vtt", "outtmpl": str(target), "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(options) as ydl: ydl.download([source])
        found = list(work_dir.glob("captions.*.vtt"))
        return found[0] if found else None
    except Exception:
        return None
def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Download a video and create an English dubbed copy.")
    parser.add_argument("url",help="YouTube URL or local video path")
    parser.add_argument("-o","--output",default="outputs/dubbed_video.mp4")
    parser.add_argument("--work-dir",default="outputs/dubbing_job")
    parser.add_argument("--model",default="small")
    parser.add_argument("--language",default="auto")
    parser.add_argument("--voice",default="en-US-AriaNeural")
    parser.add_argument("--keep-intermediate",action="store_true")
    args=parser.parse_args(argv)
    work_dir=Path(args.work_dir).resolve(); output=Path(args.output).resolve(); work_dir.mkdir(parents=True,exist_ok=True)
    try:
        source_video=download_video(args.url,work_dir/"source.mp4")
        audio=work_dir/"source.wav"
        extract_audio(source_video,audio)
        try:
            detected_language,segments=transcribe(audio,args.model,args.language)
        except (RuntimeError, OSError, ImportError) as transcription_error:
            log(f"Whisper unavailable ({transcription_error}); trying YouTube captions")
            caption=download_auto_captions(args.url,work_dir,args.language if args.language != "auto" else "hi")
            if not caption: raise
            detected_language=args.language if args.language != "auto" else "hi"
            segments=coalesce_segments(parse_vtt(caption), max_chars=1000, max_duration=60.0)
            if not segments: raise RuntimeError("caption fallback returned no speech segments")
        translate_segments(segments,detected_language)
        dubbed_audio=build_dub_audio(segments,work_dir,args.voice)
        mux_video(source_video,dubbed_audio,output)
        manifest=output.with_suffix(".json")
        write_manifest(manifest,args.url,detected_language,segments,output)
        log(f"done: {output}")
        log(f"manifest: {manifest}")
        if not args.keep_intermediate:
            for item in work_dir.iterdir():
                if item.name != "source.mp4": item.unlink(missing_ok=True)
        return 0
    except Exception as exc:
        print(f"[dub] ERROR: {exc}",file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())









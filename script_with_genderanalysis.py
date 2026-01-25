import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # important on mounts like /mnt

import gc
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"No module named 'torchcodec'",
    category=UserWarning,
)
torchaudio.set_audio_backend("ffmpeg")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("podcast")


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def pick_first_episode(root_dir: str = "fyyd_downloads") -> Tuple[Path, Path]:
    """
    Picks ONE episode:
      - first podcast folder in root_dir (alphabetical)
      - first audio file inside that folder (alphabetical, recursive)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root.resolve()}")

    podcast_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not podcast_dirs:
        raise RuntimeError(f"No podcast folders found inside: {root.resolve()}")

    first_podcast = podcast_dirs[0]
    audio_files = sorted([p for p in first_podcast.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    if not audio_files:
        raise RuntimeError(f"No audio files found inside: {first_podcast.resolve()}")

    return first_podcast, audio_files[0]


def load_audio_mono_16k(audio_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio with torchaudio and resample to mono 16k.
    Returns waveform: [1, n], sr=16000
    """
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav, sr

def extract_diarized_segments(diarization_output) -> List[Dict]:
    """
    pyannote-audio 4.x returns DiarizeOutput with fields like:
      - speaker_diarization (Annotation)
      - exclusive_speaker_diarization (Annotation)
    """
    # If it's already an Annotation
    if hasattr(diarization_output, "itertracks"):
        ann = diarization_output

    # DiarizeOutput fields in your version
    elif hasattr(diarization_output, "exclusive_speaker_diarization"):
        ann = diarization_output.exclusive_speaker_diarization

    elif hasattr(diarization_output, "speaker_diarization"):
        ann = diarization_output.speaker_diarization

    else:
        raise TypeError(
            f"Unexpected diarization output type={type(diarization_output)}; "
            "expected Annotation or DiarizeOutput with speaker_diarization fields."
        )

    diarized_segments = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        diarized_segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })
    return diarized_segments



def match_segments(transcribed: List[Dict], diarized: List[Dict]) -> List[Dict]:
    """
    Assign each Whisper segment to the speaker with maximum overlap.
    """
    result = []
    for t in transcribed:
        t_start = float(t["start"])
        t_end = float(t["end"])

        best_speaker = "Unknown"
        best_overlap = 0.0

        for d in diarized:
            d_start = float(d["start"])
            d_end = float(d["end"])

            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        result.append({
            "start": t_start,
            "end": t_end,
            "speaker": best_speaker,
            "text": t.get("text", "").strip(),
        })
    return result


class PodcastPipeline:
    def __init__(
        self,
        whisper_model_size: str,
        hf_token: str,
        use_gpu: bool = True,
        diarization_on_gpu: bool = True,
    ):
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Whisper device: {self.device}")

        # Whisper
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        # Pyannote pipeline (use 3.1 to avoid the revision error)
        # NOTE: depending on your pyannote version, the token argument name may vary.
        self.pipeline = self._load_pyannote("pyannote/speaker-diarization-3.1", hf_token)

        # Optionally move to GPU
        if diarization_on_gpu and torch.cuda.is_available():
            try:
                logger.info("Moving diarization pipeline to CUDA")
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            except Exception as e:
                logger.warning(f"Could not move diarization to CUDA, staying on CPU. Reason: {e}")

    @staticmethod
    def _load_pyannote(model_id: str, token: str) -> Pipeline:
        # Compatible across pyannote versions
        for kwargs in ({"token": token}, {"use_auth_token": token}, {"hf_token": token}):
            try:
                return Pipeline.from_pretrained(model_id, **kwargs)
            except TypeError:
                continue
        return Pipeline.from_pretrained(model_id)

    def transcribe(self, audio_path: str) -> Tuple[List[Dict], str]:
        logger.info(f"Transcribing: {audio_path}")
        # fp16 True on CUDA improves speed and reduces VRAM
        result = self.whisper_model.transcribe(audio_path, verbose=False, fp16=False)
        return result["segments"], result.get("language", "unknown")

    def diarize(self, audio_path: str) -> List[Dict]:
        logger.info(f"Diarizing: {audio_path}")

        # Preload audio -> avoids backend decoding issues
        wav, sr = load_audio_mono_16k(audio_path)

        with torch.no_grad():
            diarization_output = self.pipeline({"waveform": wav, "sample_rate": sr})
            print("DiarizeOutput type:", type(diarization_output))
            try:
                print("fields:", vars(diarization_output).keys())
            except Exception as e:
                print("no vars:", e)

        diarized_segments = extract_diarized_segments(diarization_output)

        # housekeeping
        torch.cuda.empty_cache()
        gc.collect()
        return diarized_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default="", help="Optional path to a single audio file. If empty, auto-picks from fyyd_downloads.")
    ap.add_argument("--downloads", type=str, default="fyyd_downloads", help="Root folder of downloaded podcasts.")
    ap.add_argument("--whisper_model", type=str, default="small", help="Whisper model size (tiny/base/small/medium/large).")
    ap.add_argument("--diar_gpu", action="store_true", help="Try to run diarization on GPU too.")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path.")
    args = ap.parse_args()

    hf_token = os.getenv("PYANNOTE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Missing Hugging Face token. Export PYANNOTE_TOKEN (recommended) before running.")

    if args.audio:
        episode_path = Path(args.audio)
        if not episode_path.exists():
            raise FileNotFoundError(f"Audio file not found: {episode_path}")
        podcast_dir = episode_path.parent
    else:
        podcast_dir, episode_path = pick_first_episode(args.downloads)

    logger.info(f"Picked podcast folder: {podcast_dir}")
    logger.info(f"Picked episode file:  {episode_path}")

    pipe = PodcastPipeline(
        whisper_model_size=args.whisper_model,
        hf_token=hf_token,
        use_gpu=True,
        diarization_on_gpu=args.diar_gpu,
    )

    segments, lang = pipe.transcribe(str(episode_path))
    diarized = pipe.diarize(str(episode_path))
    final_segments = match_segments(segments, diarized)

    # Print preview
    logger.info(f"Detected language: {lang}")
    logger.info("----- Preview (first 25 segments) -----")
    for seg in final_segments[:25]:
        logger.info(f"[{seg['start']:.2f}-{seg['end']:.2f}] Speaker: {seg['speaker']} -> {seg['text']}")

    # Save output
    out_path = Path(args.out) if args.out else episode_path.with_suffix(".whisper_diarized.json")
    payload = {
        "podcast_folder": str(podcast_dir),
        "episode_path": str(episode_path),
        "language": lang,
        "diarized_segments": diarized,
        "segments": final_segments,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Wrote output JSON: {out_path}")


if __name__ == "__main__":
    main()

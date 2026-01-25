import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # important on mounts like /mnt

import gc
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torchaudio
import whisper
import numpy as np
import librosa
import random
from pyannote.audio import Pipeline

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"No module named 'torchcodec'",
    category=UserWarning,
)
torchaudio.set_audio_backend("ffmpeg")

# NEW: gender model
from transformers import pipeline as hf_pipeline


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("podcast")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def list_podcast_dirs(root_dir: str = "fyyd_downloads") -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root.resolve()}")
    pods = sorted([p for p in root.iterdir() if p.is_dir()])
    if not pods:
        raise RuntimeError(f"No podcast folders found inside: {root.resolve()}")
    return pods

def list_audio_files(podcast_dir: Path) -> List[Path]:
    files = sorted([p for p in podcast_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])
    return files

def pick_random_episodes(root_dir: str = "fyyd_downloads", n_podcasts: int = 2, seed: int = 0) -> List[Tuple[Path, Path]]:
    """
    Picks n_podcasts random podcast folders, and 1 random episode from each.
    Returns list of (podcast_dir, episode_path).
    """
    rng = random.Random(seed if seed != 0 else None)

    podcasts = list_podcast_dirs(root_dir)
    if len(podcasts) < n_podcasts:
        n_podcasts = len(podcasts)

    chosen_podcasts = rng.sample(podcasts, k=n_podcasts)

    picks: List[Tuple[Path, Path]] = []
    for pod in chosen_podcasts:
        eps = list_audio_files(pod)
        if not eps:
            continue
        ep = rng.choice(eps)
        picks.append((pod, ep))

    if not picks:
        raise RuntimeError(f"Could not find any episodes in randomly chosen podcast folders under {root_dir}.")

    return picks
def estimate_gender_from_f0(
    wav: torch.Tensor,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 400.0,
    min_voiced_ratio: float = 0.25,
) -> Dict:
    """
    Returns perceived vocal category based on median F0.
    Output:
      {label, confidence, f0_median_hz, voiced_ratio}
    """
    y = wav.squeeze(0).cpu().numpy()

    # pyin returns f0 array with np.nan for unvoiced frames
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )

    if f0 is None:
        return {"label": "unknown", "confidence": 0.0, "f0_median_hz": None, "voiced_ratio": 0.0}

    voiced = ~np.isnan(f0)
    voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0

    if voiced_ratio < min_voiced_ratio:
        return {"label": "unknown", "confidence": 0.0, "f0_median_hz": None, "voiced_ratio": voiced_ratio}

    f0_voiced = f0[voiced]
    f0_median = float(np.median(f0_voiced))
    q25 = float(np.percentile(f0_voiced, 25))
    q75 = float(np.percentile(f0_voiced, 75))
    f0_iqr = q75 - q25
    # Simple, conservative bins (with a wide "unknown" overlap region)
    # You can tune these after checking your dataset distribution.
    if f0_median < 155:
        label = "male"
        confidence = min(1.0, (155 - f0_median) / 50.0)
    elif f0_median > 185:
        label = "female"
        confidence = min(1.0, (f0_median - 185) / 60.0)
    else:
        label = "borderline"
        # confidence reflects how far from the center of the overlap
        confidence = 1.0 - min(1.0, abs(f0_median - 170.0) / 15.0)


    return {
        "label": label,
        "confidence": float(confidence),
        "f0_median_hz": f0_median,
        "voiced_ratio": voiced_ratio,
        "f0_iqr_hz": f0_iqr
    }


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
    if hasattr(diarization_output, "itertracks"):
        ann = diarization_output
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


# ----------------------------
# NEW: gender helpers
# ----------------------------
def concat_turns(
    wav: torch.Tensor,
    sr: int,
    turns: List[Tuple[float, float]],
    max_total_sec: float = 90.0,
    min_turn_sec: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    Concatenate speaker turns into one waveform up to max_total_sec.
    Returns (waveform, seconds_used)
    """
    pieces = []
    total = 0.0
    n = wav.shape[1]

    for start, end in turns:
        dur = max(0.0, float(end) - float(start))
        if dur < min_turn_sec:
            continue

        s = max(0, min(int(float(start) * sr), n))
        e = max(0, min(int(float(end) * sr), n))
        if e <= s:
            continue

        pieces.append(wav[:, s:e])
        total += dur
        if total >= max_total_sec:
            break

    if not pieces:
        return torch.empty((1, 0), dtype=wav.dtype), 0.0

    out = torch.cat(pieces, dim=1)
    return out, float(out.shape[1] / sr)


def probs_to_label(probs: Dict[str, float], threshold: float, margin: float = 0.05, drop_child: bool = True):
    """
    Decide perceived vocal gender with:
      - optional removal of 'child'
      - confidence threshold
      - margin between top-1 and top-2
    """
    if not probs:
        return "unknown", 0.0

    p = dict(probs)

    if drop_child and "child" in p:
        p.pop("child", None)
        s = sum(p.values())
        if s > 0:
            p = {k: v / s for k, v in p.items()}

    items = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_conf = items[0]
    second_conf = items[1][1] if len(items) > 1 else 0.0

    if best_conf < threshold:
        return "unknown", float(best_conf)

    if (best_conf - second_conf) < margin:
        return "unknown", float(best_conf)

    return best_label, float(best_conf)



class PodcastPipeline:
    def __init__(
        self,
        whisper_model_size: str,
        hf_token: str,
        use_gpu: bool = True,
        diarization_on_gpu: bool = True,

        # NEW gender args
        enable_gender: bool = True,
        gender_model_id: str = "audeering/wav2vec2-large-robust-6-ft-age-gender",
        gender_device: str = "cpu",  # "cpu" or "cuda"
        gender_threshold: float = 0.80,
        max_speaker_sec: float = 90.0,
        min_turn_sec: float = 1.0,
    ):
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Whisper device: {self.device}")

        # Whisper
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        # Pyannote pipeline (use 3.1 to avoid revision errors)
        self.pipeline = self._load_pyannote("pyannote/speaker-diarization-3.1", hf_token)

        if diarization_on_gpu and torch.cuda.is_available():
            try:
                logger.info("Moving diarization pipeline to CUDA")
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            except Exception as e:
                logger.warning(f"Could not move diarization to CUDA, staying on CPU. Reason: {e}")

        # NEW: gender classifier
        self.enable_gender = enable_gender
        self.gender_threshold = float(gender_threshold)
        self.max_speaker_sec = float(max_speaker_sec)
        self.min_turn_sec = float(min_turn_sec)

        self.gender_clf = None
        if self.enable_gender:
            device_id = 0 if (gender_device == "cuda" and torch.cuda.is_available()) else -1
            logger.info(f"Loading gender model on {'cuda' if device_id == 0 else 'cpu'}: {gender_model_id}")
            self.gender_clf = hf_pipeline(
                task="audio-classification",
                model=gender_model_id,
                device=device_id
            )

    @staticmethod
    def _load_pyannote(model_id: str, token: str) -> Pipeline:
        for kwargs in ({"token": token}, {"use_auth_token": token}, {"hf_token": token}):
            try:
                return Pipeline.from_pretrained(model_id, **kwargs)
            except TypeError:
                continue
        return Pipeline.from_pretrained(model_id)

    def transcribe(self, audio_path: str) -> Tuple[List[Dict], str]:
        logger.info(f"Transcribing: {audio_path}")
        # Keep fp16 False for stability; you can set True after CUDA is working reliably
        result = self.whisper_model.transcribe(audio_path, verbose=False, fp16=False)
        return result["segments"], result.get("language", "unknown")

    def diarize(self, audio_path: str) -> List[Dict]:
        logger.info(f"Diarizing: {audio_path}")

        wav, sr = load_audio_mono_16k(audio_path)

        with torch.no_grad():
            diarization_output = self.pipeline({"waveform": wav, "sample_rate": sr})

        diarized_segments = extract_diarized_segments(diarization_output)

        torch.cuda.empty_cache()
        gc.collect()
        return diarized_segments

    # NEW: speaker-level gender
    def estimate_speaker_gender(self, audio_path: str, diarized: List[Dict]) -> Dict[str, Dict]:
        """
        Returns:
          {
            "SPEAKER_00": {"label": "male/female/child/unknown", "confidence": 0.91, "probs": {...}, "seconds_used": 84.2},
            ...
          }
        """
        if not self.enable_gender or self.gender_clf is None:
            return {}

        logger.info("Estimating perceived vocal gender per speaker")

        speaker_turns: Dict[str, List[Tuple[float, float]]] = {}
        for seg in diarized:
            speaker_turns.setdefault(seg["speaker"], []).append((float(seg["start"]), float(seg["end"])))

        wav, sr = load_audio_mono_16k(audio_path)

        out: Dict[str, Dict] = {}
        for speaker, turns in speaker_turns.items():
            sp_wav, seconds_used = concat_turns(
                wav, sr, turns,
                max_total_sec=self.max_speaker_sec,
                min_turn_sec=self.min_turn_sec
            )

            # Instead of hf gender model:
            g = estimate_gender_from_f0(sp_wav, sr)
            out[speaker] = {
                "label": g["label"],
                "confidence": g["confidence"],
                "f0_median_hz": g["f0_median_hz"],
                "voiced_ratio": g["voiced_ratio"],
                "seconds_used": float(seconds_used),
            }


        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default="", help="Optional path to a single audio file. If empty, auto-picks from fyyd_downloads.")
    ap.add_argument("--downloads", type=str, default="fyyd_downloads", help="Root folder of downloaded podcasts.")
    ap.add_argument("--whisper_model", type=str, default="small", help="Whisper model size (tiny/base/small/medium/large).")
    ap.add_argument("--diar_gpu", action="store_true", help="Try to run diarization on GPU too.")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path.")

    # NEW gender CLI knobs
    ap.add_argument("--gender", action="store_true", help="Enable perceived vocal gender analysis.")
    ap.add_argument("--gender_device", type=str, default="cuda", choices=["cpu", "cuda"], help="Run gender model on cpu/cuda.")
    ap.add_argument("--gender_threshold", type=float, default=0.80, help="Confidence threshold for gender label.")
    ap.add_argument("--max_speaker_sec", type=float, default=90.0, help="Max seconds per speaker to classify.")
    ap.add_argument("--sanity", action="store_true", help="Run sanity checks: process 2 random podcast episodes.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 = nondeterministic).")

    args = ap.parse_args()

    hf_token = os.getenv("PYANNOTE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("Missing Hugging Face token. Export PYANNOTE_TOKEN (recommended) before running.")

    if args.audio:
        episode_path = Path(args.audio)
        if not episode_path.exists():
            raise FileNotFoundError(f"Audio file not found: {episode_path}")
        jobs = [(episode_path.parent, episode_path)]

    elif args.sanity:
        jobs = pick_random_episodes(args.downloads, n_podcasts=2, seed=args.seed)

    else:
        podcast_dir, episode_path = pick_first_episode(args.downloads)
        jobs = [(podcast_dir, episode_path)]



    pipe = PodcastPipeline(
        whisper_model_size=args.whisper_model,
        hf_token=hf_token,
        use_gpu=True,
        diarization_on_gpu=args.diar_gpu,

        enable_gender=bool(args.gender),
        gender_device=args.gender_device,
        gender_threshold=args.gender_threshold,
        max_speaker_sec=args.max_speaker_sec,
    )

    for podcast_dir, episode_path in jobs:
        logger.info(f"Picked podcast folder: {podcast_dir}")
        logger.info(f"Picked episode file:  {episode_path}")

        whisper_segments, lang = pipe.transcribe(str(episode_path))
        whisper_text_full = " ".join(s.get("text", "").strip() for s in whisper_segments).strip()

        diarized = pipe.diarize(str(episode_path))
        final_segments = match_segments(whisper_segments, diarized)

        speaker_gender = {}
        if args.gender:
            speaker_gender = pipe.estimate_speaker_gender(str(episode_path), diarized)
            for seg in final_segments:
                g = speaker_gender.get(seg["speaker"])
                if g:
                    seg["gender"] = g["label"]
                    seg["gender_confidence"] = g["confidence"]
                    seg["f0_median_hz"] = g.get("f0_median_hz")
                    seg["voiced_ratio"] = g.get("voiced_ratio")
                else:
                    seg["gender"] = "unknown"
                    seg["gender_confidence"] = 0.0

        # preview
        logger.info(f"Detected language: {lang}")
        logger.info("----- Preview (first 10 segments) -----")
        for seg in final_segments[:10]:
            if args.gender:
                logger.info(
                    f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']} | {seg['gender']} "
                    f"({seg['gender_confidence']:.2f}) -> {seg['text']}"
                )
            else:
                logger.info(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']} -> {seg['text']}")

        # output path (unique per episode)
        out_path = episode_path.with_suffix(".whisper_diarized.json")
        payload = {
            "podcast_folder": str(podcast_dir),
            "episode_path": str(episode_path),

            # Whisper direct outputs
            "whisper_language": lang,
            "whisper_segments": whisper_segments,
            "whisper_text_full": whisper_text_full,

            # Enriched outputs
            "diarized_segments": diarized,
            "segments": final_segments,
            "speaker_gender": speaker_gender,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Wrote output JSON: {out_path}")


if __name__ == "__main__":
    main()

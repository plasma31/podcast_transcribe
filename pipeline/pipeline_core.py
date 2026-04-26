import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import gc
import hashlib
import json
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import whisper
from pyannote.audio import Pipeline

warnings.filterwarnings("ignore", message=r".*torchcodec is not installed correctly.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*set_audio_backend has been deprecated.*", category=UserWarning)

logger = logging.getLogger("podcast.pipeline")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


@dataclass
class EpisodeArtifacts:
    episode_record: Dict[str, Any]
    segment_records: List[Dict[str, Any]]
    debug_payload: Dict[str, Any]


def stable_episode_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()


def list_podcast_dirs(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root.resolve()}")
    pods = sorted([p for p in root.iterdir() if p.is_dir()])
    if not pods:
        raise RuntimeError(f"No podcast folders found inside: {root.resolve()}")
    return pods


def list_audio_files(podcast_dir: Path) -> List[Path]:
    return sorted([p for p in podcast_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def build_episode_inventory(downloads_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for podcast_dir in list_podcast_dirs(downloads_root):
        for audio_path in list_audio_files(podcast_dir):
            stat = audio_path.stat()
            rows.append(
                {
                    "episode_id": stable_episode_id(audio_path),
                    "podcast_folder": podcast_dir.name,
                    "podcast_dir": str(podcast_dir.resolve()),
                    "episode_path": str(audio_path.resolve()),
                    "episode_name": audio_path.stem,
                    "audio_ext": audio_path.suffix.lower(),
                    "file_size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return rows


def load_audio_mono_16k(audio_path: str) -> Tuple[torch.Tensor, int]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    wav = torch.from_numpy(y).float().unsqueeze(0)
    return wav, sr


def extract_diarized_segments(diarization_output) -> List[Dict[str, Any]]:
    if hasattr(diarization_output, "itertracks"):
        ann = diarization_output
    elif hasattr(diarization_output, "exclusive_speaker_diarization"):
        ann = diarization_output.exclusive_speaker_diarization
    elif hasattr(diarization_output, "speaker_diarization"):
        ann = diarization_output.speaker_diarization
    else:
        raise TypeError(f"Unexpected diarization output type={type(diarization_output)}")

    diarized_segments: List[Dict[str, Any]] = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        diarized_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return diarized_segments


def match_segments(transcribed: List[Dict[str, Any]], diarized: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for idx, t in enumerate(transcribed):
        t_start = float(t["start"])
        t_end = float(t["end"])
        best_speaker = "Unknown"
        best_overlap = 0.0

        for d in diarized:
            overlap_start = max(t_start, float(d["start"]))
            overlap_end = min(t_end, float(d["end"]))
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        result.append(
            {
                "segment_idx": idx,
                "start": t_start,
                "end": t_end,
                "speaker": best_speaker,
                "text": t.get("text", "").strip(),
            }
        )
    return result


def concat_turns(
    wav: torch.Tensor,
    sr: int,
    turns: List[Tuple[float, float]],
    max_total_sec: float = 90.0,
    min_turn_sec: float = 1.0,
) -> Tuple[torch.Tensor, float]:
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


def estimate_gender_from_f0(
    wav: torch.Tensor,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 400.0,
    min_voiced_ratio: float = 0.25,
) -> Dict[str, Any]:
    if wav.numel() == 0:
        return {"label": "unknown", "confidence": 0.0, "f0_median_hz": None, "voiced_ratio": 0.0, "f0_iqr_hz": None}

    y = wav.squeeze(0).cpu().numpy()
    f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256)
    if f0 is None:
        return {"label": "unknown", "confidence": 0.0, "f0_median_hz": None, "voiced_ratio": 0.0, "f0_iqr_hz": None}

    voiced = ~np.isnan(f0)
    voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    if voiced_ratio < min_voiced_ratio:
        return {"label": "unknown", "confidence": 0.0, "f0_median_hz": None, "voiced_ratio": voiced_ratio, "f0_iqr_hz": None}

    f0_voiced = f0[voiced]
    f0_median = float(np.median(f0_voiced))
    q25 = float(np.percentile(f0_voiced, 25))
    q75 = float(np.percentile(f0_voiced, 75))
    f0_iqr = q75 - q25

    if f0_median < 155:
        label = "male"
        confidence = min(1.0, (155 - f0_median) / 50.0)
    elif f0_median > 185:
        label = "female"
        confidence = min(1.0, (f0_median - 185) / 60.0)
    else:
        label = "borderline"
        confidence = 1.0 - min(1.0, abs(f0_median - 170.0) / 15.0)

    return {
        "label": label,
        "confidence": float(confidence),
        "f0_median_hz": f0_median,
        "voiced_ratio": voiced_ratio,
        "f0_iqr_hz": f0_iqr,
    }


def _load_pyannote(model_id: str, token: str) -> Pipeline:
    for kwargs in ({"token": token}, {"use_auth_token": token}, {"hf_token": token}):
        try:
            return Pipeline.from_pretrained(model_id, **kwargs)
        except TypeError:
            continue
    return Pipeline.from_pretrained(model_id)


class PodcastPipeline:
    def __init__(
        self,
        whisper_model_size: str,
        hf_token: str,
        use_gpu: bool = True,
        diarization_on_gpu: bool = False,
        enable_gender: bool = True,
        diar_model: str = "pyannote/speaker-diarization-3.1",
        max_speaker_sec: float = 90.0,
        min_turn_sec: float = 1.0,
    ):
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.enable_gender = enable_gender
        self.max_speaker_sec = float(max_speaker_sec)
        self.min_turn_sec = float(min_turn_sec)

        logger.info("CUDA available: %s", torch.cuda.is_available())
        logger.info("Whisper device: %s", self.device)
        logger.info("Loading Whisper model: %s", whisper_model_size)
        self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)

        logger.info("Loading pyannote pipeline: %s", diar_model)
        self.pipeline = _load_pyannote(diar_model, hf_token)
        if diarization_on_gpu and torch.cuda.is_available():
            try:
                logger.info("Moving diarization pipeline to CUDA")
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            except Exception as exc:
                logger.warning("Could not move diarization to CUDA, staying on CPU. Reason: %s", exc)

    def transcribe(self, audio_path: str) -> Tuple[List[Dict[str, Any]], str]:
        logger.info("Transcribing: %s", audio_path)
        result = self.whisper_model.transcribe(audio_path, verbose=False, fp16=False)
        return result["segments"], result.get("language", "unknown")

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        logger.info("Diarizing: %s", audio_path)
        wav, sr = load_audio_mono_16k(audio_path)
        with torch.no_grad():
            diarization_output = self.pipeline({"waveform": wav, "sample_rate": sr})
        diarized_segments = extract_diarized_segments(diarization_output)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return diarized_segments

    def estimate_speaker_gender(self, audio_path: str, diarized: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not self.enable_gender:
            return {}

        logger.info("Estimating perceived vocal gender per speaker")
        speaker_turns: Dict[str, List[Tuple[float, float]]] = {}
        for seg in diarized:
            speaker_turns.setdefault(seg["speaker"], []).append((float(seg["start"]), float(seg["end"])))

        wav, sr = load_audio_mono_16k(audio_path)
        out: Dict[str, Dict[str, Any]] = {}
        for speaker, turns in speaker_turns.items():
            sp_wav, seconds_used = concat_turns(
                wav,
                sr,
                turns,
                max_total_sec=self.max_speaker_sec,
                min_turn_sec=self.min_turn_sec,
            )
            g = estimate_gender_from_f0(sp_wav, sr)
            out[speaker] = {
                "label": g["label"],
                "confidence": g["confidence"],
                "f0_median_hz": g["f0_median_hz"],
                "voiced_ratio": g["voiced_ratio"],
                "f0_iqr_hz": g.get("f0_iqr_hz"),
                "seconds_used": float(seconds_used),
            }
        return out

    def process_episode(self, episode_path: str, podcast_folder: Optional[str] = None) -> EpisodeArtifacts:
        t0 = time.perf_counter()
        episode_path_obj = Path(episode_path)
        episode_id = stable_episode_id(episode_path_obj)
        podcast_folder = podcast_folder or episode_path_obj.parent.name

        whisper_segments, lang = self.transcribe(episode_path)
        whisper_text_full = " ".join(seg.get("text", "").strip() for seg in whisper_segments).strip()

        diarized = self.diarize(episode_path)
        final_segments = match_segments(whisper_segments, diarized)

        speaker_gender: Dict[str, Dict[str, Any]] = {}
        if self.enable_gender:
            speaker_gender = self.estimate_speaker_gender(episode_path, diarized)
            for seg in final_segments:
                g = speaker_gender.get(seg["speaker"])
                if g:
                    seg["gender"] = g["label"]
                    seg["gender_confidence"] = g["confidence"]
                    seg["f0_median_hz"] = g.get("f0_median_hz")
                    seg["voiced_ratio"] = g.get("voiced_ratio")
                    seg["f0_iqr_hz"] = g.get("f0_iqr_hz")
                else:
                    seg["gender"] = "unknown"
                    seg["gender_confidence"] = 0.0
                    seg["f0_median_hz"] = None
                    seg["voiced_ratio"] = None
                    seg["f0_iqr_hz"] = None

        runtime_sec = time.perf_counter() - t0
        speakers = sorted({s["speaker"] for s in diarized})

        episode_record = {
            "episode_id": episode_id,
            "podcast_folder": podcast_folder,
            "episode_path": str(episode_path_obj.resolve()),
            "episode_name": episode_path_obj.stem,
            "whisper_language": lang,
            "whisper_text_full": whisper_text_full,
            "runtime_sec": float(runtime_sec),
            "n_whisper_segments": len(whisper_segments),
            "n_diarized_segments": len(diarized),
            "n_segments": len(final_segments),
            "n_speakers": len(speakers),
            "speakers_json": json.dumps(speakers, ensure_ascii=False),
            "speaker_gender_json": json.dumps(speaker_gender, ensure_ascii=False),
        }

        segment_records: List[Dict[str, Any]] = []
        for seg in final_segments:
            segment_records.append(
                {
                    "episode_id": episode_id,
                    "podcast_folder": podcast_folder,
                    "episode_path": str(episode_path_obj.resolve()),
                    "episode_name": episode_path_obj.stem,
                    "whisper_language": lang,
                    "segment_idx": seg["segment_idx"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "gender": seg.get("gender", "unknown"),
                    "gender_confidence": seg.get("gender_confidence", 0.0),
                    "f0_median_hz": seg.get("f0_median_hz"),
                    "voiced_ratio": seg.get("voiced_ratio"),
                    "f0_iqr_hz": seg.get("f0_iqr_hz"),
                    "text": seg.get("text", ""),
                }
            )

        debug_payload = {
            "episode_id": episode_id,
            "podcast_folder": podcast_folder,
            "episode_path": str(episode_path_obj.resolve()),
            "whisper_language": lang,
            "whisper_segments": whisper_segments,
            "whisper_text_full": whisper_text_full,
            "diarized_segments": diarized,
            "segments": final_segments,
            "speaker_gender": speaker_gender,
            "runtime_sec": float(runtime_sec),
        }

        return EpisodeArtifacts(
            episode_record=episode_record,
            segment_records=segment_records,
            debug_payload=debug_payload,
        )

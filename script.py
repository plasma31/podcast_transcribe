import whisper
from pyannote.audio import Pipeline
import requests
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import torch
import gc
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PodcastTranscriber:
    def __init__(self, model_size: str = "small", use_gpu: bool = torch.cuda.is_available()):
        """Initialize the transcriber with specified model size and GPU settings."""
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Set specific compute type for better stability
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            self.whisper_model = whisper.load_model(model_size, device=self.device)
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token="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"
            )
            
            if use_gpu and torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device(self.device))
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise

    def transcribe_podcast(self, audio_path: str) -> Tuple[List[Dict], str]:
        """Transcribe audio file and return segments with language detection."""
        logger.info(f"Transcribing audio file: {audio_path}")
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                verbose=True,
                fp16=False,
                temperature=0.2
            )
            return result["segments"], result.get("language", "unknown")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def diarize_podcast(self, audio_path: str) -> List[Dict]:
        """Perform speaker diarization on audio file."""
        logger.info(f"Performing speaker diarization on: {audio_path}")
        try:
            with torch.no_grad():
                diarization = self.pipeline(audio_path)
            
            segments = [
                {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            
            torch.cuda.empty_cache()
            gc.collect()
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise

    @staticmethod
    def match_segments(transcribed: List[Dict], diarized: List[Dict]) -> List[Dict]:
        """Match transcribed segments with speaker segments using improved overlap detection."""
        result = []
        for t_seg in transcribed:
            # Find the speaker segment with maximum overlap
            max_overlap = 0
            current_speaker = "Unknown"
            
            for d_seg in diarized:
                overlap_start = max(t_seg["start"], d_seg["start"])
                overlap_end = min(t_seg["end"], d_seg["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    current_speaker = d_seg["speaker"]
            
            result.append({
                "start": t_seg["start"],
                "end": t_seg["end"],
                "speaker": current_speaker,
                "text": t_seg["text"],
                "language": t_seg.get("language")
            })
        return result

def download_audio(url: str, output_path: str) -> None:
    """Download audio file from URL."""
    logger.info(f"Downloading audio from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Audio downloaded successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise

def main(audio_file: str, model_size: str = "small"):
    """Main execution function."""
    try:
        # Validate input file
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Initialize transcriber
        transcriber = PodcastTranscriber(model_size=model_size)
        
        # Process audio
        transcribed, lang = transcriber.transcribe_podcast(str(audio_path))
        diarized = transcriber.diarize_podcast(str(audio_path))
        final_segments = transcriber.match_segments(transcribed, diarized)

        # Output results
        logger.info(f"Detected language: {lang}")
        for seg in final_segments:
            logger.info(
                f"[{seg['start']:.2f}-{seg['end']:.2f}] "
                f"Speaker: {seg['speaker']} ({seg['language']}) -> {seg['text']}"
            )
        
        return final_segments

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    audio_file_path = "/mnt/e/Masters Data/SEM 5/Thesis/Podcasts Transcription/audio.mp3"
    main(audio_file_path)
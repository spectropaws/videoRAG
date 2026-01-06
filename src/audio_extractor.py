"""Audio extraction from video files."""

import ffmpeg
import logging
from pathlib import Path
from typing import Union, Optional
from .utils.file_utils import ensure_directory, get_base_filename, is_valid_video_file


logger = logging.getLogger("video_rag.audio_extractor")


class AudioExtractor:
    """Extract audio from video files using ffmpeg."""
    
    def __init__(self, output_dir: str = "data/audio", sample_rate: int = 16000):
        """Initialize AudioExtractor.
        
        Args:
            output_dir: Directory to save extracted audio files
            sample_rate: Sample rate for audio extraction
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        ensure_directory(self.output_dir)
    
    def extract_audio(
        self, 
        video_path: Union[str, Path], 
        output_filename: Optional[str] = None
    ) -> Path:
        """Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_filename: Optional output filename (without extension)
            
        Returns:
            Path to extracted audio file
            
        Raises:
            ValueError: If video file is invalid
            RuntimeError: If audio extraction fails
        """
        video_path = Path(video_path)
        
        # Validate input
        if not video_path.exists():
            raise ValueError(f"Video file does not exist: {video_path}")
        
        if not is_valid_video_file(video_path):
            raise ValueError(f"Invalid video file format: {video_path}")
        
        # Determine output filename
        if output_filename is None:
            output_filename = get_base_filename(video_path)
        
        output_path = self.output_dir / f"{output_filename}.wav"
        
        try:
            logger.info(f"Extracting audio from {video_path} to {output_path}")
            
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    format='wav',
                    acodec='pcm_s16le',
                    ar=self.sample_rate,
                    ac=1  # mono
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )
            
            logger.info(f"Audio extraction completed: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = f"Failed to extract audio: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def batch_extract(self, video_paths: list[Union[str, Path]]) -> list[Path]:
        """Extract audio from multiple video files.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of extracted audio file paths
        """
        audio_paths = []
        
        for video_path in video_paths:
            try:
                audio_path = self.extract_audio(video_path)
                audio_paths.append(audio_path)
            except Exception as e:
                logger.error(f"Failed to extract audio from {video_path}: {e}")
                continue
        
        return audio_paths

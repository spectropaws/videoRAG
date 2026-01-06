"""File utility functions."""

import os
from pathlib import Path
from typing import Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(filepath: Union[str, Path]) -> str:
    """Get file extension from filepath."""
    return Path(filepath).suffix.lower()


def get_base_filename(filepath: Union[str, Path]) -> str:
    """Get base filename without extension."""
    return Path(filepath).stem


def is_valid_video_file(filepath: Union[str, Path]) -> bool:
    """Check if file is a valid video file."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return get_file_extension(filepath) in video_extensions


def is_valid_audio_file(filepath: Union[str, Path]) -> bool:
    """Check if file is a valid audio file."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    return get_file_extension(filepath) in audio_extensions

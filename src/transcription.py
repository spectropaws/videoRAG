"""Audio transcription and diarization using WhisperX optimized for CUDA."""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import whisperx

from .utils.file_utils import (ensure_directory, get_base_filename,
                               is_valid_audio_file)

logger = logging.getLogger("video_rag.transcription")


class TranscriptionService:
    """CUDA-optimized transcription and diarization service using WhisperX."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: int = 16,
        output_dir: str = "data/transcripts",
        hf_token: Optional[str] = None,
        enable_memory_optimization: bool = True
    ):
        """Initialize CUDA-optimized TranscriptionService.

        Args:
            model_size: Whisper model size
            device: Device to use (auto-detects CUDA if available)
            compute_type: Compute type for inference (auto-optimized for CUDA)
            batch_size: Batch size for processing
            output_dir: Directory to save transcriptions
            hf_token: Hugging Face token for diarization models
            enable_memory_optimization: Enable GPU memory optimization techniques
        """
        self.model_size = model_size
        self.enable_memory_optimization = enable_memory_optimization

        # CUDA-optimized device selection
        self.device = self._get_optimal_device(device)

        # CUDA-optimized compute type selection
        self.compute_type = self._get_optimal_compute_type(compute_type)

        # Adjust batch size based on GPU memory
        self.batch_size = self._optimize_batch_size(batch_size)

        self.output_dir = Path(output_dir)
        self.hf_token = hf_token

        ensure_directory(self.output_dir)

        # Initialize models
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        self._log_gpu_info()
        self._load_models()

    def _get_optimal_device(self, device: Optional[str]) -> str:
        """Determine optimal device for CUDA usage."""
        if device is not None:
            return device

        if torch.cuda.is_available():
            # Get the best GPU (highest memory)
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                best_gpu = 0
                max_memory = 0
                for i in range(gpu_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_gpu = i
                device = f"cuda:{best_gpu}"
                logger.info(f"Selected GPU {best_gpu} with {max_memory / 1e9:.1f}GB memory")
            else:
                device = "cuda"
            return device
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"

    def _get_optimal_compute_type(self, compute_type: Optional[str]) -> str:
        """Determine optimal compute type for CUDA."""
        if compute_type is not None:
            return compute_type

        if self.device == "cpu":
            return "int8"

        # Check GPU compute capability for optimal precision
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Modern GPUs support float16, older ones might need int8
            if capability[0] >= 7:  # Volta and newer
                return "float16"
            else:
                logger.warning(
                    "Older GPU detected, using int8 for compatibility")
                return "int8"

        return "float16"

    def _optimize_batch_size(self, batch_size: int) -> int:
        """Optimize batch size based on available GPU memory."""
        if self.device == "cpu":
            return min(batch_size, 8)  # Conservative for CPU

        if torch.cuda.is_available():
            # Get available GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(
                0).total_memory / 1e9

            # Adjust batch size based on GPU memory and model size
            if "large" in self.model_size:
                if gpu_memory_gb >= 24:
                    return min(batch_size, 32)
                elif gpu_memory_gb >= 12:
                    return min(batch_size, 16)
                elif gpu_memory_gb >= 8:
                    return min(batch_size, 8)
                else:
                    return min(batch_size, 4)
            elif "medium" in self.model_size:
                return min(batch_size, 24)
            else:  # small, base models
                return min(batch_size, 48)

        return batch_size

    def _log_gpu_info(self):
        """Log GPU information for debugging."""
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Compute Type: {self.compute_type}")
            logger.info(f"Optimized Batch Size: {self.batch_size}")

            if "cuda" in self.device:
                device_idx = int(self.device.split(
                    ":")[-1]) if ":" in self.device else 0
                props = torch.cuda.get_device_properties(device_idx)
                logger.info(f"GPU: {props.name}")
                logger.info(f"GPU Memory: {props.total_memory / 1e9:.1f}GB")
                logger.info(f"CUDA Compute Capability: {props.major}.{props.minor}")
        else:
            logger.warning("CUDA not available, using CPU")

    def _optimize_gpu_memory(self):
        """Apply GPU memory optimization techniques."""
        if self.enable_memory_optimization and torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Enable cuDNN autotuner for optimal performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    def _load_models(self):
        """Load WhisperX models with CUDA optimization."""
        try:
            self._optimize_gpu_memory()

            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )

            # Memory optimization after model loading
            if self.enable_memory_optimization:
                self._optimize_gpu_memory()

            logger.info("Loading alignment model")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device
            )

            if self.hf_token:
                logger.info("Loading diarization model")
                self.diarize_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device
                )
            else:
                logger.warning(
                    "No HF token provided, diarization will be disabled")

            # Final memory optimization
            if self.enable_memory_optimization:
                self._optimize_gpu_memory()

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Try fallback to CPU if CUDA fails
            if self.device != "cpu":
                logger.info("Attempting fallback to CPU")
                self.device = "cpu"
                self.compute_type = "int8"
                self._load_models()
            else:
                raise

    def unload_model(self):
        """Unload WhisperX models and free GPU memory."""
        try:
            logger.info("Unloading WhisperX models to free GPU memory...")

            # Delete heavy model references
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None

            if hasattr(self, "align_model") and self.align_model is not None:
                del self.align_model
                self.align_model = None

            if hasattr(self, "align_metadata") and self.align_metadata is not None:
                del self.align_metadata
                self.align_metadata = None

            if hasattr(self, "diarize_model") and self.diarize_model is not None:
                del self.diarize_model
                self.diarize_model = None

            # Run garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                "All WhisperX models unloaded and GPU memory released.")

        except Exception as e:
            logger.warning(f"Failed to unload models cleanly: {e}")

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: str = "en",
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict[str, Any]:
        """Transcribe audio file with CUDA-optimized processing.

        Args:
            audio_path: Path to audio file
            language: Language code for transcription
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization

        Returns:
            Dictionary containing transcription and diarization results
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise ValueError(f"Audio file does not exist: {audio_path}")

        if not is_valid_audio_file(audio_path):
            raise ValueError(f"Invalid audio file format: {audio_path}")

        logger.info(f"Transcribing audio: {audio_path}")

        try:
            # Memory optimization before processing
            if self.enable_memory_optimization:
                self._optimize_gpu_memory()

            # Load audio
            audio = whisperx.load_audio(str(audio_path))

            # Auto-detect language if needed
            if language == "auto":
                logger.info("Detecting language...")
                with torch.no_grad():  # Memory optimization for inference
                    detected_lang = self.model.detect_language(audio)
                logger.info(f"Detected language: {detected_lang}")
                language = detected_lang

            # Transcribe with memory optimization
            logger.info("Running transcription...")
            with torch.no_grad():
                result = self.model.transcribe(
                    audio,
                    batch_size=self.batch_size,
                    language=language
                )

            # Clear intermediate results from GPU memory
            if self.enable_memory_optimization and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Align
            logger.info("Aligning transcription...")
            with torch.no_grad():
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )

            # Clear alignment results from GPU memory
            if self.enable_memory_optimization and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Diarize
            if self.diarize_model and self.hf_token:
                logger.info("Running diarization...")
                with torch.no_grad():
                    diarize_segments = self.diarize_model(
                        audio,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )

                # Assign speakers
                result = whisperx.assign_word_speakers(
                    diarize_segments,
                    result
                )
            else:
                logger.info("Skipping diarization (no model or token)")

            # Final memory cleanup
            if self.enable_memory_optimization:
                del audio  # Free audio data
                self._optimize_gpu_memory()

            # Format results
            formatted_result = self._format_results(result, audio_path)

            logger.info(f"Transcription completed for {audio_path}")
            return formatted_result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            # Memory cleanup on error
            if self.enable_memory_optimization and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise

    def _format_results(self, result: Dict, audio_path: Path) -> Dict[str, Any]:
        """Format transcription results."""
        segments = result.get("segments", [])

        # Create formatted segments
        formatted_segments = []
        full_text = ""

        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()

            if text:
                formatted_segments.append({
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                    "text": text
                })

                full_text += f"[{speaker}] {text} "

        return {
            "audio_file": str(audio_path),
            "segments": formatted_segments,
            "full_text": full_text.strip(),
            "total_segments": len(formatted_segments),
            "speakers": list(set(seg["speaker"] for seg in formatted_segments if seg["speaker"] != "Unknown"))
        }

    def save_transcription(
        self,
        transcription: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> Path:
        """Save transcription to JSON and text files.

        Args:
            transcription: Transcription results
            output_filename: Optional output filename

        Returns:
            Path to saved JSON file
        """
        if output_filename is None:
            audio_file = Path(transcription["audio_file"])
            output_filename = get_base_filename(audio_file)

        # Save JSON
        json_path = self.output_dir / f"{output_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)

        # Save text
        txt_path = self.output_dir / f"{output_filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription["full_text"])

        # Save formatted transcript
        formatted_path = self.output_dir / f"{output_filename}_formatted.txt"
        with open(formatted_path, 'w', encoding='utf-8') as f:
            for segment in transcription["segments"]:
                timestamp = f"[{segment['start']:.2f}s - {segment['end']:.2f}s]"
                f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n")

        logger.info(f"Transcription saved to {json_path}")
        return json_path

    def batch_transcribe(
        self,
        audio_paths: List[Union[str, Path]],
        **transcription_kwargs
    ) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files with GPU optimization.

        Args:
            audio_paths: List of audio file paths
            **transcription_kwargs: Arguments passed to transcribe_audio

        Returns:
            List of transcription results
        """
        results = []

        for i, audio_path in enumerate(audio_paths):
            try:
                logger.info(f"Processing file {i+1}/{len(audio_paths)}: {audio_path}")
                result = self.transcribe_audio(
                    audio_path, **transcription_kwargs)
                self.save_transcription(result)
                results.append(result)

                # Memory optimization between files
                if self.enable_memory_optimization and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {e}")
                continue

        return results

    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        memory_info = {}
        for i in range(torch.cuda.device_count()):
            memory_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_properties(i).name,
                "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1e9,
                "cached_memory_gb": torch.cuda.memory_reserved(i) / 1e9,
            }

        return {
            "gpu_available": True,
            "current_device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "memory_info": memory_info
        }

    def cleanup_gpu_memory(self):
        """Manual GPU memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleaned up")

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'enable_memory_optimization') and self.enable_memory_optimization:
            self.cleanup_gpu_memory()

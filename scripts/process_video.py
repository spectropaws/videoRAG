"""Process video file: extract audio, transcribe, and add to vector database."""

import torch
import gc
import logging
import sys
from pathlib import Path
import click
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_database import VectorDatabase
from src.utils.logging_config import setup_logging
from src.utils.file_utils import is_valid_video_file
from src.transcription import TranscriptionService
from src.audio_extractor import AudioExtractor


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


@click.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config", "-c", default="config/config.yaml", help="Path to configuration file"
)
@click.option(
    "--output-name",
    "-o",
    default=None,
    help="Output name for files (default: video filename)",
)
@click.option(
    "--language",
    "-l",
    default="auto",
    help="Language for transcription (default: auto)",
)
@click.option(
    "--min-speakers",
    default=1,
    type=int,
    help="Minimum number of speakers for diarization",
)
@click.option(
    "--max-speakers",
    default=10,
    type=int,
    help="Maximum number of speakers for diarization",
)
@click.option(
    "--skip-audio-extraction",
    is_flag=True,
    help="Skip audio extraction (use existing audio file)",
)
@click.option(
    "--skip-transcription",
    is_flag=True,
    help="Skip transcription (use existing transcription file)",
)
@click.option("--skip-vector-db", is_flag=True, help="Skip adding to vector database")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.option("--log-file", default=None, help="Log file path (optional)")
def main(
    video_path: Path,
    config: str,
    output_name: str,
    language: str,
    min_speakers: int,
    max_speakers: int,
    skip_audio_extraction: bool,
    skip_transcription: bool,
    skip_vector_db: bool,
    log_level: str,
    log_file: str,
):
    """Process video file: extract audio, transcribe, and add to vector database.

    VIDEO_PATH: Path to the input video file
    """
    # Setup logging
    logger = setup_logging(log_level, log_file)

    try:
        # Load configuration
        config_data = load_config(config)
        logger.info(f"Loaded configuration from {config}")

        # Validate input
        if not is_valid_video_file(video_path):
            raise ValueError(f"Invalid video file: {video_path}")

        # Determine output name
        if output_name is None:
            output_name = video_path.stem

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output name: {output_name}")

        audio_path = None
        transcription = None

        # Step 1: Extract audio
        if not skip_audio_extraction:
            logger.info("=== STEP 1: Audio Extraction ===")
            audio_config = config_data.get("audio", {})

            extractor = AudioExtractor(
                output_dir=audio_config.get("output_dir", "data/audio"),
                sample_rate=audio_config.get("sample_rate", 16000),
            )

            audio_path = extractor.extract_audio(video_path, output_name)
            logger.info(f"Audio extracted to: {audio_path}")
        else:
            # Look for existing audio file
            audio_dir = Path(
                config_data.get("audio", {}).get("output_dir", "data/audio")
            )
            audio_path = audio_dir / f"{output_name}.wav"
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.info(f"Using existing audio file: {audio_path}")

        # Step 2: Transcription and diarization
        if not skip_transcription:
            logger.info("=== STEP 2: Transcription and Diarization ===")
            transcription_config = config_data.get("transcription", {})
            diarization_config = config_data.get("diarization", {})

            transcription_service = TranscriptionService(
                model_size=transcription_config.get("model", "large-v3"),
                device=transcription_config.get("device", "cuda"),
                compute_type=transcription_config.get(
                    "compute_type", "float16"),
                batch_size=transcription_config.get("batch_size", 16),
                output_dir=transcription_config.get(
                    "output_dir", "data/transcripts"),
                hf_token=diarization_config.get("hf_token"),
            )

            transcription = transcription_service.transcribe_audio(
                audio_path,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # Save transcription
            json_path = transcription_service.save_transcription(
                transcription, output_name
            )
            logger.info(f"Transcription saved to: {json_path}")

        else:
            # Look for existing transcription file
            transcript_dir = Path(
                config_data.get("transcription", {}).get(
                    "output_dir", "data/transcripts"
                )
            )
            json_path = transcript_dir / f"{output_name}.json"
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Transcription file not found: {json_path}")
            logger.info(f"Using existing transcription file: {json_path}")

        # Free GPU memory before moving to vector database step
        try:
            if not skip_transcription:
                logger.info("Releasing GPU memory used by transcription model...")
                transcription_service.unload_model()
        except Exception as mem_err:
            logger.warning(f"Failed to release GPU memory: {mem_err}")

        # Step 3: Add to vector database
        if not skip_vector_db:
            logger.info("=== STEP 3: Vector Database ===")
            vector_db_config = config_data.get("vector_db", {})

            vector_db = VectorDatabase(
                persist_directory=vector_db_config.get(
                    "persist_directory", "data/vector_db"
                ),
                collection_name=vector_db_config.get(
                    "collection_name", "video_transcripts"
                ),
                embedding_model=vector_db_config.get(
                    "embedding_model", "all-MiniLM-L6-v2"
                ),
                chunk_size=vector_db_config.get("chunk_size", 1000),
                chunk_overlap=vector_db_config.get("chunk_overlap", 200),
            )

            if transcription:
                chunks_added = vector_db.add_transcription(
                    transcription, output_name)
            else:
                chunks_added = vector_db.add_transcription_file(json_path)

            logger.info(f"Added {chunks_added} chunks to vector database")

            # Display collection info
            collection_info = vector_db.get_collection_info()
            logger.info(f"Vector database info: {collection_info}")

        logger.info("=== PROCESSING COMPLETE ===")
        logger.info(f"Video: {video_path}")
        if audio_path:
            logger.info(f"Audio: {audio_path}")
        if not skip_transcription or not skip_vector_db:
            logger.info(
                f"Transcription: {json_path if 'json_path' in locals() else 'N/A'}"
            )
        if not skip_vector_db:
            logger.info(
                f"Vector DB chunks: {chunks_added if 'chunks_added' in locals() else 'N/A'}"
            )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

from setuptools import setup, find_packages

setup(
    name="video-rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ffmpeg-python>=0.2.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "whisperx>=3.1.1",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.2",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "llama-cpp-python>=0.2.20",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Video transcription and RAG-based Q&A system",
    entry_points={
        "console_scripts": [
            "process-video=scripts.process_video:main",
            "query-system=scripts.query_system:main",
        ],
    },
)

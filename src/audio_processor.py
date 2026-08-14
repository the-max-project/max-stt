import asyncio
import logging
import numpy as np

MAX_CONCURRENT_TRANSCRIPTIONS = 2
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# --- DYNAMIC BACKEND DETECTION ---
try:
    import mlx_whisper

    STT_BACKEND = "mlx"
except ImportError:
    from faster_whisper import WhisperModel

    STT_BACKEND = "faster_whisper"


def get_model(model_size: str, device: str, compute_type: str):
    """Loads or prepares the model based on the active hardware backend."""
    if STT_BACKEND == "mlx":
        # MLX uses specific HuggingFace community repos for optimized Mac weights
        mlx_models = {
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
        }
        repo_id = mlx_models.get(model_size, f"mlx-community/whisper-{model_size}-mlx")
        logging.info(f"🍎 Using MLX backend. Mapped '{model_size}' to '{repo_id}'")
        return repo_id  # MLX just needs the repo string for transcription

    elif STT_BACKEND == "faster_whisper":
        logging.info("🟩 Using faster_whisper CUDA backend.")
        model_name = model_size.split('/')[-1].replace('faster-whisper-', '')
        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=2,
            num_workers=1,
        )


def _sync_transcribe(model_instance, audio_np: np.ndarray) -> str:
    """Executes the transcription based on the active backend."""
    if STT_BACKEND == "mlx":
        # MLX transcribe call (model_instance is the repo string)
        result = mlx_whisper.transcribe(audio_np, path_or_hf_repo=model_instance)
        return result.get("text", "").strip()

    elif STT_BACKEND == "faster_whisper":
        # Faster-Whisper transcribe call (model_instance is the WhisperModel object)
        segments, _ = model_instance.transcribe(audio_np, vad_filter=True)
        return "".join(segment.text for segment in segments).strip()


async def process_audio_chunk(model_instance, audio_chunk: bytes) -> str:
    if not model_instance:
        logging.error("Model is not loaded.")
        return ""
    try:
        logging.debug(f"Processing audio chunk len = {len(audio_chunk)}.")
        audio_np = np.frombuffer(audio_chunk, dtype=np.float32)

        # Offload the blocking CPU work to a separate thread
        async with transcription_semaphore:
            transcription = await asyncio.to_thread(_sync_transcribe, model_instance, audio_np)

        logging.debug(f"Transcription: {transcription}")
        return transcription

    except Exception as e:
        logging.error(f"Error in audio processing: {e}")
        return ""
import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel

MAX_CONCURRENT_TRANSCRIPTIONS = 2
transcription_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# Create a synchronous helper to handle the actual transcription
def _sync_transcribe(model: WhisperModel, audio_np: np.ndarray) -> str:
    segments, _ = model.transcribe(audio_np, vad_filter=True)

    return "".join(segment.text for segment in segments).strip()


async def process_audio_chunk(model: WhisperModel, audio_chunk: bytes) -> str:
    if not model:
        logging.error("Model is not loaded.")
        return ""
    try:
        logging.debug(f"Processing audio chunk len = {len(audio_chunk)}.")
        audio_np = np.frombuffer(audio_chunk, dtype=np.float32)

        # Offload the blocking CPU work to a separate thread
        async with transcription_semaphore:
            transcription = await asyncio.to_thread(_sync_transcribe, model, audio_np)

        logging.debug(f"Transcription: {transcription}")
        return transcription

    except Exception as e:
        logging.error(f"Error in audio processing: {e}")
        return ""
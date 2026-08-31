import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, status
from . import config
from .audio_processor import process_audio_chunk, get_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to hold the Whisper model instance
model = None


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    """
    Handles application startup and shutdown logic using lifespan context manager.
    """
    global model

    try:
        logging.info("Initializing application...")

        logging.info(
            f"Preparing model: {config.MODEL_SIZE} on device: {config.DEVICE} "
            f"with compute_type: {config.COMPUTE_TYPE}"
        )
        try:
            # The processor now handles whether this returns a WhisperModel or an MLX string
            model = get_model(
                config.MODEL_SIZE,
                config.DEVICE,
                config.COMPUTE_TYPE,
                cpu_threads=config.CPU_THREADS,
                num_workers=config.NUM_WORKERS,
            )
            logging.info("Model loaded and ready.")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            raise

        yield  # Start application after successful initialization

    finally:
        # Cleanup code can be added here if needed
        pass


# Register the lifespan handler
app = FastAPI(lifespan = app_lifespan)

@app.get("/health")
def health_check(response: Response):
    """
    Checks if the application is healthy.
    """
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "reason": "STT model not initialized"}
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles WebSocket connections for real-time audio transcription.
    It accepts a connection, receives audio chunks, processes them,
    and sends back transcriptions.
    """
    if not model:
        logging.error("Rejecting connection: STT model not initialized.")
        await websocket.accept()
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Model Not Ready")
        return

    await websocket.accept()
    logging.info("WebSocket connection established.")

    #last_processed_chunk = None
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()

            # Basic debouncing to avoid processing the same chunk multiple times
            # ** Likely no longer needed! **
            # if audio_chunk == last_processed_chunk:
            #     continue
            # last_processed_chunk = audio_chunk

            # Process the audio chunk using the globally loaded model
            transcription = await process_audio_chunk(model, audio_chunk)

            # Send the transcription result back to the client
            if transcription:
                response_data = {
                    "type": "transcription",
                    "source": "user",
                    "data": transcription,
                }
                await websocket.send_text(json.dumps(response_data))


    except WebSocketDisconnect:
        logging.info("WebSocket connection closed cleanly by client.")
    except Exception as e:
        logging.error(f"An error occurred in the WebSocket handler: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
        logging.info("WebSocket cleanup complete. Ready for new connections.")

if __name__ == "__main__":
    """
    Allows running the app directly with `python -m src.app` for development.
    """
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
import os

# --- Model Configuration ---
# These environment variables control which model is loaded and how it's run.
MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.environ.get("STT_CPU_THREADS", "2"))
NUM_WORKERS = int(os.environ.get("STT_NUM_WORKERS", "1"))
HOST = os.environ.get("UVICORN_HOST", "localhost" )
PORT = int(os.environ.get( "UVICORN_PORT", 80))

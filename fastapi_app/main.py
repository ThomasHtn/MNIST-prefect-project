from pathlib import Path

from correct import save_correction
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from predict import predict_digit

# Setup log file path (optional, central logging)
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
logger.add(LOGS_DIR / "api.log", rotation="1 day", enqueue=True)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    logger.info("📤 /predict called")
    result = predict_digit(file)
    logger.info(f"🎯 Prediction result: {result}")
    return result


@app.post("/correct")
async def correct_api(file: UploadFile = File(...), label: int = Form(...)):
    logger.info(f"📤 /correct called with label={label}")
    result = save_correction(file, label)
    logger.info("✅ Correction saved successfully")
    return result


@app.get("/health")
def health():
    model_path = BASE_DIR.parent / "models" / "latest_model.h5"
    model_exists = model_path.exists()
    logger.info(f"🔍 Health check: model exists = {model_exists}")
    return {"status": "ok", "model_loaded": model_exists}

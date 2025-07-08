import os

from correct import save_correction
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_digit

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    return predict_digit(file)


@app.post("/correct")
async def correct_api(file: UploadFile = File(...), label: int = Form(...)):
    result = save_correction(file, label)
    print(result)
    return result


@app.get("/health")
def health():
    model_exists = os.path.exists("../models/latest_model.h5")
    return {"status": "ok", "model_loaded": model_exists}

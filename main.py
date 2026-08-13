from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn as nn

import numpy as np
import cv2
import shutil
import os
import mediapipe as mp

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "model/sign_model.pt"

LABELS = [
    "hello",
    "please",
    "yes",
    "thank you",
    "sorry",
    "no",
    "I Love You",
    "help",
    "good",
    "bye"
]

INPUT_SIZE = 63

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# =====================================================
# MODEL
# =====================================================

class SignModel(nn.Module):

    def __init__(self, input_size, num_classes):

        super(SignModel, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# =====================================================
# LOAD MODEL
# =====================================================

model = SignModel(INPUT_SIZE, len(LABELS))

try:

    if os.path.exists(MODEL_PATH):

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=torch.device("cpu")
            )
        )

        model.eval()

        print("Model loaded successfully")

    else:

        print("Model file not found")

except Exception as e:

    print(f"Model loading error : {e}")

# =====================================================
# MEDIAPIPE (OPTIMIZED)
# =====================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=False,
    max_num_hands=1,

    # IMPORTANT FOR SPEED
    model_complexity=0,

    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# =====================================================
# PREPROCESS
# =====================================================

def preprocess(image_path):

    try:

        img = cv2.imread(image_path)

        if img is None:
            return None

        # =========================================
        # VERY IMPORTANT FOR SPEED
        # =========================================

        img = cv2.resize(img, (160, 120))

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        landmarks = []

        for lm in results.multi_hand_landmarks[0].landmark:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        return np.array(
            landmarks,
            dtype=np.float32
        )

    except Exception as e:

        print(f"Preprocess error : {e}")

        return None

# =====================================================
# PREDICTION
# =====================================================

def predict_sign(image_path):

    data = preprocess(image_path)

    if data is None:
        return "No hand detected"

    try:

        data = np.expand_dims(data, axis=0)

        tensor = torch.from_numpy(data)

        with torch.no_grad():

            output = model(tensor)

        pred = torch.argmax(
            output,
            dim=1
        ).item()

        return LABELS[pred]

    except Exception as e:

        print(f"Prediction error : {e}")

        return "Prediction failed"

# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Kaitexy AI Backend Running"
    }

# =====================================================
# PREDICT SIGN
# =====================================================

@app.post("/predict-sign")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # =========================================
        # SAVE IMAGE
        # =========================================

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # =========================================
        # PREDICT
        # =========================================

        result = predict_sign(file_path)

        # =========================================
        # DELETE IMAGE AFTER USE
        # =========================================

        if os.path.exists(file_path):
            os.remove(file_path)

        return JSONResponse({
            "prediction": result
        })

    except Exception as e:

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=500
        )

# =====================================================
# TEXT TO SIGN
# =====================================================

@app.get("/text-to-sign")
def text_to_sign(text: str):

    clean_text = (
        text.lower()
        .strip()
        .replace(" ", "")
    )

    image_path = (
        f"static/signs/{clean_text}.png"
    )

    if not os.path.exists(image_path):

        return JSONResponse(
            {
                "error":
                f"Sign for '{text}' not found"
            },
            status_code=404
        )

    return {
        "image_url":
        f"/static/signs/{clean_text}.png"
    }
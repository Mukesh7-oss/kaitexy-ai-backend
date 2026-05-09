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

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model/sign_model.pt"

LABELS = [
    "hello", "please", "yes", "thank you", "sorry",
    "no", "I Love You", "help", "good", "bye"
]

INPUT_SIZE = 63

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI()

# Serve static files (images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# MODEL
# -----------------------------
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

model = SignModel(INPUT_SIZE, len(LABELS))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(" Model loaded successfully")
else:
    print(" Model file not found")

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_sign(image_path):
    data = preprocess(image_path)
    if data is None:
        return "No hand detected"

    data = np.expand_dims(data, axis=0)
    tensor = torch.from_numpy(data)

    with torch.no_grad():
        output = model(tensor)

    pred = torch.argmax(output, dim=1).item()
    return LABELS[pred]

# -----------------------------
# ROUTES
# -----------------------------

# Health check
@app.get("/")
def home():
    return {
        "message": "Kaitexy AI Backend Running "
    }

# Predict sign from image
@app.post("/predict-sign")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_sign(file_path)

    return JSONResponse({
        "prediction": result
    })

# Text to sign image
@app.get("/text-to-sign")
def text_to_sign(text: str):
    clean_text = text.lower().strip().replace(" ", "")
    image_path = f"static/signs/{clean_text}.png"

    if not os.path.exists(image_path):
        return JSONResponse(
            {"error": f"Sign for '{text}' not found"},
            status_code=404
        )

    return {
        "image_url": f"/static/signs/{clean_text}.png"
    }
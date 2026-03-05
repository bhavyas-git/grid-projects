# Install packages if needed

import gradio as gr
import torch
import timm
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------- DEVICE ---------------- #

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=7
)

model.load_state_dict(
    torch.load("../notebooks/checkpoints/vit_small_best.pth", map_location=device)
)

model.to(device)
model.eval()

# Emotion labels
emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# ---------------- FACE DETECTOR ---------------- #

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- TRANSFORM ---------------- #

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- PREDICTION FUNCTION ---------------- #

def predict_emotion(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected", None, "No fairness metrics"

    probs_all = []

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        face_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        probs_all.append(probs)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    probs_all = np.array(probs_all)

    avg_probs = probs_all.mean(axis=0)

    predicted = emotions[np.argmax(avg_probs)]

    # -------- Fairness metric -------- #

    confidence = probs_all.max(axis=1)
    fairness_score = confidence.std()

    fairness_report = f"""
Faces detected: {len(probs_all)}

Confidence std deviation: {fairness_score:.4f}

Lower value = more consistent predictions
"""

    # -------- Probability Chart -------- #

    fig = plt.figure()
    plt.bar(emotions, avg_probs)
    plt.xticks(rotation=45)
    plt.title("Emotion Probabilities")

    return predicted, fig, fairness_report


# ---------------- GRADIO DASHBOARD ---------------- #

with gr.Blocks() as demo:

    gr.Markdown("# Emotion Recognition Dashboard")

    with gr.Row():
        webcam = gr.Image(sources="webcam", type="numpy")
        label = gr.Textbox(label="Predicted Emotion")

    predict_btn = gr.Button("Predict Emotion")
    chart = gr.Plot(label="Emotion Probabilities")
    fairness = gr.Textbox(label="Fairness Check")

    predict_btn.click(
        predict_emotion,
        inputs=webcam,
        outputs=[label, chart, fairness]
    )

demo.launch()

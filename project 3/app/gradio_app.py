import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# EMOTION LABELS
# -----------------------------
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# -----------------------------
# LOAD MODEL
# -----------------------------

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 7)
    )
)

model.load_state_dict(
    torch.load("../notebooks/checkpoints/efficient_net_final_best.pth", map_location=device)
)

model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_emotion(image):

    if image is None:
        return "No image", None, "No fairness check"

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        face = img
    else:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]

    face = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(face)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_class = emotion_labels[np.argmax(probs)]

    # -----------------------------
    # PLOT PROBABILITIES
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(emotion_labels, probs)
    ax.set_title("Emotion Probabilities")
    ax.set_ylim([0,1])
    plt.xticks(rotation=30)
    plt.tight_layout()

    # -----------------------------
    # FAIRNESS CHECK
    # -----------------------------
    confidence = np.max(probs)

    if confidence < 0.40:
        fairness = "⚠️ Low confidence prediction — model uncertain"
    else:
        fairness = "✅ Prediction confidence acceptable"

    return predicted_class, fig, fairness


# -----------------------------
# DASHBOARD UI
# -----------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    h1 {text-align:center}
    .gradio-container {max-width:1000px;margin:auto}
    """
) as demo:

    gr.Markdown("# Emotion Recognition Dashboard")
    gr.Markdown(
        "Capture a photo using your webcam or upload an image to detect facial emotions."
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                sources=["webcam","upload"],
                type="numpy",
                label="Capture or Upload Image"
            )

            predict_button = gr.Button(
                "Predict Emotion",
                variant="primary"
            )

        with gr.Column():

            prediction = gr.Textbox(
                label="Predicted Emotion"
            )

            probability_chart = gr.Plot(
                label="Emotion Probabilities"
            )

            fairness_box = gr.Textbox(
                label="Fairness Check"
            )

    predict_button.click(
        predict_emotion,
        inputs=image_input,
        outputs=[prediction, probability_chart, fairness_box]
    )

demo.launch()

import gradio as gr
import os
import socket
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
from collections import deque, defaultdict
from huggingface_hub import hf_hub_download
from src.models.mobilenetV3 import get_mobilenet_v3

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# EMOTION LABELS
# -----------------------------
emotion_labels = [
    "Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
model = get_mobilenet_v3(num_classes=7, pretrained=False)

def resolve_checkpoint_path():
    local_path = PROJECT_ROOT / "results" / "checkpoints" / "mobilenet_best.pth"
    if local_path.exists():
        return local_path

    repo_id = os.getenv("HF_MODEL_REPO_ID")
    filename = os.getenv("HF_MODEL_FILE", "mobilenet_best.pth")
    token = os.getenv("HF_TOKEN")

    if not repo_id:
        raise FileNotFoundError(
            "Checkpoint not found locally. Set HF_MODEL_REPO_ID to download it from Hugging Face Hub."
        )

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
    )
    return Path(downloaded_path)

checkpoint_path = resolve_checkpoint_path()
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

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
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
])

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# DEMOGRAPHIC DETECTOR (PLACEHOLDER)
# Replace with DeepFace / FairFace if needed
# -----------------------------
def detect_demographics(face):
    # MOCK DEMOGRAPHICS (replace later)
    age_group = np.random.choice(["Young", "Adult", "Senior"])
    gender = np.random.choice(["Male", "Female"])
    return age_group, gender

# -----------------------------
# FAIRNESS STORAGE (Sliding Window)
# -----------------------------
WINDOW_SIZE = 100
UNKNOWN_LABEL = "Unknown / unlabeled"

history = deque(maxlen=WINDOW_SIZE)

# -----------------------------
# LOG FILE
# -----------------------------
LOG_FILE = APP_DIR / "fairness_log.csv"

def log_prediction(entry):
    df = pd.DataFrame([entry])
    try:
        df.to_csv(LOG_FILE, mode='a', header=not pd.io.common.file_exists(LOG_FILE), index=False)
    except:
        pass

def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def create_blank_image(size=(224, 224)):
    return np.zeros((size[0], size[1], 3), dtype=np.uint8)

def resolve_server_port():
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def detect_primary_face(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, False

    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w], True

def predict_emotion(face):
    face_tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(face_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = emotion_labels[pred_idx]
    confidence = float(probs[pred_idx])
    return probs, pred_class, confidence

def build_emotion_plot(probs):
    fig, ax = plt.subplots()
    ax.bar(emotion_labels, probs)
    ax.set_title("Emotion Probabilities")
    ax.set_ylim([0, 1])
    plt.xticks(rotation=30)
    return fig_to_image(fig)

def build_heatmap(records):
    df = pd.DataFrame(records)
    if df.empty:
        return create_blank_image()

    pivot = pd.crosstab(df["group"], df["pred"]).reindex(columns=emotion_labels, fill_value=0)

    fig, ax = plt.subplots()
    im = ax.imshow(pivot, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Fairness Heatmap (Group vs Emotion)")
    plt.colorbar(im)
    return fig_to_image(fig)

def compute_fairness_metrics(records):
    labeled = [
        record for record in records
        if record.get("reference") in emotion_labels
    ]

    if not labeled:
        return {}, "Add a reference label to compute TPR/FPR fairness gaps."

    per_group = {}
    for group in sorted({record["group"] for record in labeled}):
        group_records = [record for record in labeled if record["group"] == group]
        tpr_scores = []
        fpr_scores = []

        for emotion in emotion_labels:
            tp = fp = tn = fn = 0
            for record in group_records:
                actual_positive = record["reference"] == emotion
                predicted_positive = record["pred"] == emotion

                if actual_positive and predicted_positive:
                    tp += 1
                elif actual_positive and not predicted_positive:
                    fn += 1
                elif not actual_positive and predicted_positive:
                    fp += 1
                else:
                    tn += 1

            if tp + fn > 0:
                tpr_scores.append(tp / (tp + fn))
            if fp + tn > 0:
                fpr_scores.append(fp / (fp + tn))

        per_group[group] = {
            "frames": len(group_records),
            "tpr": float(np.mean(tpr_scores)) if tpr_scores else 0.0,
            "fpr": float(np.mean(fpr_scores)) if fpr_scores else 0.0,
        }

    tpr_values = [values["tpr"] for values in per_group.values()]
    fpr_values = [values["fpr"] for values in per_group.values()]
    tpr_gap = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0.0
    fpr_gap = max(fpr_values) - min(fpr_values) if len(fpr_values) > 1 else 0.0

    lines = [f"Sliding window: {len(labeled)} labeled frames"]
    for group, values in per_group.items():
        lines.append(
            f"{group}: TPR={values['tpr']:.2f}, FPR={values['fpr']:.2f}, n={values['frames']}"
        )
    lines.append(f"TPR gap={tpr_gap:.2f} | FPR gap={fpr_gap:.2f}")
    return per_group, "\n".join(lines)

def build_fairness_alert(per_group, threshold=0.05):
    if len(per_group) < 2:
        return "Need labeled frames from at least 2 demographic groups for fairness comparison."

    tpr_values = [values["tpr"] for values in per_group.values()]
    fpr_values = [values["fpr"] for values in per_group.values()]
    tpr_gap = max(tpr_values) - min(tpr_values)
    fpr_gap = max(fpr_values) - min(fpr_values)
    max_gap = max(tpr_gap, fpr_gap)

    if max_gap > threshold:
        return f"Bias alert: max fairness gap={max_gap:.2f} exceeds {threshold:.0%} threshold"
    return f"Fairness within threshold: max gap={max_gap:.2f}"

def evaluate_image(image, reference_label, update_global_history=True, source="webcam"):
    if image is None or not isinstance(image, np.ndarray):
        blank = create_blank_image()
        return blank, blank, "No input", "No data", "No labeled data"

    face, face_detected = detect_primary_face(image)
    age_group, gender = detect_demographics(face)
    group = f"{age_group}-{gender}"
    probs, pred_class, confidence = predict_emotion(face)

    entry = {
        "timestamp": time.time(),
        "group": group,
        "pred": pred_class,
        "conf": confidence,
        "reference": reference_label,
        "face_detected": face_detected,
        "source": source,
    }

    if update_global_history:
        history.append(entry)
        log_prediction({
            "timestamp": entry["timestamp"],
            "group": group,
            "prediction": pred_class,
            "confidence": confidence,
            "reference": reference_label,
            "face_detected": face_detected,
            "source": source,
        })

    records = list(history) if update_global_history else [entry]
    per_group, fairness_summary = compute_fairness_metrics(records)
    fairness_alert = build_fairness_alert(per_group) if per_group else "No fairness alert until labeled frames are available."

    info = (
        f"{pred_class} ({confidence:.2f}) | Group: {group} | "
        f"Face detected: {'Yes' if face_detected else 'No'}"
    )

    return (
        build_emotion_plot(probs),
        build_heatmap(records),
        info,
        fairness_alert,
        fairness_summary,
    )

def summarize_video_records(records, expected_label):
    if not records:
        blank = create_blank_image()
        return blank, "No frames processed.", "No labeled data"

    confidences = [record["conf"] for record in records]
    brightness_values = [record["brightness"] for record in records]
    blur_values = [record["blur"] for record in records]
    face_rate = sum(record["face_detected"] for record in records) / len(records)
    per_group, fairness_summary = compute_fairness_metrics(records)
    fairness_alert = build_fairness_alert(per_group) if per_group else "No fairness alert until labeled frames are available."

    summary = "\n".join([
        f"Frames analyzed: {len(records)}",
        f"Expected emotion: {expected_label}",
        f"Mean confidence: {np.mean(confidences):.2f}",
        f"Face detection rate: {face_rate:.2%}",
        f"Brightness range: {min(brightness_values):.1f} to {max(brightness_values):.1f}",
        f"Blur range (Laplacian var): {min(blur_values):.1f} to {max(blur_values):.1f}",
        fairness_alert,
    ])
    return build_heatmap(records), summary, fairness_summary

# -----------------------------
def process_frame(image, reference_label):
    return evaluate_image(image, reference_label, update_global_history=True, source="webcam")

def analyze_video(video_path, expected_label):
    if not video_path:
        blank = create_blank_image()
        return blank, "No video provided.", "No labeled data"

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        blank = create_blank_image()
        return blank, "Unable to open the uploaded video.", "No labeled data"

    records = []
    frame_index = 0
    sample_stride = 10

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % sample_stride == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face, face_detected = detect_primary_face(rgb_frame)
            age_group, gender = detect_demographics(face)
            group = f"{age_group}-{gender}"
            _, pred_class, confidence = predict_emotion(face)

            records.append({
                "timestamp": time.time(),
                "group": group,
                "pred": pred_class,
                "conf": confidence,
                "reference": expected_label,
                "face_detected": face_detected,
                "brightness": float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))),
                "blur": float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
                "source": "video",
            })

            log_prediction({
                "timestamp": records[-1]["timestamp"],
                "group": group,
                "prediction": pred_class,
                "confidence": confidence,
                "reference": expected_label,
                "face_detected": face_detected,
                "source": "video",
            })

        frame_index += 1
        if len(records) >= WINDOW_SIZE:
            break

    capture.release()
    return summarize_video_records(records, expected_label)


# -----------------------------
# GRADIO UI (REAL-TIME)
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 🎥 Real-Time Emotion + Fairness Dashboard")
    gr.Markdown(
        "Use the optional reference label when you know the true emotion. "
        "That enables the 100-frame TPR/FPR fairness tracking required for auditing."
    )

    with gr.Tab("Webcam Monitor"):
        with gr.Row():
            webcam = gr.Image(sources=["webcam"], type="numpy")

            with gr.Column():
                reference_label = gr.Dropdown(
                    choices=[UNKNOWN_LABEL] + emotion_labels,
                    value=UNKNOWN_LABEL,
                    label="Reference Emotion"
                )
                emotion_plot = gr.Image(label="Emotion Distribution")
                heatmap_plot = gr.Image(label="Fairness Heatmap")
                info_box = gr.Textbox(label="Prediction Info")
                alert_box = gr.Textbox(label="Fairness Alert")
                metrics_box = gr.Textbox(label="Sliding-Window Fairness Metrics", lines=8)

        predict_btn = gr.Button("Start Inference")
        predict_btn.click(
            process_frame,
            inputs=[webcam, reference_label],
            outputs=[emotion_plot, heatmap_plot, info_box, alert_box, metrics_box],
            api_name=False
        )

    with gr.Tab("Video Audit"):
        gr.Markdown(
            "Upload a video with lighting, angle, or occlusion changes to measure robustness "
            "and demographic fairness over sampled frames."
        )
        with gr.Row():
            video_input = gr.Video(label="Audit Video")
            with gr.Column():
                expected_label = gr.Dropdown(
                    choices=[UNKNOWN_LABEL] + emotion_labels,
                    value=UNKNOWN_LABEL,
                    label="Expected Emotion"
                )
                video_heatmap = gr.Image(label="Video Fairness Heatmap")
                video_summary = gr.Textbox(label="Robustness Summary", lines=8)
                video_metrics = gr.Textbox(label="Video Fairness Metrics", lines=8)

        audit_btn = gr.Button("Run Video Audit")
        audit_btn.click(
            analyze_video,
            inputs=[video_input, expected_label],
            outputs=[video_heatmap, video_summary, video_metrics],
            api_name=False
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=resolve_server_port()
    )

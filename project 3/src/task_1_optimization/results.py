import coremltools as ct
import numpy as np

# =============================
# Load models
# =============================
model_fp32 = ct.models.MLModel("../../results/model.mlmodel")
model_int8 = ct.models.MLModel("../../results/model_quantized.mlmodel")

input_name = model_fp32.get_spec().description.input[0].name

# =============================
# Input
# =============================
x = np.random.rand(1, 3, 224, 224).astype(np.float32)

# =============================
# Helper functions
# =============================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def process_output(output):
    key = list(output.keys())[0]
    logits = output[key]
    probs = softmax(logits)
    pred = np.argmax(probs)
    return logits, probs, pred

# =============================
# FP32 prediction
# =============================
out_fp32 = model_fp32.predict({input_name: x})
logits_fp32, probs_fp32, pred_fp32 = process_output(out_fp32)

# =============================
# INT8 prediction
# =============================
out_int8 = model_int8.predict({input_name: x})
logits_int8, probs_int8, pred_int8 = process_output(out_int8)

# =============================
# Print comparison
# =============================
def print_probs(title, probs):
    print(f"\n{title}")
    for i, p in enumerate(probs[0]):
        print(f"{class_names[i]:<10}: {p:.4f}")

# =============================
# ONNX output
# =============================
import onnxruntime as ort

session = ort.InferenceSession("../../results/model_fp32.onnx")
input_name_onnx = session.get_inputs()[0].name

onnx_out = session.run(None, {input_name_onnx: x})[0]
onnx_probs = softmax(onnx_out)
onnx_pred = np.argmax(onnx_probs)

# =============================
# PRINT EVERYTHING CLEANLY
# =============================

print_probs("🔵 CoreML FP32", probs_fp32)
print_probs("🟢 CoreML INT8", probs_int8)
print_probs("🟡 ONNX", onnx_probs)

# Difference
diff = np.abs(probs_fp32 - probs_int8)
print(f"\n📊 Max difference: {diff.max():.4f}")

# =============================
# Difference
# =============================
diff = np.abs(probs_fp32 - probs_int8)

print("\n📊 Difference (|FP32 - INT8|):")
print(diff)
print("Max difference:", diff.max())

import onnxruntime as ort

session = ort.InferenceSession("../../results/model_fp32.onnx")

input_name = session.get_inputs()[0].name

onnx_out = session.run(None, {input_name: x})[0]

onnx_probs = softmax(onnx_out)
onnx_pred = np.argmax(onnx_probs)
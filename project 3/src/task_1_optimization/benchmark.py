# optimization/benchmark.py

import time
import torch
import numpy as np
from pathlib import Path
import coremltools as ct

def get_model_size(path):
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    return size_mb


def measure_latency(model, dataloader, device, runs=50):
    model.eval()

    latencies = []

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= runs:
                break

            images = images.to(device)

            start = time.time()
            _ = model(images)
            end = time.time()

            latencies.append((end - start) * 1000)  # ms

    return np.mean(latencies), np.percentile(latencies, 95)


def evaluate_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def benchmark_pytorch(model, val_loader, device):
    import time
    import numpy as np
    import torch

    model.eval()

    correct = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()
            outputs = model(images)
            end = time.time()

            latencies.append((end - start) * 1000)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    latencies = np.array(latencies)

    mean_latency = latencies.mean()
    p95_latency = np.percentile(latencies, 95)

    print("\n📊 PyTorch Benchmark")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")

    # ✅ ADD THIS RETURN
    return {
        "accuracy": accuracy,
        "mean_latency": mean_latency,
        "p95_latency": p95_latency
    }

def benchmark_coreml(model_path, val_loader, input_shape=(1, 3, 224, 224), runs=100):
    """
    Benchmarks a CoreML model (latency + accuracy)

    Args:
        model_path (str): path to CoreML model
        val_loader: PyTorch dataloader for accuracy
        input_shape (tuple): dummy input shape
        runs (int): number of latency runs

    Returns:
        dict: latency + accuracy
    """

    print(f"\n📊 CoreML Benchmark → {model_path}")

    # =============================
    # Load model
    # =============================
    model = ct.models.MLModel(model_path)

    input_name = model.get_spec().description.input[0].name

    # =============================
    # Latency Benchmark
    # =============================
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = model.predict({input_name: input_data})

    latencies = []

    for _ in range(runs):
        start = time.time()
        _ = model.predict({input_name: input_data})
        end = time.time()

        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    # =============================
    # Accuracy Evaluation
    # =============================
    correct = 0
    total = 0

    for images, labels in val_loader:
        images_np = images.numpy().astype(np.float32)

        for i in range(images_np.shape[0]):
            inp = images_np[i:i+1]

            output = model.predict({input_name: inp})

            out_key = list(output.keys())[0]
            preds = output[out_key]

            pred_class = np.argmax(preds)
            true_class = labels[i].item()

            if pred_class == true_class:
                correct += 1

            total += 1

    accuracy = 100 * correct / total

    # =============================
    # Print results
    # =============================
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")

    return {
        "accuracy": accuracy,
        "mean_latency": mean_latency,
        "p95_latency": p95_latency
    }

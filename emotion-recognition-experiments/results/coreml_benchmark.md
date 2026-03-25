# Model Optimization & Benchmark Report

## Overview
This report compares performance and size across model formats:
- PyTorch (FP32)
- ONNX (FP32)
- CoreML (FP32)
- CoreML (INT8 Quantized)

## Performance Comparison

| Model | Mean Latency (ms) | P95 Latency (ms) | Accuracy (%) |
|------|------------------|------------------|--------------|
| PyTorch (FP32) | 7.16 | 7.81 | 67.26 |
| CoreML (FP32) | 1.79 | 2.00 | 67.40 |
| CoreML (INT8) | 3.55 | 4.31 | 66.75 |

## Model Size Comparison

| Model | Size (MB) |
|------|-----------|
| PyTorch (FP32) | 6.92 |
| ONNX (FP32) | 6.81 |
| CoreML (FP32) | 6.79 |
| CoreML (INT8) | 1.78 |

## Analysis

- CoreML (FP32) achieves a **3.99× speedup** over PyTorch.
- CoreML (INT8) achieves a **2.01× speedup** over PyTorch.
- INT8 quantization reduces model size by **3.80×**.

- Accuracy remains stable across optimizations, indicating minimal degradation.
- CoreML leverages Apple’s Neural Engine for fast on-device inference.

## Conclusion

Optimization techniques including pruning and quantization significantly improve latency and reduce model size while maintaining comparable accuracy. CoreML deployment enables efficient real-time inference on Apple devices.

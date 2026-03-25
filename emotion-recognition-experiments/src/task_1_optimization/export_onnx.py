# optimization/export_onnx.py

import torch


def export_to_onnx(model, save_path="model.onnx", input_size=(1, 3, 224, 224)):
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"✅ ONNX model saved at: {save_path}")
# optimization/convert_coreml.py

import coremltools as ct


def convert_to_coreml(onnx_path, save_path="model.mlmodel"):
    # Convert ONNX → CoreML
    model = ct.convert(
        onnx_path,
        source="onnx",
        compute_units=ct.ComputeUnit.ALL  # uses Neural Engine
    )

    # Set compute units (Neural Engine)
    model.compute_units = ct.ComputeUnit.ALL  # includes NE

    model.save(save_path)
    print(f"✅ CoreML model saved at: {save_path}")


def quantize_coreml_model(mlmodel_path, save_path="model_quantized.mlmodel"):
    model = ct.models.MLModel(mlmodel_path)

    # Apply INT8 quantization
    quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
        model,
        nbits=8
    )

    quantized_model.save(save_path)
    print(f"✅ Quantized CoreML model saved at: {save_path}")
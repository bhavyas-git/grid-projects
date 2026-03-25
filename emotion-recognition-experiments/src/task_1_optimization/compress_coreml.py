import coremltools as ct
import shutil
import os

def compress_coreml_model(input_path, output_path):
    """
    Robust CoreML compression that works across different coremltools versions.

    Supports:
    - .mlmodel (NeuralNetwork)
    - .mlpackage (ML Program)

    Falls back gracefully if compression is not supported.
    """

    print("🔧 Loading CoreML model...")

    model = ct.models.MLModel(input_path)

    # =============================
    # Try compression (INT8)
    # =============================
    try:
        print("⚙️ Attempting INT8 compression...")

        from coremltools.models.neural_network.quantization_utils import quantize_weights

        compressed_model = quantize_weights(
            model,
            nbits=8
        )

        compressed_model.save(output_path)
        print(f"✅ Compression successful → {output_path}")
        return

    except Exception as e:
        print("⚠️ Compression not supported for this model type.")
        print(f"Reason: {e}")

    # =============================
    # Fallback (copy model)
    # =============================
    print("📦 Falling back to model copy (no compression)...")

    if os.path.isdir(input_path):  # .mlpackage
        shutil.copytree(input_path, output_path, dirs_exist_ok=True)
    else:  # .mlmodel
        shutil.copy2(input_path, output_path)

    print(f"✅ Fallback complete → {output_path}")
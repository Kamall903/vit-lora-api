
import json
from pathlib import Path

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    ViTForImageClassification,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "vit_quantized_int8.pt"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
ASSETS_PATH = BASE_DIR / "models" / "model_assets"

def load_model():
    print("Loading model metadata...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Quantized model not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata["model_name"]
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]

    if len(class_names) != num_classes:
        raise ValueError("Class metadata is inconsistent")

    if metadata["quantization"] != "dynamic_int8":
        raise ValueError("Unsupported quantization type")

    id2label = {
        i: name for i, name in enumerate(class_names)
    }

    label2id = {
        name: i for i, name in enumerate(class_names)
    }

    # Reconstruct the ViT architecture without downloading
    # the pretrained weights again.
    config = AutoConfig.from_pretrained(
    ASSETS_PATH,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True
)
    model = ViTForImageClassification(config)
    model.eval()

    print("Applying dynamic INT8 quantization...")

    model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )

    print("Loading quantized checkpoint...")

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    # All expected model weights must be present.
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    processor = AutoImageProcessor.from_pretrained(
    ASSETS_PATH,
    local_files_only=True
)

    print("Quantized ViT loaded successfully!")

    return model, processor, class_names
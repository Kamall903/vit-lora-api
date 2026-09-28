
import json
import os
import time

import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import ViTForImageClassification

from training.dataset import get_dataloaders
from training.lora import (
    apply_lora,
    get_vit_blocks,
    get_attention,
    load_lora_state_dict,
)


CHECKPOINT_PATH = "models/lora_best.pt"
MERGED_PATH = "models/vit_merged_fp32.pt"
QUANTIZED_PATH = "models/vit_quantized_int8.pt"
METADATA_PATH = "models/model_metadata.json"

DEVICE = torch.device("cpu")

# Use a limited number of images for initial benchmarking.
BENCHMARK_BATCHES = 10


# ============================================================
# 1. LOAD THE BEST LORA MODEL
# ============================================================

def load_best_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = ViTForImageClassification.from_pretrained(
        checkpoint["model_name"],
        num_labels=checkpoint["num_classes"],
        id2label={
            i: name for i, name in enumerate(class_names)
        },
        label2id={
            name: i for i, name in enumerate(class_names)
        },
        ignore_mismatched_sizes=True
    )

    model = apply_lora(
        model,
        rank=checkpoint["lora_rank"],
        alpha=checkpoint["lora_alpha"],
        dropout=checkpoint["lora_dropout"],
        target_blocks=checkpoint["target_blocks"]
    )

    load_lora_state_dict(
        model,
        checkpoint["model_state_dict"]
    )

    model.eval()
    return model, checkpoint


# ============================================================
# 2. MERGE LORA INTO PRETRAINED WEIGHTS
# ============================================================

@torch.no_grad()
def merge_lora(model, target_blocks):
    blocks = get_vit_blocks(model)

    for block_idx in target_blocks:
        attention, query_name, value_name = (
            get_attention(blocks[block_idx])
        )

        for name in (query_name, value_name):
            lora_layer = getattr(attention, name)

            original = lora_layer.original_layer

            # Delta W = (alpha / rank) * B @ A
            delta_weight = (
                lora_layer.lora_B @ lora_layer.lora_A
            ) * lora_layer.scaling

            original.weight.add_(delta_weight)

            # Replace LoRA wrapper with merged linear layer.
            setattr(attention, name, original)

        print(f"Merged LoRA block {block_idx}")

    model.eval()
    return model


# ============================================================
# 3. MEASURE MODEL SIZE
# ============================================================

def file_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# ============================================================
# 4. EVALUATE ACCURACY
# ============================================================

@torch.inference_mode()
def evaluate(model, loader, max_batches=None):
    model.eval()

    correct = 0
    total = 0

    for index, batch in enumerate(tqdm(
        loader,
        desc="Evaluating"
    )):
        if max_batches is not None and index >= max_batches:
            break

        images = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=images)
        predictions = outputs.logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ============================================================
# 5. MEASURE INFERENCE LATENCY
# ============================================================

@torch.inference_mode()
def benchmark(model, loader, max_batches=10):
    model.eval()

    times = []

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break

        images = batch["pixel_values"].to(DEVICE)

        # Warm up before measuring.
        if index == 0:
            for _ in range(3):
                model(pixel_values=images)

        start = time.perf_counter()

        model(pixel_values=images)

        elapsed = time.perf_counter() - start

        times.append(elapsed / images.size(0))

    return sum(times) / len(times) * 1000


# ============================================================
# 6. MAIN QUANTIZATION PIPELINE
# ============================================================

def main():
    os.makedirs("models", exist_ok=True)

    print("\nLoading best LoRA checkpoint...")
    model, checkpoint = load_best_model()

    print("\nMerging LoRA weights...")
    model = merge_lora(
        model,
        checkpoint["target_blocks"]
    )

    # Save the merged FP32 model.
    torch.save(model.state_dict(), MERGED_PATH)

    print("\nLoading test dataset...")
    _, _, test_loader, _ = get_dataloaders()

    print("\nEvaluating original FP32 model...")
    fp32_accuracy = evaluate(model, test_loader)
    fp32_latency = benchmark(
        model, test_loader, BENCHMARK_BATCHES
    )

    print("\nApplying dynamic INT8 quantization...")

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False
    )

    quantized_model.eval()

    print("\nEvaluating quantized model...")
    int8_accuracy = evaluate(
        quantized_model, test_loader
    )
    int8_latency = benchmark(
        quantized_model,
        test_loader,
        BENCHMARK_BATCHES
    )

    # Save the quantized model's state dictionary.
    torch.save(
        quantized_model.state_dict(),
        QUANTIZED_PATH
    )

    fp32_size = file_size_mb(MERGED_PATH)
    int8_size = file_size_mb(QUANTIZED_PATH)

    metadata = {
        "model_name": checkpoint["model_name"],
        "num_classes": checkpoint["num_classes"],
        "class_names": checkpoint["class_names"],
        "quantization": "dynamic_int8",
        "fp32_accuracy": fp32_accuracy,
        "int8_accuracy": int8_accuracy,
        "fp32_latency_ms_per_image": fp32_latency,
        "int8_latency_ms_per_image": int8_latency,
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n========== QUANTIZATION RESULTS ==========")

    print(f"FP32 accuracy: {fp32_accuracy:.2%}")
    print(f"INT8 accuracy: {int8_accuracy:.2%}")

    print(f"FP32 size: {fp32_size:.2f} MB")
    print(f"INT8 size: {int8_size:.2f} MB")

    reduction = (1 - int8_size / fp32_size) * 100

    print(f"Size reduction: {reduction:.2f}%")

    print(f"FP32 latency: {fp32_latency:.2f} ms/image")
    print(f"INT8 latency: {int8_latency:.2f} ms/image")

    print("==========================================")

    print(f"\nSaved quantized model: {QUANTIZED_PATH}")


if __name__ == "__main__":
    main()
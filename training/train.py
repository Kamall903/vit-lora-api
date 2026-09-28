
import os
import random
import time

import numpy as np
import torch

from tqdm import tqdm
from transformers import ViTForImageClassification

from training.config import (
    MODEL_NAME,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    EPOCHS,
    LEARNING_RATE,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    SEED,
)

from training.dataset import get_dataloaders

from training.lora import (
    apply_lora,
    count_parameters,
    get_lora_state_dict,
    load_lora_state_dict,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"

PATIENCE = 4

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "lora_best.pt"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    epoch
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} - Training"
    )

    for batch in progress:
        images = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass.
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            outputs = model(
                pixel_values=images,
                labels=labels
            )

            loss = outputs.loss

        # Backpropagation with gradient scaling.
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters()
             if p.requires_grad),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        predictions = outputs.logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total:.4f}",
            accuracy=f"{100 * correct / total:.2f}%"
        )

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# 4. VALIDATION / TESTING
# ============================================================

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(
        loader,
        desc="Evaluating",
        leave=False
    ):
        images = batch["pixel_values"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["labels"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            outputs = model(
                pixel_values=images,
                labels=labels
            )

        predictions = outputs.logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += outputs.loss.item() * batch_size

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================

def main():
    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("=" * 55)
    print("ViT + LoRA TRAINING")
    print("=" * 55)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading EuroSAT dataset...")

    (
        train_loader,
        val_loader,
        test_loader,
        class_names
    ) = get_dataloaders()

    assert len(class_names) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(class_names)}"
    )

    print("Classes:", class_names)
    print("Training:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Testing:", len(test_loader.dataset))

    # --------------------------------------------------------
    # Load pretrained ViT
    # --------------------------------------------------------

    print("\nLoading pretrained ViT...")

    id2label = {
        i: name
        for i, name in enumerate(class_names)
    }

    label2id = {
        name: i
        for i, name in enumerate(class_names)
    }

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # --------------------------------------------------------
    # Apply LoRA
    # --------------------------------------------------------

    model = apply_lora(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )

    model = model.to(DEVICE)

    count_parameters(model)

    # --------------------------------------------------------
    # Optimizer and mixed precision
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        print(
            f"\n{'=' * 20} "
            f"EPOCH {epoch}/{EPOCHS} "
            f"{'=' * 20}"
        )

        start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader
        )

        elapsed = time.perf_counter() - start_time

        print(f"\nTrain loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.2%}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.2%}")
        print(f"Epoch time: {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict":
                    get_lora_state_dict(model),

                "epoch": epoch,
                "best_val_accuracy": val_acc,

                "model_name": MODEL_NAME,

                "num_classes": NUM_CLASSES,
                "class_names": class_names,

                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_blocks": [9, 10, 11],

                "seed": SEED,
                "history": history.copy()
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

            print(
                f"\nNew best model saved: "
                f"{CHECKPOINT_PATH}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # Reload the best checkpoint
    # --------------------------------------------------------

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    load_lora_state_dict(
        model,
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\nEvaluating best model on test set...")

    test_loss, test_accuracy = evaluate(
        model,
        test_loader
    )

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)

    print("Best epoch:", best_epoch)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2%}"
    )

    print(f"Test loss: {test_loss:.4f}")

    print(
        f"Test accuracy: {test_accuracy:.2%}"
    )

    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
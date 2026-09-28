
import torch

from transformers import ViTForImageClassification

from training.config import (
    MODEL_NAME,
    NUM_CLASSES,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT
)

from training.lora import (
    apply_lora,
    count_parameters,
    print_lora_modules
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load pretrained ViT with a new classification head.
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    # Insert LoRA into the attention layers.
    model = apply_lora(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT
    )

    model = model.to(device)

    count_parameters(model)
    print_lora_modules(model)

    # Verify which modules received LoRA.
    for name, module in model.named_modules():
        if name.endswith("query") or name.endswith("value"):
            if hasattr(module, "lora_A"):
                print("LoRA applied:", name)

    # Test one dummy image.
    model.eval()

    dummy_image = torch.randn(
        1, 3, 224, 224,
        device=device
    )

    with torch.no_grad():
        output = model(
            pixel_values=dummy_image
        )

    print("Output shape:", output.logits.shape)


if __name__ == "__main__":
    main()
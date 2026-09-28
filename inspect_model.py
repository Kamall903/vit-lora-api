from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

print("\n=== ViT STRUCTURE ===")
print(model.vit._modules.keys())

print("\n=== FIRST TRANSFORMER BLOCK ===")
block = model.vit.layers[0]
print(block)

print("\n=== ATTENTION STRUCTURE ===")
print(block.attention)

print("\n=== LINEAR LAYERS ===")
for name, module in model.named_modules():
    if name.startswith("vit.layers.9") and hasattr(module, "in_features"):
        print(name, type(module).__name__)
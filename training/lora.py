
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LoRA LINEAR LAYER
# ============================================================

class LoRALinear(nn.Module):
    """
    Adds LoRA to an existing linear layer.

    Original weights remain frozen.
    Only LoRA matrices A and B are trained.
    """

    def __init__(
        self,
        original_layer,
        rank=16,
        alpha=32,
        dropout=0.1
    ):
        super().__init__()

        if not isinstance(original_layer, nn.Linear):
            raise TypeError(
                "LoRALinear requires an nn.Linear layer"
            )

        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.original_layer = original_layer

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze pretrained weights.
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrix A: [rank, in_features]
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                in_features,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype
            )
        )

        # LoRA matrix B: [out_features, rank]
        self.lora_B = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype
            )
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize A randomly and B with zeros.
        # Initially, LoRA does not change the original output.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5)
        )

    def forward(self, x):

        # Original frozen projection.
        original_output = self.original_layer(x)

        # Low-rank adaptation.
        lora_output = F.linear(
            self.dropout(x),
            self.lora_A
        )

        lora_output = F.linear(
            lora_output,
            self.lora_B
        )

        return (
            original_output
            + self.scaling * lora_output
        )


# ============================================================
# 2. FIND VIT TRANSFORMER BLOCKS
# ============================================================

def get_vit_blocks(model):
    """
    Supports both older and newer Hugging Face
    ViT architectures.
    """

    vit = model.vit

    # Your installed Transformers version.
    if hasattr(vit, "layers"):
        return vit.layers

    # Older Transformers versions.
    if hasattr(vit, "encoder"):
        return vit.encoder.layer

    # Alternative architecture.
    if hasattr(vit, "layer"):
        return vit.layer

    raise RuntimeError(
        "Cannot locate ViT transformer blocks. "
        f"Available modules: {list(vit._modules.keys())}"
    )


# ============================================================
# 3. FIND ATTENTION PROJECTIONS
# ============================================================

def get_attention(block):
    """
    Finds the query and value projection names
    for different Transformers versions.
    """

    attention = block.attention

    # Older attention structure.
    if hasattr(attention, "attention"):
        attention = attention.attention

    # Your installed version uses q_proj and v_proj.
    if (
        hasattr(attention, "q_proj")
        and hasattr(attention, "v_proj")
    ):
        return attention, "q_proj", "v_proj"

    # Older Transformers versions.
    if (
        hasattr(attention, "query")
        and hasattr(attention, "value")
    ):
        return attention, "query", "value"

    raise RuntimeError(
        "Unsupported attention structure. "
        f"Available modules: {list(attention._modules.keys())}"
    )


# ============================================================
# 4. APPLY LORA TO VIT
# ============================================================

def apply_lora(
    model,
    rank=16,
    alpha=32,
    dropout=0.1,
    target_blocks=(9, 10, 11)
):
    """
    Applies LoRA to query and value projections
    in selected ViT transformer blocks.

    All pretrained parameters are frozen.
    The classification head remains trainable.
    """

    # Freeze the entire pretrained model.
    for param in model.parameters():
        param.requires_grad = False

    blocks = get_vit_blocks(model)

    print(f"\nFound {len(blocks)} transformer blocks")

    for block_idx in target_blocks:

        if not 0 <= block_idx < len(blocks):
            raise ValueError(
                f"Invalid transformer block: {block_idx}"
            )

        block = blocks[block_idx]

        attention, query_name, value_name = (
            get_attention(block)
        )

        # Apply LoRA to query and value.
        for projection_name in (
            query_name,
            value_name
        ):

            original_layer = getattr(
                attention,
                projection_name
            )

            # Prevent accidental double wrapping.
            if isinstance(original_layer, LoRALinear):
                raise RuntimeError(
                    f"LoRA already applied to "
                    f"block {block_idx}.{projection_name}"
                )

            lora_layer = LoRALinear(
                original_layer=original_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )

            setattr(
                attention,
                projection_name,
                lora_layer
            )

        print(
            f"LoRA applied to block {block_idx}: "
            f"{query_name} + {value_name}"
        )

    # Keep the classification head trainable.
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ============================================================
# 5. COUNT MODEL PARAMETERS
# ============================================================

def count_parameters(model):
    """
    Prints total, trainable and frozen parameters.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    print("\n========== MODEL PARAMETERS ==========")

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")

    print(
        f"Trainable percentage: "
        f"{100 * trainable / total:.4f}%"
    )

    print("======================================\n")

    return total, trainable


# ============================================================
# 6. EXTRACT LORA CHECKPOINT
# ============================================================

def get_lora_state_dict(model):
    """
    Extracts only LoRA matrices and classifier weights.

    The frozen pretrained ViT weights are not saved.
    """

    state_dict = {}

    for name, param in model.named_parameters():

        if (
            "lora_A" in name
            or "lora_B" in name
            or name.startswith("classifier.")
        ):
            state_dict[name] = (
                param.detach().cpu().clone()
            )

    return state_dict


# ============================================================
# 7. LOAD LORA CHECKPOINT
# ============================================================

def load_lora_state_dict(model, state_dict):
    """
    Loads a previously saved LoRA checkpoint.

    Call apply_lora() before loading the checkpoint.
    """

    expected_keys = {
        name
        for name, _ in model.named_parameters()
        if (
            "lora_A" in name
            or "lora_B" in name
            or name.startswith("classifier.")
        )
    }

    checkpoint_keys = set(state_dict.keys())

    missing = expected_keys - checkpoint_keys
    unexpected = checkpoint_keys - expected_keys

    if missing or unexpected:
        raise RuntimeError(
            "LoRA checkpoint mismatch.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    # Validate tensor shapes before loading.
    model_state = model.state_dict()

    for name, tensor in state_dict.items():
        if tensor.shape != model_state[name].shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: "
                f"expected {model_state[name].shape}, "
                f"received {tensor.shape}"
            )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys: {result.unexpected_keys}"
        )

    print("LoRA checkpoint loaded successfully")

    return model


# ============================================================
# 8. DISPLAY LORA MODULES
# ============================================================

def print_lora_modules(model):
    """
    Prints all layers where LoRA was applied.
    """

    print("\n========== LORA MODULES ==========")

    count = 0

    for name, module in model.named_modules():

        if isinstance(module, LoRALinear):
            print(name)
            count += 1

    print(f"\nTotal LoRA modules: {count}")
    print("==================================\n")
"""
Shared model-loading helpers for the VoxCPM2 + Hinglish LoRA service.
(Copied from voxcpm_api so this app is self-contained for deployment on the E2E box.)
"""

import json
import os

from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

BASE_MODEL = os.environ.get("VOXCPM_BASE_MODEL", "openbmb/VoxCPM2")


def resolve_checkpoint_dir(lora_dir: str, step: int) -> str:
    """Resolve <lora_dir>/step_<step:07d>, raising a helpful error if missing."""
    checkpoint_dir = os.path.join(lora_dir, f"step_{step:07d}")
    if not os.path.isdir(checkpoint_dir):
        available = sorted(os.listdir(lora_dir)) if os.path.isdir(lora_dir) else []
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_dir}' not found. Available in '{lora_dir}': {available}"
        )
    return checkpoint_dir


def load_lora_config(checkpoint_dir: str) -> LoRAConfig:
    """Read lora_config.json from a checkpoint directory and return a LoRAConfig."""
    with open(os.path.join(checkpoint_dir, "lora_config.json")) as f:
        cfg = json.load(f)["lora_config"]

    return LoRAConfig(
        enable_lm=cfg["enable_lm"],
        enable_dit=cfg["enable_dit"],
        enable_proj=cfg.get("enable_proj", False),
        r=cfg["r"],
        alpha=cfg["alpha"],
        dropout=cfg.get("dropout", 0.0),
        target_modules_lm=cfg.get("target_modules_lm", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        target_modules_dit=cfg.get("target_modules_dit", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        target_proj_modules=cfg.get("target_proj_modules", []),
    )


def load_model(lora_dir: str, step: int, load_denoiser: bool = True) -> VoxCPM:
    """Load VoxCPM2 base + inject the LoRA adapter at the given training step."""
    checkpoint_dir = resolve_checkpoint_dir(lora_dir, step)
    lora_config = load_lora_config(checkpoint_dir)
    print(
        f"[model] base={BASE_MODEL} step={step} "
        f"r={lora_config.r} alpha={lora_config.alpha} "
        f"lm={lora_config.enable_lm} dit={lora_config.enable_dit}",
        flush=True,
    )
    model = VoxCPM.from_pretrained(
        BASE_MODEL,
        load_denoiser=load_denoiser,
        lora_config=lora_config,
        lora_weights_path=checkpoint_dir,
    )
    print(f"[model] loaded. lora_enabled={model.lora_enabled}", flush=True)
    return model

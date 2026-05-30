#!/usr/bin/env python3
"""Transformers-based SAM3 model factory for local official checkpoints."""

from __future__ import annotations

from pathlib import Path


def create_model(checkpoint=None, model_config=None, device="cpu"):
    from transformers import Sam3Model, Sam3VideoModel

    model_dir = Path(model_config or checkpoint or "models/upstream/modelscope/facebook__sam3")
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing SAM3 config.json in {model_dir}")
    cls = Sam3VideoModel if (model_dir / "sam3.pt").exists() or (model_dir / "sam3.1_multiplex.pt").exists() else Sam3Model
    model = cls.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return model

#!/usr/bin/env python3
"""Tiny fake SAM3 model factory for export adapter dry-run checks."""

from __future__ import annotations


class _Module:
    def eval(self) -> None:
        return None


class FakeSam3Model:
    def __init__(self) -> None:
        self.image_encoder = _Module()
        self.text_encoder = _Module()
        self.geometry_encoder = _Module()
        self.detector = _Module()
        self.mask_decoder = _Module()
        self.memory_encoder = _Module()
        self.tracker = _Module()
        self.multiplex_detector = _Module()
        self.multiplex_tracker = _Module()

    def eval(self) -> None:
        return None


def create_model(checkpoint=None, model_config=None, device="cpu") -> FakeSam3Model:
    del checkpoint, model_config, device
    return FakeSam3Model()

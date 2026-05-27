#!/usr/bin/env python3
"""Generate SAM3 detector and geometry encoder export contracts for S600 conversion."""

from sam3_export_contract import main


if __name__ == "__main__":
    raise SystemExit(main(["geometry_encoder", "detector"]))

#!/usr/bin/env python3
"""Generate SAM3.1 multiplex export contracts for S600 conversion."""

from sam_s600_tools.export.contract import main


if __name__ == "__main__":
    raise SystemExit(main(["multiplex_detector", "multiplex_tracker", "memory_encoder"]))

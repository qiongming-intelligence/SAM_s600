#!/usr/bin/env python3
"""Generate SAM3 video memory/tracker export contracts for S600 conversion."""

from sam_s600_tools.export.contract import main


if __name__ == "__main__":
    raise SystemExit(main(["memory_encoder", "tracker"]))

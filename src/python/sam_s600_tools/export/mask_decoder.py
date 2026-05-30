#!/usr/bin/env python3
"""Generate the SAM3 mask decoder export contract for S600 conversion."""

from sam_s600_tools.export.contract import main


if __name__ == "__main__":
    raise SystemExit(main(["mask_decoder"]))

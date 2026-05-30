# Model manifests

This directory contains the checked-in SAM3 model manifests used by the C++ runtime.

The manifests reference generated HBM files under `models/hbm/`. Those HBM files are local build artifacts and are intentionally not committed.

Regenerate manifests after conversion with:

```bash
PYTHONPATH=src/python python3 -m sam_s600_tools.manifest.generate --hbm-dir models/hbm --out-dir configs/manifests
```

#!/usr/bin/env python3
"""Download authorized SAM3/SAM3.1 checkpoints from official mirrors.

This script does not bypass upstream access controls. Users must comply with the
upstream license for whichever mirror they use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

REPOS = {
    "sam3": {
        "repo_id": "facebook/sam3",
        "modelscope_id": "facebook/sam3",
        "files": [
            "LICENSE",
            "README.md",
            "config.json",
            "configuration.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
            "sam3.pt",
        ],
    },
    "sam3.1": {
        "repo_id": "facebook/sam3.1",
        "modelscope_id": "facebook/sam3.1",
        "files": [
            "LICENSE",
            "README.md",
            "config.json",
            "configuration.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "sam3.1_multiplex.pt",
        ],
    },
}


def download_huggingface(repo_id: str, files: list[str], out_dir: Path, token: str | None, snapshot: bool) -> None:
    if snapshot:
        print(snapshot_download(repo_id=repo_id, local_dir=out_dir, token=token))
        return
    for filename in files:
        print(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=out_dir, token=token))


def download_modelscope(model_id: str, out_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    print(ms_snapshot_download(model_id, local_dir=str(out_dir)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download official SAM3/SAM3.1 assets after license acceptance.")
    parser.add_argument("--variant", choices=sorted(REPOS), default="sam3")
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--out-dir", type=Path, default=Path("models/upstream"))
    parser.add_argument("--token", help="Hugging Face token; defaults to cached login token")
    parser.add_argument("--snapshot", action="store_true", help="download the whole repo snapshot when using Hugging Face")
    args = parser.parse_args()

    spec = REPOS[args.variant]
    out_dir = args.out_dir / args.source / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.source == "modelscope":
            download_modelscope(spec["modelscope_id"], out_dir)
        else:
            download_huggingface(spec["repo_id"], spec["files"], out_dir, args.token, args.snapshot)
    except GatedRepoError as error:
        raise SystemExit(
            f"Access denied for gated repo {spec['repo_id']}. Accept the upstream license and run `huggingface-cli login`, "
            "or pass --token with an authorized token."
        ) from error
    except HfHubHTTPError as error:
        raise SystemExit(f"failed to download {spec['repo_id']}: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

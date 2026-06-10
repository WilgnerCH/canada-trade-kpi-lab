"""
src/upload_hf.py
================
Upload the processed trade parquet to Hugging Face Hub.

Requires: HF_TOKEN environment variable (write access token)
Dataset:  WilgnerCH/canada-trade-data
"""

import logging
import os
import pathlib

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
LOG = logging.getLogger(__name__)

REPO_ID       = "WilgnerCH/canada-trade-data"
REPO_TYPE     = "dataset"
LOCAL_FILE    = "data/canada_trade.parquet"
REMOTE_FILE   = "canada_trade.parquet"


def upload(local: str = LOCAL_FILE) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    path = pathlib.Path(local)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {local}")

    size_mb = path.stat().st_size / 1e6
    LOG.info("Uploading %s (%.1f MB) → %s/%s", local, size_mb, REPO_ID, REMOTE_FILE)

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=REMOTE_FILE,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update Canada trade data from Statistics Canada",
    )

    LOG.info("Upload complete → https://huggingface.co/datasets/%s", REPO_ID)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Upload trade parquet to Hugging Face Hub")
    ap.add_argument("--file", default=LOCAL_FILE, help="Local parquet file path")
    upload(local=ap.parse_args().file)

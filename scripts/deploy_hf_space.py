#!/usr/bin/env python3
"""Deploy the full NeuroSight Docker Space to Hugging Face.

This stages the FastAPI backend, static Next.js frontend source, synthetic demo
data, and Docker metadata into `.deploy/hf_space_stage`, then uploads that
staged folder as a Docker Space. It does not upload raw/private data,
checkpoints, local caches, `.env` files, or secrets.

Usage:
    export HF_TOKEN=...
    python3 scripts/deploy_hf_space.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, get_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_USERNAME = os.environ.get("HF_USERNAME", "mohi679").strip()
SPACE_NAME = os.environ.get("HF_SPACE_NAME", "neurosight").strip()
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
SPACE_URL = f"https://huggingface.co/spaces/{REPO_ID}"
APP_URL = f"https://{HF_USERNAME}-{SPACE_NAME}.hf.space"
STAGE_DIR = PROJECT_ROOT / ".deploy" / "hf_space_stage"

COPY_DIRS = (
    "api",
    "data",
    "evaluation",
    "frontend",
    "neurosight",
)
COPY_FILES = (
    "Dockerfile",
    "knowledge_graph.py",
    "requirements-space.txt",
    "requirements.txt",
    "requirements.lock",
    "spaces_config.py",
)
IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    "node_modules",
    "out",
    "dist",
    "build",
    ".DS_Store",
}


def _copytree_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in IGNORE_NAMES or name.endswith((".pyc", ".log", ".tmp")):
            ignored.add(name)
    if Path(directory).as_posix().endswith("frontend/public"):
        # Avoid binary image assets in the public Space repo. The Space uses
        # live API responses and links back to GitHub docs for report figures.
        ignored.add("figures")
    return ignored


def require_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip() or (get_token() or "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Set it in your shell with a Hugging Face token "
            "that has write access to the Space, then rerun this script."
        )
    return token


def _copy_path(relative_path: str) -> None:
    src = PROJECT_ROOT / relative_path
    dst = STAGE_DIR / relative_path
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_copytree_ignore)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage_files() -> None:
    """Stage the full Docker Space without private or generated artifacts."""
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)

    for relative_path in COPY_DIRS:
        _copy_path(relative_path)
    for relative_path in COPY_FILES:
        _copy_path(relative_path)

    shutil.copy2(PROJECT_ROOT / "hf_space" / "README.md", STAGE_DIR / "README.md")

    # Keep only public synthetic data in the staged data directory.
    data_dir = STAGE_DIR / "data"
    for child in list(data_dir.iterdir()) if data_dir.exists() else []:
        if child.name not in {"ADNIMERGE_synthetic.csv", "neurosight_kg.json", "processed"}:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / ".gitkeep").touch()

    staged_files = sum(1 for item in STAGE_DIR.rglob("*") if item.is_file())
    print(f"Staged {staged_files} files to {STAGE_DIR}")


def main() -> int:
    token = require_token()
    api = HfApi(token=token)

    print(f"Staging full NeuroSight Docker Space for {REPO_ID}...")
    stage_files()

    print("Creating/updating Hugging Face Docker Space...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    print("Setting non-secret Space variables...")
    api.add_space_variable(REPO_ID, key="APP_ENV", value="local")
    api.add_space_variable(REPO_ID, key="NEUROSIGHT_RUNTIME_MODE", value="demo")
    api.add_space_variable(REPO_ID, key="DISABLE_MRI_WARMUP", value="1")
    api.add_space_variable(REPO_ID, key="NEUROSIGHT_FRONTEND_DIR", value="/app/frontend/out")

    print("Uploading staged full-stack Space...")
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(STAGE_DIR),
        commit_message="Deploy NeuroSight full-stack Docker Space",
    )

    print("")
    print("Deployment uploaded.")
    print(f"Space repo: {SPACE_URL}")
    print(f"App URL:    {APP_URL}")
    print(f"Health URL: {APP_URL}/healthz")
    print(f"API docs:   {APP_URL}/docs")
    print(f"Commit:     {commit.commit_url}")
    print("")
    print("Hugging Face will build the Docker image now; the first build can take several minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

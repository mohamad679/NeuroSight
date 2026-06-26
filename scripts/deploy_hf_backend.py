#!/usr/bin/env python3
"""Deploy the NeuroSight FastAPI backend to a Hugging Face Docker Space.

This script intentionally stages only backend code and never uploads local data.
Set HF_TOKEN securely before running:

    export HF_TOKEN=...
    python3 scripts/deploy_hf_backend.py
"""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path

from huggingface_hub import HfApi, get_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_USERNAME = os.environ.get("HF_USERNAME", "mohi679").strip()
SPACE_NAME = os.environ.get("HF_SPACE_NAME", "neurosight").strip()
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"
BACKEND_URL = f"https://{HF_USERNAME}-{SPACE_NAME}.hf.space"
STAGE_DIR = PROJECT_ROOT / ".deploy" / "hf_backend_stage"

BACKEND_DIRS = ("api", "evaluation", "neurosight")
BACKEND_FILES = ("knowledge_graph.py", "spaces_config.py")
IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
}


def _copytree_ignore(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORE_NAMES or name.endswith(".pyc")}


def require_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip() or (get_token() or "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Run `huggingface-cli login` or "
            "`export HF_TOKEN=...` in your terminal, then rerun this script."
        )
    return token


def resolve_api_key() -> str:
    return os.environ.get("NEUROSIGHT_API_KEY", "").strip() or f"ns-{secrets.token_urlsafe(32)}"


def write_local_env(api_key: str) -> None:
    env_path = PROJECT_ROOT / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                f"BACKEND_URL={BACKEND_URL}",
                f"API_BASE_URL={BACKEND_URL}",
                f"NEUROSIGHT_API_URL={BACKEND_URL}",
                f"NEUROSIGHT_API_KEY={api_key}",
                "PORT=7861",
                "",
            ]
        ),
        encoding="utf-8",
    )


def stage_backend() -> None:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)

    for dirname in BACKEND_DIRS:
        shutil.copytree(
            PROJECT_ROOT / dirname,
            STAGE_DIR / dirname,
            ignore=_copytree_ignore,
        )

    for filename in BACKEND_FILES:
        shutil.copy2(PROJECT_ROOT / filename, STAGE_DIR / filename)

    shutil.copy2(PROJECT_ROOT / "hf_space" / "Dockerfile", STAGE_DIR / "Dockerfile")
    shutil.copy2(PROJECT_ROOT / "hf_space" / "README.md", STAGE_DIR / "README.md")
    shutil.copy2(
        PROJECT_ROOT / "hf_space" / "requirements_backend.txt",
        STAGE_DIR / "requirements.txt",
    )


def main() -> int:
    token = require_token()
    api_key = resolve_api_key()
    api = HfApi(token=token)

    print(f"Staging backend for {REPO_ID}...")
    stage_backend()

    print("Creating/updating Hugging Face Docker Space...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    print("Syncing Space secrets and variables...")
    api.add_space_secret(REPO_ID, key="NEUROSIGHT_API_KEY", value=api_key)
    api.add_space_variable(REPO_ID, key="APP_ENV", value="production")
    api.add_space_variable(REPO_ID, key="DISABLE_MRI_WARMUP", value="1")
    api.add_space_variable(
        REPO_ID,
        key="ALLOWED_ORIGINS",
        value="http://127.0.0.1:7861,http://localhost:7861,http://127.0.0.1:3000,http://localhost:3000",
    )
    for key in ("NEUROSIGHT_PATIENT_CSV_PATH", "NEUROSIGHT_MRI_DIR", "NEUROSIGHT_EEG_DIR"):
        value = os.environ.get(key, "").strip()
        if value:
            api.add_space_variable(REPO_ID, key=key, value=value)

    print("Uploading backend files...")
    commit = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=str(STAGE_DIR),
        commit_message="Deploy NeuroSight FastAPI backend",
    )
    write_local_env(api_key)

    print("")
    print("Deployment uploaded.")
    print(f"Space repo: https://huggingface.co/spaces/{REPO_ID}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Health URL:  {BACKEND_URL}/healthz")
    print(f"Docs URL:    {BACKEND_URL}/docs")
    print(f"Commit:      {commit.commit_url}")
    print("")
    print("Hugging Face will build the Docker image now; the first build can take several minutes.")
    print("Local frontend config was written to .env.local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

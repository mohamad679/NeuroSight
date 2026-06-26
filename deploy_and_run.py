#!/usr/bin/env python3
"""Automate NeuroSight backend deployment to Hugging Face and local Gradio launch.

Run with:
    python3 deploy_and_run.py
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
import select
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.error
import urllib.request
import venv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required to run deploy_and_run.py") from exc


PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG = {
    "project_root": PROJECT_ROOT,
    "hf_id": "mohi679",
    "space_name": "neurosight",
    "hf_token": os.environ.get("HF_TOKEN", "").strip(),
    "space_sdk": "docker",
    "space_hardware": "cpu-basic",
    "space_private": False,
    "backend_public_url": "https://mohi679-neurosight.hf.space",
    "space_repo_url": "https://huggingface.co/spaces/mohi679/neurosight",
    "frontend_env_file": PROJECT_ROOT / "frontend" / ".env",
    "frontend_script_hint": PROJECT_ROOT / "app_local.py",
    "frontend_ports": [7860, 7861],
    "frontend_log_file": PROJECT_ROOT / "logs" / "frontend_local.log",
    "frontend_venv_dir": PROJECT_ROOT / ".venv_frontend",
    "backend_stage_dir": PROJECT_ROOT / ".deploy" / "backend_space",
    "space_git_clone_dir": PROJECT_ROOT / ".deploy" / "space_repo",
    "analysis_auto_continue_seconds": 2,
    "secret_prompt_timeout_seconds": 10,
    "runtime_poll_seconds": 20,
    "runtime_timeout_seconds": 8 * 60,
    "health_wait_seconds": 5,
    "frontend_requirements": [
        "gradio>=4.40.0,<5",
        "fastapi<0.113",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "huggingface_hub<1.0",
    ],
    "api_key": os.environ.get("NEUROSIGHT_API_KEY", "").strip(),
}

IGNORE_PATTERNS = [
    ".env",
    ".env.*",
    ".env.local",
    ".env.production",
    "*.pyc",
    "__pycache__",
    "*.pyo",
    ".git",
    ".gitignore",
    ".gitattributes",
    "*.log",
    "*.tmp",
    "frontend/",
    "frontend",
    "tests/",
    "test/",
    "*.test.py",
    "*.ipynb",
    ".ipynb_checkpoints",
    "node_modules/",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.bin",
    "data/",
    "datasets/",
    "*.egg-info",
    "dist/",
    "build/",
    ".DS_Store",
    "Thumbs.db",
    "DEPLOY_LOG.md",
    "deploy_and_run.py",
]

BACKEND_INCLUDE_DIRS = ["api", "evaluation", "neurosight"]
BACKEND_INCLUDE_FILES = ["knowledge_graph.py", "spaces_config.py"]
IGNORED_SCAN_DIRS = {".deploy", ".git", ".venv", "__pycache__", "node_modules"}


class DeploymentError(RuntimeError):
    """Custom deployment exception with a human-readable suggestion."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.suggestion = suggestion or "Inspect the failing phase output and retry after fixing the issue."


@dataclass
class AnalysisSummary:
    backend_entrypoint_file: Path
    backend_module: str
    backend_app_var: str
    backend_requirements_source: Path | None
    config_files: list[Path] = field(default_factory=list)
    env_vars: list[str] = field(default_factory=list)
    model_references: list[str] = field(default_factory=list)
    frontend_script: Path | None = None
    frontend_dep_packages: list[str] = field(default_factory=list)
    frontend_backend_url_refs: list[str] = field(default_factory=list)
    skip_candidates: list[str] = field(default_factory=list)
    data_path_refs: list[str] = field(default_factory=list)

    @property
    def backend_app_import(self) -> str:
        return f"{self.backend_module}:{self.backend_app_var}"


def redact(text: str) -> str:
    """Avoid leaking sensitive values to stdout/stderr."""
    token = str(CONFIG.get("hf_token") or "")
    return text.replace(token, "***REDACTED***") if token else text


def require_hf_token() -> str:
    """Return the Hugging Face token or stop before any deploy action."""
    token = str(CONFIG.get("hf_token") or "")
    if not token:
        raise DeploymentError(
            "HF_TOKEN is required for Hugging Face deployment.",
            suggestion=(
                "Create a Hugging Face token with write access, then run "
                "`export HF_TOKEN=...` before executing deploy_and_run.py."
            ),
        )
    return token


def resolve_api_key() -> str:
    """Return the API key used by both the Space backend and local frontend."""
    api_key = str(CONFIG.get("api_key") or "").strip()
    if api_key:
        return api_key
    api_key = f"ns-{secrets.token_urlsafe(24)}"
    CONFIG["api_key"] = api_key
    return api_key


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def info(message: str) -> None:
    print(message, flush=True)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and raise a DeploymentError with context on failure."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=check,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as exc:
        stderr = redact(exc.stderr or "")
        stdout = redact(exc.stdout or "")
        raise DeploymentError(
            f"Command failed: {' '.join(cmd[:3])}...\n{stderr or stdout}".strip(),
            suggestion="Re-run the script after verifying network access and required local tooling.",
        ) from exc


def ensure_packages() -> None:
    """Install baseline Python packages if they are missing locally."""
    needed = {
        "huggingface_hub": "huggingface_hub>=0.24.0,<1.0",
    }
    missing = [spec for module, spec in needed.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    section("Bootstrap")
    info(f"Installing bootstrap packages: {', '.join(missing)}")
    run([sys.executable, "-m", "pip", "install", *missing], cwd=PROJECT_ROOT)


ensure_packages()

from huggingface_hub import HfApi  # noqa: E402


def wait_or_continue(message: str, timeout: int) -> None:
    """Allow a short interactive pause but auto-continue by default."""
    info(message)
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return
    info(f"Press ENTER to continue now, or wait {timeout} seconds to auto-continue.")
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        sys.stdin.readline()


def http_call(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    data: bytes | None = None,
) -> tuple[int, str]:
    """Perform a simple HTTP request without relying on external packages."""
    ssl_context = ssl._create_unverified_context()
    request = urllib.request.Request(url, method=method, headers=headers or {}, data=data)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DeploymentError(
            f"HTTP {method} {url} failed: {redact(str(exc))}",
            suggestion="Check network connectivity and rerun once the connection is stable.",
        ) from exc


def discover_env_vars() -> list[str]:
    pattern = re.compile(
        r'os\.environ\.get\("([A-Z0-9_]+)"|os\.getenv\("([A-Z0-9_]+)"|os\.environ\["([A-Z0-9_]+)"\]'
    )
    envs: set[str] = set()
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in IGNORED_SCAN_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in pattern.finditer(text):
            env_name = next(group for group in match.groups() if group)
            envs.add(env_name)
    envs.update({"BACKEND_URL", "API_BASE_URL", "NEUROSIGHT_API_URL"})
    return sorted(envs)


def detect_backend_entrypoint() -> tuple[Path, str, str]:
    candidates: list[tuple[Path, str]] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in (IGNORED_SCAN_DIRS | {"tests"}) for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "FastAPI(" not in text:
            continue
        app_match = re.search(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FastAPI\(", text, re.MULTILINE)
        app_var = app_match.group(1) if app_match else "app"
        candidates.append((path, app_var))

    if not candidates:
        raise DeploymentError(
            "Could not detect a FastAPI entrypoint in the repository.",
            suggestion="Ensure the backend defines an app = FastAPI(...) entrypoint before running deployment.",
        )

    preferred_order = [
        PROJECT_ROOT / "api" / "main.py",
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "app" / "main.py",
    ]
    for preferred in preferred_order:
        for candidate, app_var in candidates:
            if candidate == preferred:
                module = candidate.relative_to(PROJECT_ROOT).with_suffix("").as_posix().replace("/", ".")
                return candidate, module, app_var

    candidate, app_var = sorted(candidates, key=lambda item: len(item[0].parts))[0]
    module = candidate.relative_to(PROJECT_ROOT).with_suffix("").as_posix().replace("/", ".")
    return candidate, module, app_var


def detect_frontend_script() -> tuple[Path | None, list[str], list[str]]:
    candidates: list[Path] = []
    for path in [PROJECT_ROOT / "app_local.py", PROJECT_ROOT / "app.py"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "gradio" in text.lower():
            candidates.append(path)
    if not candidates:
        return None, [], []

    preferred = PROJECT_ROOT / "app_local.py"
    script = preferred if preferred in candidates else candidates[0]
    text = script.read_text(encoding="utf-8", errors="ignore")
    refs = sorted(
        {
            match.group(1)
            for match in re.finditer(r'(BACKEND_URL|API_BASE_URL|NEUROSIGHT_API_URL)', text)
        }
    )
    deps: list[str] = []
    if "import gradio" in text:
        deps.append("gradio")
    if "import requests" in text:
        deps.append("requests")
    if "load_dotenv" in text:
        deps.append("python-dotenv")
    return script, deps, refs


def collect_config_files() -> list[Path]:
    paths: list[Path] = []
    for rel in [
        "requirements.txt",
        "requirements_ui.txt",
        "pyproject.toml",
        "neurosight/configs/default.yaml",
        "spaces_config.py",
        "hf_space/Dockerfile",
        "hf_space/README.md",
        "hf_space/requirements_backend.txt",
    ]:
        path = PROJECT_ROOT / rel
        if path.exists():
            paths.append(path)
    return paths


def collect_model_references() -> list[str]:
    patterns = [
        (PROJECT_ROOT / "app.py", r"data/neurosight_kg\.json"),
        (PROJECT_ROOT / "api" / "main.py", r"data/ADNIMERGE_synthetic\.csv"),
        (PROJECT_ROOT / "scripts" / "train.py", r"best_fusion\.pt"),
        (PROJECT_ROOT / "scripts" / "evaluate.py", r"checkpoints?/best_fusion\.pt"),
        (PROJECT_ROOT / "neurosight" / "models" / "mri.py", r"ViT"),
        (PROJECT_ROOT / "neurosight" / "models" / "eeg.py", r"preprocess_eeg|Transformer"),
    ]
    refs: list[str] = []
    for path, pattern in patterns:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(pattern, text):
            refs.append(str(path.relative_to(PROJECT_ROOT)))
    return refs


def collect_skip_candidates() -> list[str]:
    matches: set[str] = set()
    for path in PROJECT_ROOT.rglob("*"):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if any(part in IGNORED_SCAN_DIRS for part in path.parts):
            continue
        if path.is_dir() and rel in {"tests", "test", "frontend", "data", "datasets", "__pycache__", ".git"}:
            matches.add(rel + "/")
        if path.suffix in {".pyc", ".pyo", ".ipynb"}:
            matches.add(rel)
        if "__pycache__" in path.parts:
            matches.add(rel)
        if path.name == ".env" or path.name.startswith(".env."):
            matches.add(rel)
        if path.is_file() and path.stat().st_size > 500 * 1024 * 1024:
            matches.add(rel)
        if path.suffix in {".pt", ".pth", ".ckpt", ".bin"}:
            matches.add(rel)
    return sorted(matches)


def collect_data_path_refs() -> list[str]:
    refs: list[str] = []
    for path in [PROJECT_ROOT / "api" / "main.py", PROJECT_ROOT / "neurosight" / "configs" / "default.yaml"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "data/" in line or "checkpoints/" in line:
                refs.append(f"{path.relative_to(PROJECT_ROOT)}: {line.strip()}")
    return refs


def analyze_project() -> AnalysisSummary:
    backend_file, backend_module, backend_app_var = detect_backend_entrypoint()
    frontend_script, frontend_deps, frontend_refs = detect_frontend_script()

    requirements_source = None
    for candidate in [
        PROJECT_ROOT / "hf_space" / "requirements_backend.txt",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pyproject.toml",
    ]:
        if candidate.exists():
            requirements_source = candidate
            break

    summary = AnalysisSummary(
        backend_entrypoint_file=backend_file,
        backend_module=backend_module,
        backend_app_var=backend_app_var,
        backend_requirements_source=requirements_source,
        config_files=collect_config_files(),
        env_vars=discover_env_vars(),
        model_references=collect_model_references(),
        frontend_script=frontend_script,
        frontend_dep_packages=frontend_deps,
        frontend_backend_url_refs=frontend_refs,
        skip_candidates=collect_skip_candidates(),
        data_path_refs=collect_data_path_refs(),
    )

    section("Phase 1 - Analysis Summary")
    info(json.dumps(
        {
            "backend": {
                "entrypoint_file": str(summary.backend_entrypoint_file.relative_to(PROJECT_ROOT)),
                "entrypoint_import": summary.backend_app_import,
                "requirements_source": str(summary.backend_requirements_source.relative_to(PROJECT_ROOT))
                if summary.backend_requirements_source
                else None,
                "config_files": [str(path.relative_to(PROJECT_ROOT)) for path in summary.config_files],
                "model_refs": summary.model_references,
                "data_path_refs": summary.data_path_refs,
            },
            "frontend": {
                "script": str(summary.frontend_script.relative_to(PROJECT_ROOT))
                if summary.frontend_script
                else None,
                "backend_url_refs": summary.frontend_backend_url_refs,
                "dependencies": summary.frontend_dep_packages,
                "launch_command": "python app_local.py" if summary.frontend_script else None,
            },
            "environment_variables": summary.env_vars,
            "do_not_upload": summary.skip_candidates,
        },
        indent=2,
    ))

    wait_or_continue(
        "Phase 1 complete. Auto-proceeding to Phase 2.",
        CONFIG["analysis_auto_continue_seconds"],
    )
    return summary


def derive_backend_requirements(summary: AnalysisSummary) -> str:
    curated = PROJECT_ROOT / "hf_space" / "requirements_backend.txt"
    if curated.exists():
        return curated.read_text(encoding="utf-8").strip() + "\n"

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        raise DeploymentError(
            "Could not derive backend runtime dependencies because pyproject.toml is missing.",
            suggestion="Add either hf_space/requirements_backend.txt or pyproject.toml to the repository.",
        )

    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    poetry_deps = data["tool"]["poetry"]["dependencies"]

    ordered_names = [
        "fastapi",
        "uvicorn",
        "slowapi",
        "torch",
        "monai",
        "einops",
        "mne",
        "nibabel",
        "pydicom",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "shap",
        "networkx",
        "langgraph",
        "langchain",
        "huggingface-hub",
    ]

    def convert_spec(name: str, value: Any) -> str:
        if isinstance(value, dict):
            version = value.get("version", "")
        else:
            version = str(value)
        if name == "python":
            return ""
        if name == "uvicorn":
            name = "uvicorn[standard]"
        if name == "huggingface-hub":
            name = "huggingface_hub"
        if name == "slowapi":
            name = "slowapi"
        if version.startswith("^"):
            base = version[1:]
            parts = [int(piece) for piece in base.split(".")]
            if parts[0] > 0:
                upper = f"<{parts[0] + 1}.0.0"
            elif len(parts) > 1 and parts[1] > 0:
                upper = f"<0.{parts[1] + 1}.0"
            else:
                upper = f"<0.0.{parts[2] + 1}"
            return f"{name}>={base},{upper}"
        if version.startswith((">", "<", "=", "!")):
            return f"{name}{version}"
        return f"{name}=={version}"

    lines = [
        "# Auto-generated NeuroSight backend runtime requirements",
        "python-multipart>=0.0.9",
    ]
    for dep_name in ordered_names:
        if dep_name not in poetry_deps:
            continue
        converted = convert_spec(dep_name, poetry_deps[dep_name])
        if converted:
            lines.append(converted)
    return "\n".join(lines) + "\n"


def generate_backend_dockerfile(summary: AnalysisSummary) -> str:
    return textwrap.dedent(
        f"""\
        FROM python:3.11-slim

        WORKDIR /app

        RUN apt-get update && apt-get install -y \\
            libgomp1 \\
            libglib2.0-0 \\
            libsm6 \\
            libxext6 \\
            libxrender1 \\
            libgl1 \\
            build-essential \\
            git \\
            && rm -rf /var/lib/apt/lists/*

        COPY requirements.txt .
        RUN pip install --no-cache-dir --upgrade pip && \\
            pip install --no-cache-dir -r requirements.txt

        COPY . .

        EXPOSE 7860

        ENV PYTHONPATH=/app
        ENV APP_ENV=production
        ENV DISABLE_MRI_WARMUP=1

        CMD ["uvicorn", "{summary.backend_app_import}", "--host", "0.0.0.0", "--port", "7860"]
        """
    )


def generate_space_readme() -> str:
    return textwrap.dedent(
        """\
        ---
        title: NeuroSight Backend
        emoji: 🧠
        colorFrom: blue
        colorTo: purple
        sdk: docker
        pinned: false
        ---

        # NeuroSight Backend API

        Multimodal neurological diagnosis platform backend.
        Access `/docs` for Swagger UI.
        """
    )


def write_backend_artifacts(summary: AnalysisSummary, requirements_text: str) -> tuple[Path, Path, Path]:
    hf_space_dir = PROJECT_ROOT / "hf_space"
    hf_space_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = hf_space_dir / "Dockerfile"
    readme_path = hf_space_dir / "README.md"
    requirements_path = hf_space_dir / "requirements_backend.txt"

    dockerfile_path.write_text(generate_backend_dockerfile(summary), encoding="utf-8")
    readme_path.write_text(generate_space_readme(), encoding="utf-8")
    requirements_path.write_text(requirements_text, encoding="utf-8")

    return dockerfile_path, readme_path, requirements_path


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_into_stage(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_backend_stage(summary: AnalysisSummary, dockerfile_path: Path, readme_path: Path, requirements_path: Path) -> tuple[Path, list[str]]:
    stage_dir = Path(CONFIG["backend_stage_dir"])
    clear_dir(stage_dir)

    uploaded: list[str] = []
    for directory in BACKEND_INCLUDE_DIRS:
        src = PROJECT_ROOT / directory
        if src.exists():
            dst = stage_dir / directory
            copy_into_stage(src, dst)
    for filename in BACKEND_INCLUDE_FILES:
        src = PROJECT_ROOT / filename
        if src.exists():
            copy_into_stage(src, stage_dir / filename)

    copy_into_stage(dockerfile_path, stage_dir / "Dockerfile")
    copy_into_stage(readme_path, stage_dir / "README.md")
    copy_into_stage(requirements_path, stage_dir / "requirements.txt")

    for path in sorted(stage_dir.rglob("*")):
        if path.is_file():
            uploaded.append(path.relative_to(stage_dir).as_posix())

    return stage_dir, uploaded


def maybe_warn_resource_pressure(requirements_text: str) -> None:
    if any(dep in requirements_text for dep in ("torch", "monai", "mne")):
        info("Warning: MONAI/Torch/MNE are included. Hugging Face cpu-basic may be memory-constrained for cold starts.")


def ensure_space(api: HfApi) -> None:
    repo_id = f"{CONFIG['hf_id']}/{CONFIG['space_name']}"
    try:
        exists = api.repo_exists(repo_id=repo_id, repo_type="space")
    except Exception as exc:  # noqa: BLE001
        raise DeploymentError(
            f"Failed to query Hugging Face Space existence: {redact(str(exc))}",
            suggestion="Check your Hugging Face token and network connectivity, then re-run the script.",
        ) from exc

    if not exists:
        section("Phase 2 - Create Space")
        info(f"Creating Space {repo_id} with SDK={CONFIG['space_sdk']} visibility=public")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk=CONFIG["space_sdk"],
                private=CONFIG["space_private"],
                exist_ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise DeploymentError(
                f"Failed to create Hugging Face Space: {redact(str(exc))}",
                suggestion="Verify the token has write access to the target account and retry.",
            ) from exc
    try:
        api.request_space_hardware(repo_id=repo_id, hardware=CONFIG["space_hardware"])
    except Exception:
        info("Note: hardware request skipped or already satisfied.")


def sync_space_secrets(api: HfApi) -> None:
    """Ensure the Hugging Face Space has the secrets required by this repo."""
    repo_id = f"{CONFIG['hf_id']}/{CONFIG['space_name']}"
    api_key = resolve_api_key()
    section("Phase 2 - Sync Space Secrets")
    try:
        api.add_space_secret(repo_id=repo_id, key="NEUROSIGHT_API_KEY", value=api_key)
        info("Configured Hugging Face Space secret: NEUROSIGHT_API_KEY")
    except Exception as exc:  # noqa: BLE001
        raise DeploymentError(
            f"Failed to sync Hugging Face Space secrets: {redact(str(exc))}",
            suggestion="Verify the token can manage Space secrets, then retry the deployment.",
        ) from exc


def upload_stage_with_hub(api: HfApi, stage_dir: Path) -> str:
    repo_id = f"{CONFIG['hf_id']}/{CONFIG['space_name']}"
    for attempt in range(1, 4):
        try:
            commit_url = api.upload_folder(
                folder_path=str(stage_dir),
                repo_id=repo_id,
                repo_type="space",
                ignore_patterns=IGNORE_PATTERNS,
            )
            return str(commit_url)
        except Exception as exc:  # noqa: BLE001
            info(f"[{ts()}] upload_folder attempt {attempt}/3 failed: {redact(str(exc))}")
            if attempt < 3:
                time.sleep(3 * attempt)
    raise DeploymentError(
        "huggingface_hub.upload_folder failed after 3 attempts.",
        suggestion="Retry on a more stable network or let the script fall back to git-based push.",
    )


def sync_stage_to_clone(stage_dir: Path, clone_dir: Path) -> None:
    for child in clone_dir.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in stage_dir.iterdir():
        dst = clone_dir / child.name
        copy_into_stage(child, dst)


def upload_stage_with_git(stage_dir: Path) -> str:
    clone_dir = Path(CONFIG["space_git_clone_dir"])
    public_url = CONFIG["space_repo_url"]
    git_env = os.environ.copy()
    git_env["GIT_SSL_NO_VERIFY"] = "1"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", public_url, str(clone_dir)], cwd=PROJECT_ROOT, env=git_env)
    sync_stage_to_clone(stage_dir, clone_dir)
    run(["git", "-C", str(clone_dir), "config", "user.email", f"{CONFIG['hf_id']}@users.noreply.huggingface.co"])
    run(["git", "-C", str(clone_dir), "config", "user.name", CONFIG["hf_id"]])
    status = run(["git", "-C", str(clone_dir), "status", "--short"], cwd=PROJECT_ROOT)
    if not (status.stdout or "").strip():
        head = run(["git", "-C", str(clone_dir), "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT)
        return f"{CONFIG['space_repo_url']}/commit/{head.stdout.strip()}"
    run(["git", "-C", str(clone_dir), "add", "."], cwd=PROJECT_ROOT)
    run(["git", "-C", str(clone_dir), "commit", "-m", "Automated NeuroSight backend deploy"], cwd=PROJECT_ROOT)
    auth_url = f"https://{CONFIG['hf_id']}:{CONFIG['hf_token']}@huggingface.co/spaces/{CONFIG['hf_id']}/{CONFIG['space_name']}"
    run(["git", "-C", str(clone_dir), "push", auth_url, "main"], cwd=PROJECT_ROOT, env=git_env)
    head = run(["git", "-C", str(clone_dir), "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT)
    return f"{CONFIG['space_repo_url']}/commit/{head.stdout.strip()}"


def upload_stage(api: HfApi, stage_dir: Path) -> str:
    section("Phase 3 - Upload Backend")
    try:
        commit_url = upload_stage_with_hub(api, stage_dir)
        info(f"Backend upload complete via huggingface_hub: {commit_url}")
        return commit_url
    except DeploymentError as hub_error:
        info(f"Hub upload failed, falling back to git push: {hub_error}")
        commit_url = upload_stage_with_git(stage_dir)
        info(f"Backend upload complete via git: {commit_url}")
        return commit_url


def hf_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {CONFIG['hf_token']}"}


def restart_space() -> None:
    url = f"https://huggingface.co/api/spaces/{CONFIG['hf_id']}/{CONFIG['space_name']}/restart"
    try:
        status, body = http_call(url, method="POST", headers=hf_headers(), timeout=60, data=b"")
        if status >= 400:
            info(f"Restart warning: HTTP {status} {body}")
    except Exception as exc:  # noqa: BLE001
        info(f"Restart warning: {redact(str(exc))}")


def poll_space_runtime() -> str:
    section("Phase 3 - Poll Runtime")
    runtime_url = f"https://huggingface.co/api/spaces/{CONFIG['hf_id']}/{CONFIG['space_name']}/runtime"
    started = time.time()
    last_stage = "UNKNOWN"
    while time.time() - started <= CONFIG["runtime_timeout_seconds"]:
        try:
            status, body = http_call(runtime_url, headers=hf_headers(), timeout=60)
            if status >= 400:
                raise DeploymentError(
                    f"Runtime endpoint returned HTTP {status}: {body}",
                    suggestion="Check your Hugging Face token, Space visibility, and network connectivity.",
                )
            payload = json.loads(body)
            stage = payload.get("stage", "UNKNOWN")
            last_stage = stage
            elapsed = int(time.time() - started)
            info(f"[{ts()}] Space status: {stage}... ({elapsed}s elapsed)")
            if stage == "RUNNING":
                return stage
            if stage in {"BUILD_ERROR", "RUNTIME_ERROR"}:
                error_message = payload.get("errorMessage", "").strip()
                raise DeploymentError(
                    f"Space failed with stage {stage}: {error_message or 'no error message returned'}",
                    suggestion="Check the Space logs on Hugging Face and fix the failing dependency or startup path.",
                )
        except DeploymentError:
            raise
        except Exception as exc:  # noqa: BLE001
            elapsed = int(time.time() - started)
            info(f"[{ts()}] Space status: UNREACHABLE ({elapsed}s elapsed) — {redact(str(exc))}")
        time.sleep(CONFIG["runtime_poll_seconds"])
    raise DeploymentError(
        f"Timed out after {CONFIG['runtime_timeout_seconds']}s waiting for the Space to reach RUNNING (last stage: {last_stage}).",
        suggestion="Inspect the Space runtime/build logs on Hugging Face and retry once the root cause is fixed.",
    )


def backend_healthcheck() -> str:
    for endpoint in ["/healthz", "/docs"]:
        url = CONFIG["backend_public_url"].rstrip("/") + endpoint
        try:
            status, _ = http_call(url, timeout=30)
            if status == 200:
                return url
        except Exception:
            continue
    raise DeploymentError(
        "Backend health check failed for both /healthz and /docs.",
        suggestion="Verify the Space is fully running and reachable over HTTPS before retrying.",
    )


def find_free_port(preferred: Iterable[int]) -> int:
    for port in preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise DeploymentError(
        "Could not find a free frontend port from the preferred list.",
        suggestion="Free up ports 7860/7861 or adjust CONFIG['frontend_ports'].",
    )


def ensure_frontend_env() -> Path:
    env_path = Path(CONFIG["frontend_env_file"])
    env_path.parent.mkdir(parents=True, exist_ok=True)
    api_key = resolve_api_key()
    env_text = textwrap.dedent(
        f"""\
        BACKEND_URL={CONFIG['backend_public_url']}
        API_BASE_URL={CONFIG['backend_public_url']}
        NEUROSIGHT_API_URL={CONFIG['backend_public_url']}
        NEUROSIGHT_API_KEY={api_key}
        """
    )
    env_path.write_text(env_text, encoding="utf-8")
    return env_path


def frontend_python_path() -> Path:
    venv_dir = Path(CONFIG["frontend_venv_dir"])
    return venv_dir / "bin" / "python"


def install_frontend_deps(summary: AnalysisSummary) -> None:
    section("Phase 4 - Install Frontend Dependencies")
    _ = summary
    deps = list(CONFIG["frontend_requirements"])
    venv_dir = Path(CONFIG["frontend_venv_dir"])
    python_path = frontend_python_path()
    if not python_path.exists():
        info(f"Creating local frontend virtualenv at {venv_dir.relative_to(PROJECT_ROOT)}")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    info("Installing lightweight local frontend dependencies...")
    run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], cwd=PROJECT_ROOT)
    run([str(python_path), "-m", "pip", "install", *deps], cwd=PROJECT_ROOT)


def launch_frontend(summary: AnalysisSummary) -> tuple[int, subprocess.Popen[Any] | None]:
    if summary.frontend_script is None:
        raise DeploymentError(
            "No local frontend script could be detected.",
            suggestion="Ensure app_local.py or another Gradio frontend script exists before launching.",
        )

    port = find_free_port(CONFIG["frontend_ports"])
    log_file = Path(CONFIG["frontend_log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "BACKEND_URL": CONFIG["backend_public_url"],
            "API_BASE_URL": CONFIG["backend_public_url"],
            "NEUROSIGHT_API_URL": CONFIG["backend_public_url"],
            "HOST": "127.0.0.1",
            "PORT": str(port),
        }
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            return port, None

    log_handle = log_file.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [str(frontend_python_path()), str(summary.frontend_script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return port, process


def frontend_healthcheck(port: int) -> str:
    url = f"http://localhost:{port}"
    try:
        status, body = http_call(url, timeout=20)
        if status == 200:
            return url
    except Exception as exc:  # noqa: BLE001
        raise DeploymentError(
            f"Frontend health check failed: {exc}",
            suggestion="Open the frontend log and verify the local Gradio process started successfully.",
        ) from exc
    raise DeploymentError(
        f"Frontend health check returned non-200 output: {body[:200]}",
        suggestion="Inspect the local frontend log and retry after fixing the launch issue.",
    )


def write_deploy_log(summary: AnalysisSummary, uploaded_files: list[str], backend_health_url: str, frontend_url: str) -> None:
    log_path = PROJECT_ROOT / "DEPLOY_LOG.md"
    skipped = "\n".join(f"- `{item}`" for item in (summary.skip_candidates or ["(none detected)"]))
    uploaded = "\n".join(f"- `{item}`" for item in uploaded_files)
    env_vars = "\n".join(f"- `{item}`" for item in summary.env_vars)
    text = textwrap.dedent(
        f"""\
        # Deploy Log

        - Timestamp: {datetime.now(timezone.utc).isoformat()}
        - Hugging Face Space: {CONFIG['space_repo_url']}
        - Backend URL: {CONFIG['backend_public_url']}
        - Frontend URL: {frontend_url}
        - Backend Health URL: {backend_health_url}

        ## Uploaded Files
        {uploaded}

        ## Skipped Files
        {skipped}

        ## Environment Variables / Secrets To Set
        {env_vars}
        """
    )
    log_path.write_text(text, encoding="utf-8")


def print_final_report(backend_url: str, frontend_url: str) -> None:
    report = textwrap.dedent(
        f"""\
        ╔══════════════════════════════════════════════════════════╗
        ║           NeuroSight Deployment Status Report            ║
        ╠══════════════════════════════════════════════════════════╣
        ║ 🧠 Backend (HuggingFace Space)                          ║
        ║    URL: {backend_url:<46}║
        ║    Status: ✅ RUNNING                                   ║
        ╠══════════════════════════════════════════════════════════╣
        ║ 🖥️  Frontend (Local Gradio)                             ║
        ║    URL: {frontend_url:<46}║
        ║    Status: ✅ RUNNING                                   ║
        ╠══════════════════════════════════════════════════════════╣
        ║ 🔗 API Connection: ✅ OK                                ║
        ║ 📋 Swagger Docs:   {backend_url}/docs ║
        ╚══════════════════════════════════════════════════════════╝
        """
    )
    info(report)


def main() -> None:
    summary = analyze_project()
    requirements_text = derive_backend_requirements(summary)
    maybe_warn_resource_pressure(requirements_text)
    dockerfile_path, readme_path, requirements_path = write_backend_artifacts(summary, requirements_text)
    stage_dir, uploaded_files = prepare_backend_stage(summary, dockerfile_path, readme_path, requirements_path)

    section("Phase 2 - Prepare Space")
    api = HfApi(token=require_hf_token())
    ensure_space(api)
    sync_space_secrets(api)

    commit_url = upload_stage(api, stage_dir)
    info(f"Latest backend commit: {commit_url}")
    restart_space()
    poll_space_runtime()
    backend_health_url = backend_healthcheck()
    info(f"🧠 NeuroSight Backend is live at: {CONFIG['backend_public_url']}")

    section("Phase 4 - Configure Local Frontend")
    env_path = ensure_frontend_env()
    info(f"Updated frontend env file: {env_path.relative_to(PROJECT_ROOT)}")
    install_frontend_deps(summary)
    port, process = launch_frontend(summary)
    if process is not None:
        info(f"Started local frontend process with PID {process.pid}")
    time.sleep(CONFIG["health_wait_seconds"])
    frontend_url = frontend_healthcheck(port)
    info(f"🖥️  NeuroSight Gradio UI is running at: {frontend_url}")

    write_deploy_log(summary, uploaded_files, backend_health_url, frontend_url)
    print_final_report(CONFIG["backend_public_url"], frontend_url)


if __name__ == "__main__":
    try:
        main()
    except DeploymentError as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1] if tb else None
        section("Deployment Failed")
        info(f"Error: {exc}")
        if last is not None:
            info(f"Location: {last.filename}:{last.lineno}")
        info(f"Suggested fix: {exc.suggestion}")
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)
        last = tb[-1] if tb else None
        section("Deployment Failed")
        info(f"Error: {redact(str(exc))}")
        if last is not None:
            info(f"Location: {last.filename}:{last.lineno}")
        info("Suggested fix: Inspect the traceback above, fix the failing phase, and rerun deploy_and_run.py.")
        raise SystemExit(1) from exc

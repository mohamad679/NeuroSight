#!/usr/bin/env python3
"""Backend proof script for NeuroSight.

This script exercises real FastAPI routes with synthetic, GitHub-safe inputs:

- GET /healthz
- GET /v1/modalities/status
- GET /v1/models/checkpoint/status
- GET /v1/xai/status
- GET /v1/eval/report
- GET /v1/demo/readiness
- POST /v1/upload/mri with a generated .npy volume
- POST /v1/upload/eeg with a generated .npy signal
- POST /v1/risk-profile using cognitive scores plus returned embeddings

It is intended for reviewers and CI-style local verification. It does not make
clinical claims and does not require real patient data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "dev-key"


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    status_code: int | None = None
    request_id: str | None = None


class BackendSmoke:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.results: list[StepResult] = []
        self.mri_embedding: list[float] | None = None
        self.eeg_embedding: list[float] | None = None

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def add_result(
        self,
        name: str,
        ok: bool,
        detail: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.results.append(
            StepResult(
                name=name,
                ok=ok,
                detail=detail,
                status_code=status_code,
                request_id=request_id,
            )
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> requests.Response:
        return requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            headers=headers,
            json=json_payload,
            files=files,
            timeout=self.timeout,
        )

    def get_json(self, name: str, path: str, *, auth: bool = True) -> dict[str, Any] | None:
        headers = self.auth_headers if auth else None
        try:
            response = self.request("GET", path, headers=headers)
            request_id = response.headers.get("X-Request-ID")
            payload = parse_response_json(response)
            if response.ok and isinstance(payload, dict):
                summary = summarize_payload(payload)
                self.add_result(name, True, summary, response.status_code, request_id)
                return payload
            detail = error_detail(payload, response.text)
            self.add_result(name, False, detail, response.status_code, request_id)
            return None
        except requests.RequestException as exc:
            self.add_result(name, False, str(exc))
            return None

    def healthz(self) -> bool:
        payload = self.get_json("healthz", "/healthz", auth=False)
        return payload is not None

    def protected_status_routes(self) -> None:
        self.get_json("modalities_status", "/v1/modalities/status")
        self.get_json("checkpoint_status", "/v1/models/checkpoint/status")
        self.get_json("xai_status", "/v1/xai/status")
        self.get_json("eval_report", "/v1/eval/report")
        self.get_json("demo_readiness", "/v1/demo/readiness")

    def upload_file(self, name: str, path: str, file_path: Path, filename: str) -> list[float] | None:
        try:
            with file_path.open("rb") as file_obj:
                response = self.request(
                    "POST",
                    path,
                    headers=self.auth_headers,
                    files={"file": (filename, file_obj, "application/octet-stream")},
                )
            request_id = response.headers.get("X-Request-ID")
            payload = parse_response_json(response)
            if response.ok and isinstance(payload, dict):
                embedding = payload.get("embedding")
                embedding_dim = payload.get("embedding_dim")
                if isinstance(embedding, list) and isinstance(embedding_dim, int):
                    self.add_result(
                        name,
                        True,
                        f"embedding_dim={embedding_dim}, embedding_len={len(embedding)}",
                        response.status_code,
                        request_id,
                    )
                    return [float(value) for value in embedding if isinstance(value, int | float)]
                self.add_result(
                    name,
                    False,
                    "upload response missing embedding or embedding_dim",
                    response.status_code,
                    request_id,
                )
                return None
            detail = error_detail(payload, response.text)
            self.add_result(name, False, detail, response.status_code, request_id)
            return None
        except requests.RequestException as exc:
            self.add_result(name, False, str(exc))
            return None
        except OSError as exc:
            self.add_result(name, False, f"unable to read generated file: {exc}")
            return None

    def upload_synthetic_modalities(self) -> None:
        with tempfile.TemporaryDirectory(prefix="neurosight-smoke-") as temp_dir:
            temp_path = Path(temp_dir)
            mri_path = temp_path / "synthetic_mri.npy"
            eeg_path = temp_path / "synthetic_eeg.npy"
            write_synthetic_mri(mri_path)
            write_synthetic_eeg(eeg_path)
            self.mri_embedding = self.upload_file(
                "upload_mri",
                "/v1/upload/mri",
                mri_path,
                "synthetic_mri.npy",
            )
            self.eeg_embedding = self.upload_file(
                "upload_eeg",
                "/v1/upload/eeg",
                eeg_path,
                "synthetic_eeg.npy",
            )

    def diagnose(self) -> None:
        payload: dict[str, Any] = {
            "query": "Smoke test: run a synthetic multimodal diagnosis.",
            "cognitive_scores": {
                "mmse": 24.0,
                "moca": 20.0,
                "cdrsb": 4.5,
                "adas11": 18.0,
                "ravlt_immediate": 28.0,
                "ravlt_learning": 2.0,
                "faq": 6.0,
                "age": 70.0,
            },
        }
        if self.mri_embedding:
            payload["mri_embedding"] = self.mri_embedding
        if self.eeg_embedding:
            payload["eeg_embedding"] = self.eeg_embedding

        try:
            response = self.request(
                "POST",
                "/v1/risk-profile",
                headers={**self.auth_headers, "Content-Type": "application/json"},
                json_payload=payload,
            )
            request_id = response.headers.get("X-Request-ID")
            response_payload = parse_response_json(response)
            if response.ok and isinstance(response_payload, dict):
                missing = [
                    key
                    for key in ["diagnosis", "confidence", "requires_review", "report_text"]
                    if key not in response_payload
                ]
                if missing:
                    self.add_result(
                        "diagnose",
                        False,
                        f"response missing keys: {', '.join(missing)}",
                        response.status_code,
                        request_id,
                    )
                    return
                diagnosis = response_payload.get("diagnosis")
                confidence = response_payload.get("confidence")
                self.add_result(
                    "diagnose",
                    True,
                    f"diagnosis={diagnosis}, confidence={confidence}",
                    response.status_code,
                    request_id,
                )
                return
            detail = error_detail(response_payload, response.text)
            self.add_result("diagnose", False, detail, response.status_code, request_id)
        except requests.RequestException as exc:
            self.add_result("diagnose", False, str(exc))

    def passed(self) -> bool:
        return all(result.ok for result in self.results)


def parse_response_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def error_detail(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail, sort_keys=True)
    return fallback.strip() or "request failed"


def summarize_payload(payload: dict[str, Any]) -> str:
    useful_keys = [
        "status",
        "mode",
        "runtime_mode",
        "version",
        "trained_checkpoint_loaded",
        "message",
    ]
    parts = []
    for key in useful_keys:
        value = payload.get(key)
        if isinstance(value, str | int | float | bool):
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else f"keys={len(payload)}"


def write_synthetic_mri(path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    volume = rng.normal(loc=0.0, scale=0.25, size=(24, 24, 24)).astype("float32")
    volume[8:16, 8:16, 8:16] += 0.75
    np.save(path, volume)


def write_synthetic_eeg(path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(123)
    seconds = 4
    sfreq = 256
    time = np.linspace(0, seconds, seconds * sfreq, endpoint=False, dtype=np.float32)
    channels = []
    for channel_index in range(19):
        base = np.sin(2 * np.pi * (6 + channel_index % 5) * time)
        noise = rng.normal(0.0, 0.05, size=time.shape)
        channels.append(base + noise)
    eeg = np.asarray(channels, dtype="float32")
    np.save(path, eeg)


def print_human_report(smoke: BackendSmoke) -> None:
    print("\nNeuroSight backend smoke test")
    print(f"Base URL: {smoke.base_url}")
    print(f"API key: {'configured' if smoke.api_key else 'missing'}")
    print("")

    for result in smoke.results:
        marker = "PASS" if result.ok else "FAIL"
        status = f" status={result.status_code}" if result.status_code is not None else ""
        request_id = f" request_id={result.request_id}" if result.request_id else ""
        print(f"[{marker}] {result.name}{status}{request_id}")
        print(f"       {result.detail}")

    print("")
    print("Result:", "PASS" if smoke.passed() else "FAIL")
    if not smoke.passed():
        print("Hint: start FastAPI first, for example:")
        print("      NEUROSIGHT_API_KEY=dev-key uvicorn api.main:app --reload --port 8000")
        print("      If FastAPI is already running with another key, pass --api-key <matching-key>.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_api_key = discover_api_key()
    parser = argparse.ArgumentParser(
        description="Exercise real NeuroSight FastAPI backend routes with synthetic safe inputs."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEUROSIGHT_BACKEND_URL", DEFAULT_BASE_URL),
        help=f"Backend base URL. Defaults to NEUROSIGHT_BACKEND_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        help=(
            "API key for X-API-Key. Defaults to NEUROSIGHT_API_KEY, NEXT_PUBLIC_API_KEY, "
            "local .env files, or dev-key."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-uploads",
        action="store_true",
        help="Skip MRI/EEG upload routes and run diagnosis with cognitive scores only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human report.",
    )
    return parser.parse_args(argv)


def discover_api_key() -> str:
    for key in ["NEUROSIGHT_API_KEY", "NEXT_PUBLIC_API_KEY"]:
        value = os.environ.get(key, "").strip()
        if value:
            return value

    for env_path in [Path(".env.local"), Path("frontend/.env.local"), Path(".env.example")]:
        parsed = read_env_file(env_path)
        for key in ["NEUROSIGHT_API_KEY", "NEXT_PUBLIC_API_KEY"]:
            value = parsed.get(key, "").strip()
            if value:
                return value

    return DEFAULT_API_KEY


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return values


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    smoke = BackendSmoke(args.base_url, args.api_key, args.timeout)

    if smoke.healthz():
        smoke.protected_status_routes()
        if not args.skip_uploads:
            smoke.upload_synthetic_modalities()
        smoke.diagnose()

    if args.json:
        print(json.dumps([asdict(result) for result in smoke.results], indent=2))
    else:
        print_human_report(smoke)

    return 0 if smoke.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

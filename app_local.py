"""NeuroSight local frontend.

This lightweight server runs on the Mac and proxies browser requests to the
remote Hugging Face FastAPI backend. It intentionally avoids torch, MONAI, MNE,
and Gradio so the local UI can run on a storage-limited machine.

Run:
    python3 app_local.py
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

import requests


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "local_ui"
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


_load_env_file(ROOT / "frontend" / ".env")
_load_env_file(ROOT / ".env.local", override=True)
_load_env_file(ROOT / ".env")

BACKEND_URL = (
    os.environ.get("BACKEND_URL")
    or os.environ.get("API_BASE_URL")
    or os.environ.get("NEUROSIGHT_API_URL")
    or "https://mohi679-neurosight.hf.space"
).rstrip("/")
API_KEY = os.environ.get("NEUROSIGHT_API_KEY", "dev-key")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "7861"))
TIMEOUT_SECONDS = float(os.environ.get("NEUROSIGHT_FRONTEND_TIMEOUT_SECONDS", "300"))
TEXT_RESPONSE_PREVIEW_CHARS = 4000


# Browser UI lives in local_ui/ and is served as static assets.
def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_backend_response(handler: BaseHTTPRequestHandler, response: requests.Response) -> None:
    try:
        payload: object = response.json()
    except ValueError:
        body = response.text
        payload = {
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "body_preview": body[:TEXT_RESPONSE_PREVIEW_CHARS],
            "truncated": len(body) > TEXT_RESPONSE_PREVIEW_CHARS,
        }
    _send_json(handler, response.status_code, payload)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(length).decode("utf-8") or "{}")


def _send_static(handler: BaseHTTPRequestHandler, request_path: str) -> bool:
    route_path = "index.html" if request_path in {"/", "/index.html"} else request_path.lstrip("/")
    file_path = (STATIC_ROOT / route_path).resolve()
    try:
        file_path.relative_to(STATIC_ROOT.resolve())
    except ValueError:
        _send_json(handler, 404, {"detail": "Not found"})
        return True

    if not file_path.exists() or not file_path.is_file():
        return False

    body = file_path.read_bytes()
    if file_path.suffix == ".html":
        body = body.decode("utf-8").replace("__BACKEND_URL__", BACKEND_URL).encode("utf-8")

    content_type = STATIC_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _backend_headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _backend_upload_headers(content_type: str | None) -> dict[str, str]:
    headers = {"X-API-Key": API_KEY}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


class NeuroSightHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[local-ui] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html", "/style.css", "/app.js"}:
            if not _send_static(self, path):
                _send_json(self, 404, {"detail": "Static asset not found"})
            return

        if path == "/api/health":
            try:
                response = requests.get(f"{BACKEND_URL}/healthz", timeout=30)
                _send_backend_response(self, response)
            except requests.RequestException as exc:
                _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})
            return

        if path == "/api/backend/root":
            try:
                response = requests.get(f"{BACKEND_URL}/", timeout=30)
                _send_backend_response(self, response)
            except requests.RequestException as exc:
                _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})
            return

        if path in {"/api/data/status", "/api/data/demo-patients"}:
            endpoint = path.replace("/api", "/v1", 1)
            if parsed.query:
                endpoint = f"{endpoint}?{parsed.query}"
            self._handle_get_proxy(endpoint)
            return

        if path == "/api/demo/readiness":
            self._handle_get_proxy("/v1/demo/readiness")
            return

        if path == "/api/modalities/status":
            self._handle_get_proxy("/v1/modalities/status")
            return

        if path == "/api/governance/status":
            self._handle_get_proxy("/v1/governance/status")
            return

        if path in {"/api/eval/metrics", "/api/eval/history", "/api/eval/report"}:
            self._handle_get_proxy(path.replace("/api", "/v1", 1))
            return

        if path in {"/api/models", "/api/models/production"}:
            self._handle_get_proxy(path.replace("/api", "/v1", 1))
            return

        if path == "/api/models/checkpoint/status":
            self._handle_get_proxy("/v1/models/checkpoint/status")
            return

        if path == "/api/xai":
            self._handle_xai_proxy(parsed.query)
            return

        if path == "/api/xai/status":
            self._handle_get_proxy("/v1/xai/status")
            return

        _send_json(self, 404, {"detail": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/api/diagnose":
            self._handle_diagnose()
            return

        if path == "/api/diagnose/stream":
            self._handle_stream_proxy()
            return

        if path == "/api/upload/cognitive":
            self._handle_json_proxy("/v1/upload/cognitive")
            return

        if path in {"/api/upload/mri", "/api/upload/eeg"}:
            endpoint = path.replace("/api", "/v1", 1)
            self._handle_upload_proxy(endpoint)
            return

        if path == "/api/kg/query":
            self._handle_json_proxy("/v1/kg/query")
            return

        if path in {"/api/eval/run", "/api/eval/cv", "/api/eval/benchmark"}:
            self._handle_post_proxy(path.replace("/api", "/v1", 1))
            return

        if path == "/api/models/promote":
            self._handle_model_promote_proxy()
            return

        _send_json(self, 404, {"detail": "Not found"})

    def _handle_get_proxy(self, endpoint: str) -> None:
        try:
            response = requests.get(
                f"{BACKEND_URL}{endpoint}",
                headers=_backend_headers(),
                timeout=TIMEOUT_SECONDS,
            )
            _send_backend_response(self, response)
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})

    def _handle_xai_proxy(self, raw_query: str) -> None:
        query = parse_qs(raw_query)
        patient_id = (query.get("patient_id") or [""])[0].strip()
        if not patient_id:
            _send_json(self, 422, {"detail": "patient_id is required"})
            return

        params: dict[str, str] = {"modality": (query.get("modality") or ["cognitive"])[0]}
        target_class = (query.get("target_class") or [""])[0].strip()
        if target_class:
            params["target_class"] = target_class
        endpoint = f"/v1/xai/{quote(patient_id)}?{urlencode(params)}"
        self._handle_get_proxy(endpoint)

    def _handle_post_proxy(self, endpoint: str) -> None:
        try:
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                headers=_backend_headers(),
                timeout=TIMEOUT_SECONDS,
            )
            _send_backend_response(self, response)
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})

    def _handle_model_promote_proxy(self) -> None:
        try:
            payload = _read_json_body(self)
            run_id = str(payload.get("run_id") or "").strip()
            if not run_id:
                _send_json(self, 422, {"detail": "run_id is required"})
                return
            response = requests.post(
                f"{BACKEND_URL}/v1/models/{quote(run_id)}/promote",
                headers=_backend_headers(),
                timeout=TIMEOUT_SECONDS,
            )
            _send_backend_response(self, response)
        except (json.JSONDecodeError, ValueError) as exc:
            _send_json(self, 400, {"detail": f"Invalid request: {exc}"})
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})

    def _handle_json_proxy(self, endpoint: str) -> None:
        try:
            payload = _read_json_body(self)
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                headers=_backend_headers(),
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            _send_backend_response(self, response)
        except (json.JSONDecodeError, ValueError) as exc:
            _send_json(self, 400, {"detail": f"Invalid request: {exc}"})
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})

    def _handle_upload_proxy(self, endpoint: str) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                headers=_backend_upload_headers(self.headers.get("Content-Type")),
                data=body,
                timeout=TIMEOUT_SECONDS,
            )
            _send_backend_response(self, response)
        except ValueError as exc:
            _send_json(self, 400, {"detail": f"Invalid request: {exc}"})
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})

    def _handle_stream_proxy(self) -> None:
        try:
            payload = _read_json_body(self)
            response = requests.post(
                f"{BACKEND_URL}/v1/diagnose/stream",
                headers=_backend_headers(),
                json=payload,
                stream=True,
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                _send_backend_response(self, response)
                response.close()
                return

            self.send_response(response.status_code)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            finally:
                response.close()
        except (json.JSONDecodeError, ValueError) as exc:
            _send_json(self, 400, {"detail": f"Invalid request: {exc}"})
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})

    def _handle_diagnose(self) -> None:
        try:
            payload = _read_json_body(self)
            patient_id = str(payload.get("patient_id") or "").strip()

            if patient_id:
                endpoint = f"/v1/diagnose/patient/{quote(patient_id)}"
                backend_payload = {"query": payload.get("query")}
            else:
                endpoint = "/v1/diagnose"
                backend_payload = {
                    "query": payload.get("query"),
                    "cognitive_scores": payload.get("cognitive_scores", {}),
                }
                for key in ("mri_embedding", "eeg_embedding", "cog_embedding"):
                    if payload.get(key):
                        backend_payload[key] = payload[key]

            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                headers=_backend_headers(),
                json=backend_payload,
                timeout=TIMEOUT_SECONDS,
            )
            _send_backend_response(self, response)
        except (json.JSONDecodeError, ValueError) as exc:
            _send_json(self, 400, {"detail": f"Invalid request: {exc}"})
        except requests.RequestException as exc:
            _send_json(self, 502, {"detail": str(exc), "backend": BACKEND_URL})


def main() -> None:
    print()
    print("NeuroSight Local UI")
    print(f"Backend : {BACKEND_URL}")
    print(f"Key     : {API_KEY[:6]}...")
    print(f"Local   : http://{HOST}:{PORT}")
    print()
    server = ThreadingHTTPServer((HOST, PORT), NeuroSightHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

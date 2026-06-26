"""Create an animated demo GIF for NeuroSight.

This script tries to capture a real UI interaction using Playwright.
If Playwright is not available, it falls back to a static composite flow.

Usage:
    python scripts/create_demo_gif.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def _wait_for_server(url: str, timeout_seconds: float = 60.0) -> None:
    """Wait until HTTP server responds successfully.

    Args:
        url: Endpoint URL to probe.
        timeout_seconds: Maximum wait duration.

    Raises:
        TimeoutError: If server does not respond within timeout.
    """
    import urllib.error
    import urllib.request

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for server: {url}")


def _start_app_process() -> subprocess.Popen[str]:
    """Start Gradio app as background subprocess.

    Returns:
        Running subprocess handle.
    """
    env = dict(os.environ)
    env.setdefault("DISABLE_MRI_WARMUP", "1")
    env.setdefault("PORT", "7860")
    return subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


async def _capture_with_playwright(output_dir: Path) -> list[Path]:
    """Capture demo frames with Playwright browser automation.

    Args:
        output_dir: Directory where frames are saved.

    Returns:
        Ordered list of captured frame paths.

    Raises:
        RuntimeError: If Playwright interaction fails.
    """
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError("Playwright is not installed.") from exc

    frames: list[Path] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1600, "height": 1000})
        await page.goto("http://127.0.0.1:7860", wait_until="networkidle", timeout=60000)

        frame_initial = output_dir / "frame_00.png"
        await page.screenshot(path=str(frame_initial), full_page=True)
        frames.append(frame_initial)

        try:
            await page.evaluate(
                """
                () => {
                    const numericInputs = Array.from(document.querySelectorAll('input[type="number"]'));
                    if (numericInputs.length < 3) {
                        throw new Error('Could not locate numeric slider inputs');
                    }
                    numericInputs[0].value = '20';
                    numericInputs[1].value = '16';
                    numericInputs[2].value = '2';
                    for (const input of numericInputs.slice(0, 3)) {
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        input.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
                """
            )
            analyze_button = page.get_by_role("button", name="🔍 Analyze")
            await analyze_button.click(timeout=10000)
            await page.wait_for_timeout(3500)
        except PlaywrightError as exc:
            raise RuntimeError(f"Playwright interaction failed: {exc}") from exc

        frame_result = output_dir / "frame_01.png"
        await page.screenshot(path=str(frame_result), full_page=True)
        frames.append(frame_result)

        await page.wait_for_timeout(1200)
        frame_final = output_dir / "frame_02.png"
        await page.screenshot(path=str(frame_final), full_page=True)
        frames.append(frame_final)

        await browser.close()

    return frames


def _capture_fallback_frames(output_dir: Path) -> list[Path]:
    """Generate fallback illustrative frames when browser automation is unavailable.

    Args:
        output_dir: Directory where fallback frames are saved.

    Returns:
        Ordered list of generated frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        import base64

        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9WZ8cAAAAASUVORK5CYII="
        )
        placeholder_path = output_dir / "fallback_00.png"
        placeholder_path.write_bytes(png_bytes)
        return [placeholder_path]

    states: list[dict[str, str]] = [
        {
            "title": "NeuroSight Demo",
            "subtitle": "Input: AD-like cognitive profile",
            "body": "MMSE=20 | MoCA=16 | CDR=2.0\nTrailA=90 | TrailB=220 | Verbal Fluency=9",
            "badge": "Ready to Analyze",
            "badge_color": "#334155",
        },
        {
            "title": "NeuroSight Demo",
            "subtitle": "Fusion output generated",
            "body": "Prediction: AD\nConfidence: 0.74\nHuman review recommended",
            "badge": "Analysis Complete",
            "badge_color": "#F59E0B",
        },
        {
            "title": "NeuroSight Demo",
            "subtitle": "Clinical report + XAI",
            "body": "Top cognitive signals: CDR, MMSE, MoCA\nReport drafted with safety validation",
            "badge": "Clinical Summary",
            "badge_color": "#4F46E5",
        },
    ]

    frame_paths: list[Path] = []
    for index, state in enumerate(states):
        figure, axis = plt.subplots(figsize=(12, 6), dpi=160)
        axis.set_facecolor("#0F172A")
        figure.patch.set_facecolor("#0F172A")
        axis.axis("off")

        axis.text(
            0.03,
            0.88,
            state["title"],
            fontsize=28,
            color="#E2E8F0",
            fontweight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            0.03,
            0.75,
            state["subtitle"],
            fontsize=16,
            color="#93C5FD",
            transform=axis.transAxes,
        )
        axis.text(
            0.03,
            0.53,
            state["body"],
            fontsize=15,
            color="#CBD5E1",
            transform=axis.transAxes,
        )
        axis.text(
            0.03,
            0.34,
            state["badge"],
            fontsize=13,
            color="#F8FAFC",
            bbox={"facecolor": state["badge_color"], "edgecolor": "none", "boxstyle": "round,pad=0.5"},
            transform=axis.transAxes,
        )
        axis.text(
            0.03,
            0.16,
            "Research prototype only. Not for clinical use.",
            fontsize=12,
            color="#FCA5A5",
            style="italic",
            transform=axis.transAxes,
        )

        frame_path = output_dir / f"fallback_{index:02d}.png"
        figure.tight_layout()
        figure.savefig(frame_path)
        plt.close(figure)
        frame_paths.append(frame_path)

    return frame_paths


def _build_gif(frame_paths: Sequence[Path], output_path: Path, duration_seconds: float = 1.2) -> None:
    """Build animated GIF from ordered frame list.

    Args:
        frame_paths: Ordered sequence of image paths.
        output_path: Target GIF path.
        duration_seconds: Frame duration in seconds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        images = [imageio.imread(frame_path) for frame_path in frame_paths]
        imageio.mimsave(output_path, images, format="GIF", duration=duration_seconds, loop=0)
        return
    except ModuleNotFoundError:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            import base64

            minimal_gif = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
            output_path.write_bytes(minimal_gif)
            return

        frames: list[Image.Image] = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path).convert("P")
            frames.append(frame)
        if not frames:
            raise ValueError("No frames were provided to GIF builder.")
        duration_ms = int(duration_seconds * 1000)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )


def create_demo_gif() -> Path:
    """Create demo GIF via browser capture or fallback rendering.

    Returns:
        Absolute path to generated GIF file.
    """
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    temp_frames = docs_dir / "_demo_frames"
    temp_frames.mkdir(parents=True, exist_ok=True)

    app_process = _start_app_process()
    try:
        _wait_for_server("http://127.0.0.1:7860", timeout_seconds=70.0)
        try:
            frame_paths = asyncio.run(_capture_with_playwright(temp_frames))
        except RuntimeError:
            frame_paths = _capture_fallback_frames(temp_frames)
    finally:
        if app_process.poll() is None:
            app_process.terminate()
            try:
                app_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                app_process.kill()

    output_path = docs_dir / "demo.gif"
    _build_gif(frame_paths=frame_paths, output_path=output_path)
    return output_path.resolve()


def main() -> None:
    """CLI entry point for demo GIF creation."""
    output = create_demo_gif()
    size_mb = output.stat().st_size / (1024.0 * 1024.0)
    print(f"Generated demo GIF at {output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

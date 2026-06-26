"""Execute NeuroSight demo notebook and persist pre-run outputs.

Usage:
    python scripts/prerun_notebook.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_notebook(notebook_path: Path) -> Any:
    """Load notebook object from disk.

    Args:
        notebook_path: Path to notebook file.

    Returns:
        Parsed notebook object.
    """
    try:
        import nbformat
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nbformat is required to load notebooks.") from exc

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as handle:
        return nbformat.read(handle, as_version=4)


def _execute_notebook(
    notebook: Any,
    notebook_path: Path,
    timeout_seconds: int = 900,
) -> Any:
    """Execute notebook with nbclient.

    Args:
        notebook: Notebook object.
        notebook_path: Path to notebook file.
        timeout_seconds: Maximum execution timeout.

    Returns:
        Executed notebook object.
    """
    try:
        from nbclient import NotebookClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nbclient is required to execute notebooks.") from exc

    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    return client.execute()


def _save_notebook(notebook: Any, notebook_path: Path) -> None:
    """Save executed notebook to disk.

    Args:
        notebook: Notebook object.
        notebook_path: Destination path.
    """
    import nbformat

    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)


def _generate_figures() -> int:
    """Generate docs figures used by notebook and report.

    Returns:
        Number of generated figures.
    """
    from scripts.generate_all_figures import generate_all_figures

    generated = generate_all_figures()
    return len(generated)


def main() -> None:
    """CLI entry point for notebook pre-run workflow."""
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "neurosight_demo.ipynb"

    notebook = _load_notebook(notebook_path)
    executed = _execute_notebook(notebook, notebook_path)
    _save_notebook(executed, notebook_path)
    figure_count = _generate_figures()

    print(f"Executed notebook and embedded outputs: {notebook_path}")
    print(f"Generated {figure_count} figures in docs/figures/")


if __name__ == "__main__":
    main()

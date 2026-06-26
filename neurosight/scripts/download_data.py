import numpy as np
from pathlib import Path


def generate_synthetic_data(output_dir="data/raw", n_samples=2):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic data in", output_dir)

    expected_files = {
        f"syn_{modality}_{i:04d}.npy"
        for i in range(n_samples)
        for modality in ("mri", "eeg", "cog")
    }
    for stale_file in output_path.glob("syn_*_*.npy"):
        if stale_file.name not in expected_files:
            stale_file.unlink()

    rng = np.random.default_rng(42)
    for i in range(n_samples):
        mri = rng.normal(size=(16, 16, 16)).astype(np.float32)
        np.save(output_path / f"syn_mri_{i:04d}.npy", mri)
        eeg = rng.normal(size=(19, 256)).astype(np.float32)
        np.save(output_path / f"syn_eeg_{i:04d}.npy", eeg)
        cog = rng.random(8).astype(np.float32)
        np.save(output_path / f"syn_cog_{i:04d}.npy", cog)
    print("Synthetic data generated.")

if __name__ == "__main__":
    generate_synthetic_data()

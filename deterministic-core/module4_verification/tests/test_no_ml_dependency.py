"""The learned/deterministic boundary, asserted as a test as well as in CI.

If this ever fails, the verification argument in the paper no longer holds — see
docs/adr/0003-deterministic-core-isolation.md.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

FORBIDDEN = (
    "torch",
    "tensorflow",
    "jax",
    "sklearn",
    "scikit-learn",
    "transformers",
    "onnxruntime",
)


def test_pyproject_declares_no_ml_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    deps = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    offenders = [d for d in deps if any(f in d.lower() for f in FORBIDDEN)]
    assert not offenders, f"deterministic-core must declare no ML dependency, found: {offenders}"


def test_no_ml_module_is_importable_at_runtime() -> None:
    """Asserts that importing verification in a clean process loads zero ML dependencies."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; import verification; "
            "forbidden = ('torch', 'tensorflow', 'jax', 'sklearn', 'transformers', 'onnxruntime'); "
            "offenders = [m for m in forbidden if m in sys.modules]; "
            "assert not offenders, f'Runtime leakage: {offenders}'"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Runtime import leakage: {res.stderr}"

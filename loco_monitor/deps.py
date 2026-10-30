"""Dependency check / optional auto-install (set AUTO_INSTALL_DEPS=1)."""
from __future__ import annotations

import sys
from .config import AUTO_INSTALL_DEPS, ENABLE_VLM


def _ensure_package(module_name: str, package_spec: str) -> None:
    try:
        __import__(module_name)
    except Exception:
        if not AUTO_INSTALL_DEPS:
            raise RuntimeError(
                f"Missing dependency '{module_name}'. Install '{package_spec}' or set AUTO_INSTALL_DEPS=1."
            )
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])


def ensure_dependencies() -> None:
    """Check (or auto-install) required packages. Runs once when the package is imported."""
    for _module, _package in [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("fpdf", "fpdf2==2.8.1"),
        ("ultralytics", "ultralytics==8.2.92"),
        ("mediapipe", "mediapipe==0.10.18"),
        ("fastapi", "fastapi"),
        ("jinja2", "jinja2"),
        ("uvicorn", "uvicorn[standard]"),
        ("multipart", "python-multipart"),
    ]:
        _ensure_package(_module, _package)
    if ENABLE_VLM:
        for _module, _package in [
            ("torch", "torch"),
            ("transformers", "transformers>=4.52.0"),
            ("accelerate", "accelerate>=0.34.0"),
            ("bitsandbytes", "bitsandbytes>=0.44.0"),
            ("qwen_vl_utils", "qwen-vl-utils>=0.0.8"),
        ]:
            _ensure_package(_module, _package)

"""
===============================================================================
SETUP_ENV.PY - Tu dong khoi tao venv311 va cai dat dependencies
===============================================================================
Script nay:
    1. Tu dong tao virtual environment 'venv311' (Python 3.11 - tuong thich
       voi TensorFlow de export TFLite).
    2. Tu dong cai dat toan bo dependencies tu requirements.txt.
    3. In ra huong dan su dung tiep theo.

Su dung:
    python setup_env.py              # Tu dong: tim Python 3.11, tao venv311
    python setup_env.py --recreate   # Xoa venv311 cu va tao moi
    python setup_env.py --no-venv    # Cai truc tiep vao Python hien tai
    python setup_env.py --python 3.12 # Chi dinh Python interpreter

Luu y:
    - Python 3.13+ KHONG duoc TensorFlow ho tro nen KHONG the export TFLite.
    - Nen dung Python 3.11 hoac 3.12.
    - Sau khi setup xong, moi lenh train/export deu phai chay trong venv311.
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT: Path = Path(__file__).resolve().parent
VENV_DIR: Path = WORKSPACE_ROOT / "venv311"
REQUIREMENTS: Path = WORKSPACE_ROOT / "requirements.txt"


def log(level: str, msg: str) -> None:
    colors = {
        "INFO": "\033[96m",
        "OK": "\033[92m",
        "WARN": "\033[93m",
        "ERR": "\033[91m",
        "END": "\033[0m",
    }
    c = colors.get(level, "")
    print(f"{c}[{level}]{colors['END']} {msg}")


def banner(msg: str) -> None:
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)
    print()


def find_python_311() -> str | None:
    """
    Tim Python 3.11 (hoac 3.12) trong he thong.
    Tra ve duong dan executable hoac None neu khong tim thay.
    """
    # 1) Thu cac duong dan pho bien tren Windows
    win_candidates = [
        r"C:\Python311\python.exe",
        r"C:\Python312\python.exe",
        r"C:\Program Files\Python311\python.exe",
        r"C:\Program Files\Python312\python.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\python.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Python311\python.exe"),
    ]

    for path in win_candidates:
        if os.path.isfile(path):
            try:
                out = subprocess.check_output(
                    [path, "--version"], text=True, timeout=5
                )
                if "3.11" in out or "3.12" in out:
                    log("OK", f"Tim thay Python: {path} ({out.strip()})")
                    return path
            except Exception:
                continue

    # 2) Thu py launcher (Windows)
    if os.name == "nt":
        for minor in ("3.11", "3.12"):
            try:
                out = subprocess.check_output(
                    ["py", f"-{minor}", "--version"], text=True, timeout=5,
                    stderr=subprocess.STDOUT,
                )
                if "3.11" in out or "3.12" in out:
                    log("OK", f"Tim thay qua py launcher: py -{minor} ({out.strip()})")
                    return f"py -{minor}"
            except Exception:
                continue

    # 3) Thu cac duong dan Unix
    unix_candidates = [
        "/usr/bin/python3.11", "/usr/bin/python3.12",
        "/usr/local/bin/python3.11", "/usr/local/bin/python3.12",
    ]
    for path in unix_candidates:
        if os.path.isfile(path):
            return path

    return None


def create_venv(python_exe: str, recreate: bool = False) -> Path:
    """Tao venv311. Tra ve duong den python.exe trong venv."""
    if VENV_DIR.exists():
        if recreate:
            log("WARN", f"Xoa venv cu: {VENV_DIR}")
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        else:
            log("INFO", f"venv311 da ton tai: {VENV_DIR}")
            py = venv_python()
            if py.exists():
                return py

    log("INFO", f"Tao venv311 (co the mat 5-10 giay)...")
    if os.path.isfile(python_exe):
        cmd = [python_exe]
    else:
        cmd = python_exe.split()
    subprocess.check_call(cmd + ["-m", "venv", str(VENV_DIR)])
    log("OK", f"Da tao venv311: {VENV_DIR}")
    return venv_python()


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements(python_exe: Path) -> None:
    """Cai dat requirements.txt vao venv."""
    if not REQUIREMENTS.exists():
        log("ERR", f"Khong tim thay requirements.txt tai: {REQUIREMENTS}")
        sys.exit(1)

    log("INFO", "Upgrade pip...")
    subprocess.check_call(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
        stdout=subprocess.DEVNULL,
    )

    log("INFO", f"Cai dat dependencies tu {REQUIREMENTS.name} (co the mat 5-15 phut)...")
    # Cai tung phan de biet package nao loi
    reqs = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    packages = [
        line.strip() for line in reqs
        if line.strip() and not line.strip().startswith("#")
    ]

    failed = []
    for pkg in packages:
        # Comment phia sau ten package (vd: ultralytics>=8.3.0 # comment)
        clean = pkg.split("#", 1)[0].strip()
        if not clean:
            continue
        print(f"  -> {clean}")
        try:
            subprocess.check_call(
                [str(python_exe), "-m", "pip", "install", clean],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            failed.append(clean)
            log("WARN", f"  Khong cai duoc: {clean} (tiep tuc...)")

    if failed:
        log("WARN", f"Cac package cai loi: {failed}")
    else:
        log("OK", "Da cai dat tat ca dependencies!")

    # Verify
    log("INFO", "Kiem tra cac package chinh...")
    for mod in ("ultralytics", "torch", "cv2", "numpy", "yaml"):
        try:
            subprocess.check_call(
                [str(python_exe), "-c", f"import {mod}; print(f'  {mod} OK')"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError:
            log("WARN", f"  Module '{mod}' chua duoc cai (mot so tinh nang se khong hoat dong)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Khoi tao venv311 va cai dat dependencies cho nail_ai_workspace.",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Xoa venv311 cu va tao moi.",
    )
    parser.add_argument(
        "--no-venv", action="store_true",
        help="Cai dat truc tiep vao Python hien tai (khong tao venv).",
    )
    parser.add_argument(
        "--python", type=str, default=None,
        help="Chi dinh Python interpreter (vd: 'C:/Python311/python.exe' hoac '3.11').",
    )
    args = parser.parse_args()

    banner("NAIL AI WORKSPACE - SETUP MOI TRUONG")

    if args.no_venv:
        log("INFO", f"Cai truc tiep vao Python: {sys.executable}")
        install_requirements(Path(sys.executable))
        print_next_steps(use_venv=False)
        return 0

    # Tim Python 3.11 / 3.12
    if args.python:
        python_exe = args.python
        log("INFO", f"Su dung Python chi dinh: {python_exe}")
    else:
        log("INFO", "Tim Python 3.11/3.12 trong he thong...")
        python_exe = find_python_311()

    if not python_exe:
        log("ERR", "Khong tim thay Python 3.11 hoac 3.12!")
        log("ERR", "Vui long:")
        log("ERR", "  1. Tai Python 3.11 tu https://www.python.org/downloads/")
        log("ERR", "  2. Hoac trong VS Code mo Command Palette > 'Python: Select Interpreter'")
        log("ERR", "  3. Hoac chay: py -3.11 -m venv venv311")
        return 1

    # Tao venv + cai dat
    py = create_venv(python_exe, recreate=args.recreate)
    install_requirements(py)
    print_next_steps(use_venv=True, python_exe=py)
    return 0


def print_next_steps(use_venv: bool, python_exe: Path | None = None) -> None:
    print()
    print("=" * 70)
    print("  HOAN TAT!")
    print("=" * 70)
    print()
    if use_venv and python_exe:
        if os.name == "nt":
            print("  Kich hoat venv:")
            print("    .\\venv311\\Scripts\\activate")
            print()
            print("  Hoac go truc tiep (khong can activate):")
            print(f"    .\\venv311\\Scripts\\python.exe train.py")
        else:
            print("  Kich hoat venv:")
            print("    source venv311/bin/activate")
            print()
            print("  Hoac go truc tiep (khong can activate):")
            print(f"    {python_exe} train.py")
    else:
        print("  Cai dat truc tiep vao Python hien tai.")
        print("  Chay training ngay:")
        print("    python train.py")
    print()
    print("  Buoc tiep theo:")
    print("    1. python train.py           # Train (hoac train_finetune_5class.py)")
    print("    2. python export_all.py      # Export ONNX + TFLite cho mobile")
    print("    3. python test_inference.py  # Test model voi anh bat ky")
    print()
    print("  Xem huong dan chi tiet: README.md")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

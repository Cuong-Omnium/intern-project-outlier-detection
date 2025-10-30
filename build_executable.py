"""
Build script for creating standalone executable.
"""

import shutil
from pathlib import Path

import PyInstaller.__main__


def build():
    """Build the executable."""

    print("=" * 80)
    print("Building Account Outlier Detection Executable")
    print("=" * 80)

    # Clean previous builds
    for folder in ["build", "dist"]:
        if Path(folder).exists():
            print(f"Cleaning {folder}/...")
            shutil.rmtree(folder)

    # PyInstaller arguments
    args = [
        "app/main.py",  # Entry point
        "--name=OutlierDetectionApp",
        "--onedir",  # Create a folder (not single file - faster startup)
        "--windowed",  # No console window
        "--add-data=src;src",  # Include source code
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=sklearn",
        "--hidden-import=plotly",
        "--hidden-import=scipy",
        "--hidden-import=yaml",
        "--hidden-import=pydantic",
        "--collect-all=streamlit",
        "--collect-all=plotly",
        "--noconfirm",
        "--clean",
    ]

    print("\nRunning PyInstaller...")
    PyInstaller.__main__.run(args)

    print("\n" + "=" * 80)
    print("✅ Build Complete!")
    print(f"📁 Executable location: dist/OutlierDetectionApp/")
    print("=" * 80)


if __name__ == "__main__":
    build()

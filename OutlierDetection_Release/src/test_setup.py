"""Test that all required packages are installed."""

import sys


def test_imports():
    """Test all critical imports."""
    packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("sklearn.cluster", "HDBSCAN"),
        ("plotly.graph_objects", "go"),
        ("streamlit", "st"),
        ("pydantic", "BaseModel"),
    ]

    print("Testing imports...")
    print(f"Python: {sys.version}")
    print(f"Location: {sys.executable}\n")

    for module, item in packages:
        try:
            if item == "HDBSCAN":
                from sklearn.cluster import HDBSCAN

                print(f"✓ sklearn.cluster.HDBSCAN")
            else:
                exec(f"import {module}")
                print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False

    print("\n✅ All packages installed correctly!")
    return True


if __name__ == "__main__":
    test_imports()

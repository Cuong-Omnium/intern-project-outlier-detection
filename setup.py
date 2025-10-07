"""Setup configuration for outlier-detection-app."""

from setuptools import find_packages, setup

setup(
    name="outlier-detection-app",
    version="0.1.0",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
)

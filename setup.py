"""
Setup script for the Large Data Models (LDMs) package.

Allows installing the project as an editable package:
    pip install -e .

Enables imports such as:
    from src.models import MMTTTransformerEncoder, PeriodicNumericalEmbedding
    from src.data import SyntheticTransactionGenerator, LDMDataModule
"""

from setuptools import setup, find_packages

setup(
    name="ldm-research-lab",
    version="0.2.0",
    description="Large Data Models (LDMs) — Self-Supervised Foundation Models for Structured & Transactional Data",
    author="Vinicius Caridá",
    author_email="vfcarida@gmail.com",
    url="https://github.com/vfcarida/Large-Data-Models-LDMs-",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.0",
        "torchmetrics>=1.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "scikit-learn>=1.3.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "tracking": ["wandb>=0.16.0", "tensorboard>=2.15.0"],
        "viz": ["matplotlib>=3.8.0", "seaborn>=0.13.0"],
        "peft": ["peft>=0.7.0", "transformers>=4.36.0", "accelerate>=0.25.0"],
        "gpu": ["cudf-cu12"],
        "dev": ["pytest>=7.4.0", "hypothesis>=6.90.0", "ruff>=0.3.0", "mypy>=1.8.0"],
    },
    entry_points={
        "console_scripts": [
            "ldm-pipeline=src.main:main",
        ],
    },
)

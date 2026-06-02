"""
Setup script for the Large Data Models (LDMs) package.

Allows installing the project as an editable package:
    pip install -e .

This enables imports like:
    from src.models import MMTTTransformerEncoder
    from src.data import SyntheticTransactionGenerator
"""

from setuptools import setup, find_packages

setup(
    name="ldm-research-lab",
    version="0.1.0",
    description="Large Data Models (LDMs) Research Lab — Foundation models for structured/tabular data",
    author="LDM Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.0",
        "torchmetrics>=1.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "tracking": ["wandb>=0.16.0", "tensorboard>=2.15.0"],
        "viz": ["matplotlib>=3.8.0", "seaborn>=0.13.0"],
        "peft": ["peft>=0.7.0", "transformers>=4.36.0", "accelerate>=0.25.0"],
        "gpu": ["cudf-cu12"],
    },
)

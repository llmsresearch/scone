#!/usr/bin/env python
"""Setup script for SCONE."""

import os
from setuptools import setup, find_packages

# Get package version
with open(os.path.join("scone", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

# Get long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Get requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if not line.startswith("#") and line.strip()]

# Define extras
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "black>=22.3.0",
        "isort>=5.10.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
    "quantization": [
        "bitsandbytes>=0.35.0",
    ],
    "visualization": [
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
        "wandb>=0.15.0",
    ],
    "azure": [
        "azure-storage-blob>=12.0.0",
        "azureml-core>=1.40.0",
    ],
}

# Add all extras to "all"
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="scone",
    version=version,
    description="SCONE: Scaling Embedding Layers in Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SCONE Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/scone",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras_require,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, language-models, embeddings, machine-learning, deep-learning",
) 
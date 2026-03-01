"""
setup.py — TorchScript Custom Op Parser
========================================
Setuptools-based build configuration.
Supports both `pip install -e .` (editable/dev) and `pip install .` (production).
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    # Package metadata
    name="torchscript-custom-op-parser",
    version="1.0.0",
    description=(
        "Walks TorchScript IR graphs — including custom operators — "
        "and generates self-contained C++ inference headers (mini-SOFIE)."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Rahul",
    author_email="rahulgunwani.study@gmail.com",
    url="https://github.com/rahul/torchscript-custom-op-parser",
    license="MIT",

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.9",

    install_requires=[
        "torch>=2.0",
    ],
    extras_require={
        "dev": [
            "torch>=2.0",
            "numpy>=1.24",
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "minimal": [],
    },

    #CLI entry points─
    entry_points={
        "console_scripts": [
            "ts-parser=ts_parser:main",
        ],
    },

    #  PyPI classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Compilers",
    ],
    keywords=[
        "torchscript", "pytorch", "inference", "code-generation",
        "custom-operators", "sofie", "cern", "hep", "c++",
    ],

    # Package data
    include_package_data=True,
    zip_safe=False,
)
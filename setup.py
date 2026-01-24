"""
Setup script for Cats vs Dogs MLOps Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cats-dogs-classifier",
    version="1.0.0",
    author="MLOps Student",
    author_email="student@example.com",
    description="MLOps Pipeline for Cats vs Dogs Binary Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/mlops-cats-dogs-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=src.models.train:main",
            "run-api=src.api.main:main",
        ],
    },
)

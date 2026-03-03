"""Legacy setup.py for backward compatibility with older pip versions."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="utils-yolo-validation",
    version="0.1.0",
    author="Data Validation Team",
    description="Professional toolkit for YOLO pose dataset validation and manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/utils-yolo-validation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "open-clip-torch>=2.0.0",
        "faiss-cpu>=1.7.0",
        "tqdm>=4.50.0",
        "openpyxl>=3.0.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "cuda": ["faiss-gpu>=1.7.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "mypy>=0.910",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.0.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn[standard]>=0.21.0",
        ],
    },
)

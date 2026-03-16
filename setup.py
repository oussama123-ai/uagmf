from setuptools import setup, find_packages

setup(
    name="uagmf",
    version="1.0.0",
    description=(
        "UAG-MF: Uncertainty-Aware Generative Multimodal Fusion for "
        "continuous pain estimation in non-verbal patients under clinical occlusions."
    ),
    author="Oussama El Othmani, Sami Naouali",
    author_email="salnawali@kfu.edu.sa",
    url="https://github.com/oussama123-ai/uagmf",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "albumentations>=1.3.0",
        "PyYAML>=6.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "opacus>=1.4.0",
        "pingouin>=0.5.3",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)

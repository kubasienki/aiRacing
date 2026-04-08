from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vdrift_rl",
    version="0.1.0",
    author="Kuba Sienki",
    author_email="your.email@example.com",  # TODO: Update with your email
    description="Reinforcement Learning environment for VDrift racing simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vdrift-rl",  # TODO: Update with your repo
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.8",
    install_requires=[
        'gym>=0.26.0,<0.27.0',
        'numpy>=1.19.0',
        'redis>=4.0.0',
    ],
    extras_require={
        'training': [
            'stable-baselines3>=2.0.0',
            'torch>=1.10.0',
            'tensorboard>=2.10.0',
        ],
        'visualization': [
            'matplotlib>=3.3.0',
            'pygame>=2.0.0',
        ],
        'dev': [
            'pytest>=6.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
        ],
    },
    keywords='reinforcement-learning racing simulation vdrift gym openai-gym',
    project_urls={
        'Documentation': 'https://github.com/yourusername/vdrift-rl/tree/master/docs',
        'Source': 'https://github.com/yourusername/vdrift-rl',
        'Tracker': 'https://github.com/yourusername/vdrift-rl/issues',
    },
)

#!/usr/bin/env python3
"""
Setup script for CoCoDe.
"""

from setuptools import setup, find_packages

with open("cocode/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="cocode",
    version="0.1.0",
    description="A code-focused AI assistant powered by GPT-4.1",
    author="CoCoDe Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cocode=cocode.main:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
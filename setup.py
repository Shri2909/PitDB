from setuptools import find_packages, setup

setup(
    name="pitdb",
    version="0.1.0",
    python_requires=">=3.9",
    packages=find_packages(include=["src", "src.*"]),
)

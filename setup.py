from setuptools import find_packages, setup

# Supported interpreters: Python 3.11 and newer.  Verified with the exact pins
# in requirements.txt on 3.11 and with the ranges below on 3.14 (3.13 was not
# run directly).  The floor comes from the optional extras (pillow 13 needs
# >=3.11).  The numpy, pandas and matplotlib ranges let pip pick builds that
# exist for newer interpreters.  sqlglot is held to 23.9.x because the SQL
# parser is tested against that release (23.17 changes statement handling).
#
# Only the `src` package is shipped.  `benchmarks/` and `charts/` are
# source-checkout tools (they read data/ and write benchmarks/results/ relative
# to the checkout) and must not be installed under generic top-level names.
setup(
    name="pitdb",
    version="0.1.0",
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.26.4,<3",
        "pandas>=2.2.1,<3",
        "sqlglot>=23.9.0,<23.10",
    ],
    extras_require={
        "benchmarks": [
            "matplotlib>=3.8.3,<4",
            "psutil>=5.9.8",
            "pillow>=13.0.0",
        ],
        "download": [
            "yfinance>=1.5.1",
            "certifi>=2026.10.23",
            "soupsieve>=2.9",
            "urllib3>=2.8.0",
        ],
        "dev": [
            "pytest>=9.0.3",
            "hypothesis>=6.99.6",
            "ruff>=0.15.22",
            # tests/test_packaging.py evaluates setup.py; venvs on Python
            # 3.12+ no longer bundle setuptools.
            "setuptools>=61",
        ],
    },
)

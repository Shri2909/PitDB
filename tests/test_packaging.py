"""Distribution metadata contracts, checked without downloading packages."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

pytest.importorskip("setuptools")
from setuptools import find_packages  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Requires-Python metadata of every pinned release in requirements.txt, copied
# from PyPI (https://pypi.org/pypi/<name>/<version>/json) when the pins were
# chosen. Kept as data so the compatibility check below runs offline; update
# an entry whenever its pin changes.
PIN_REQUIRES_PYTHON = {
    "numpy==1.26.4": ">=3.9",
    "pandas==2.2.1": ">=3.9",
    "sqlglot==23.9.0": ">=3.7",
    "matplotlib==3.8.3": ">=3.9",
    "psutil==5.9.8": ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    "pillow==13.0.0": ">=3.11",
    "pytest==9.0.3": ">=3.10",
    "hypothesis==6.99.6": ">=3.8",
    "ruff==0.15.22": ">=3.7",
    "setuptools==79.0.1": ">=3.9",
    "yfinance==1.5.1": "",
    "certifi==2026.10.23": ">=3.7",
    "soupsieve==2.9": ">=3.10",
    "urllib3==2.8.0": ">=3.10",
}
# numpy/pandas/matplotlib wheels for these pins stop at cp312, so the exact-pin
# environment (requirements.txt / `make setup`) is a 3.11-3.12 environment.
PINNED_ENV_PYTHONS = [Version("3.11"), Version("3.12")]


@pytest.fixture(scope="module")
def package_metadata() -> dict[str, object]:
    # Evaluate the actual setup declaration while suppressing command execution.
    with patch("setuptools.setup") as setup:
        exec(compile((PROJECT_ROOT / "setup.py").read_text(), "setup.py", "exec"), {})
    return setup.call_args.kwargs


def _requirements_pins() -> dict[str, Requirement]:
    pins = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            requirement = Requirement(line.strip())
            pins[requirement.name.lower()] = requirement
    return pins


def _declared_requirements(
    package_metadata: dict[str, object],
) -> dict[str, Requirement]:
    declared = {}
    groups = [package_metadata["install_requires"]]
    groups.extend(package_metadata["extras_require"].values())
    for group in groups:
        for text in group:
            requirement = Requirement(text)
            declared[requirement.name.lower()] = requirement
    return declared


def _python_floor(package_metadata: dict[str, object]) -> Version:
    floors = [
        Version(spec.version)
        for spec in SpecifierSet(package_metadata["python_requires"])
        if spec.operator == ">="
    ]
    assert len(floors) == 1
    return floors[0]


def _classifier_pythons(package_metadata: dict[str, object]) -> list[Version]:
    return [
        Version(match.group(1))
        for classifier in package_metadata["classifiers"]
        if (
            match := re.fullmatch(
                r"Programming Language :: Python :: (3\.\d+)", classifier
            )
        )
    ]


def test_distribution_declares_core_runtime_dependencies(
    package_metadata: dict[str, object],
) -> None:
    core = {Requirement(text).name for text in package_metadata["install_requires"]}
    assert core == {"numpy", "pandas", "sqlglot"}


def test_every_distribution_dependency_has_a_satisfying_development_pin(
    package_metadata: dict[str, object],
) -> None:
    declared = _declared_requirements(package_metadata)
    pins = _requirements_pins()
    assert set(declared) == set(pins)
    for name, requirement in declared.items():
        pin = pins[name]
        assert [spec.operator for spec in pin.specifier] == ["=="], pin
        (spec,) = pin.specifier
        assert requirement.specifier.contains(spec.version, prereleases=True), (
            f"{pin} violates the range declared in setup.py: {requirement}"
        )


def test_every_pin_has_recorded_requires_python_metadata() -> None:
    pinned = {str(requirement) for requirement in _requirements_pins().values()}
    assert pinned == set(PIN_REQUIRES_PYTHON)


def test_python_floor_satisfies_every_pinned_dependency(
    package_metadata: dict[str, object],
) -> None:
    supported = [_python_floor(package_metadata), *PINNED_ENV_PYTHONS]
    for pin, requires in PIN_REQUIRES_PYTHON.items():
        for version in supported:
            assert SpecifierSet(requires).contains(version), (
                f"{pin} requires Python {requires or '(unspecified)'}, but "
                f"python_requires allows {version}"
            )


def test_declared_floor_is_the_lowest_python_the_extras_allow(
    package_metadata: dict[str, object],
) -> None:
    """The floor must not be raised/lowered without revisiting the pins."""

    floor = _python_floor(package_metadata)
    required_floors = []
    for requires in PIN_REQUIRES_PYTHON.values():
        for spec in SpecifierSet(requires):
            if spec.operator == ">=":
                required_floors.append(Version(spec.version))
    assert floor == max(required_floors)


def test_python_requires_classifiers_and_tooling_agree(
    package_metadata: dict[str, object],
) -> None:
    floor = _python_floor(package_metadata)
    classifiers = _classifier_pythons(package_metadata)
    assert classifiers, "declare supported Python versions as classifiers"
    assert min(classifiers) == floor
    assert all(
        SpecifierSet(package_metadata["python_requires"]).contains(v)
        for v in classifiers
    )
    assert set(PINNED_ENV_PYTHONS) <= set(classifiers)

    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    target = pyproject["tool"]["ruff"]["target-version"]
    assert target == f"py{floor.major}{floor.minor}"

    makefile = (PROJECT_ROOT / "Makefile").read_text()
    assert f"PYTHON_FLOOR := {floor}" in makefile
    assert "python3.11 -m venv" not in makefile


def test_distribution_ships_only_the_src_package(
    package_metadata: dict[str, object],
) -> None:
    packages = set(package_metadata["packages"])
    assert {"src", "src.replay", "src.store", "src.query"} <= packages
    # Source-checkout tools must not be installed into site-packages as
    # generic top-level packages.
    assert {name.split(".")[0] for name in packages} == {"src"}
    assert packages <= set(find_packages(where=PROJECT_ROOT))

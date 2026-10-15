"""Cythonize src/ business logic into compiled extension modules.

Excludes __init__.py (kept plain so packages import normally), app.py
(thin Streamlit entrypoint, stays interpreted so `streamlit run` can
target it directly), and pydantic BaseModel modules that define
validators/computed methods on the model class: Cython turns those
methods into `cyfunction` objects, and pydantic's class-namespace
inspector rejects unannotated cyfunction attributes at import time
(pydantic.errors.PydanticUserError). Config-only BaseModel classes
(no methods) compile fine and are not excluded.
"""
from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup

SRC = Path("src")
EXCLUDE_FILES = {"__init__.py", "app.py"}
EXCLUDE_PATHS = {
    SRC / "models" / "derived.py",
    SRC / "models" / "evidence.py",
    SRC / "models" / "mcts.py",
    SRC / "models" / "section.py",
}

sources = [
    str(p)
    for p in SRC.rglob("*.py")
    if p.name not in EXCLUDE_FILES and p not in EXCLUDE_PATHS
]

setup(
    name="mvp-v2-compiled",
    ext_modules=cythonize(
        sources,
        compiler_directives={"language_level": "3"},
        build_dir="build",
    ),
)

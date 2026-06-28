"""Repo-root conftest: make the ``src/`` package importable without an install.

Until ``pip install -e .`` (pyproject, bead 1ug.g9v) is wired, prepend ``src/``
so tests and scripts can ``import gpusitter`` (src-layout).
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

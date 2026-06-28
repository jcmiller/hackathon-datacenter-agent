"""Fast smoke test: the installed package and its core subpackages import cleanly.

This is the canary for packaging/wiring breakage — it runs in milliseconds and
fails loudly if the editable install is broken or any core subpackage has an
import-time error, before the heavier behavioral suite even starts.
"""

import importlib

import pytest

CORE_SUBPACKAGES = ["agent", "detection", "telemetry", "rca", "domain"]


@pytest.mark.smoke
def test_package_imports():
    import gpusitter

    assert gpusitter.__name__ == "gpusitter"


@pytest.mark.smoke
@pytest.mark.parametrize("sub", CORE_SUBPACKAGES)
def test_core_subpackage_imports(sub):
    module = importlib.import_module(f"gpusitter.{sub}")
    assert module.__name__ == f"gpusitter.{sub}"

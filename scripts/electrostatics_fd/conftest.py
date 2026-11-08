"""pytest configuration for the ``electrostatics_fd`` suite.

Two jobs, both of which have to work no matter how pytest is invoked (from the
repo root, from ``scripts/``, from inside the package, with a directory
argument or with a single file):

1. Put the package's PARENT directory on ``sys.path`` so the modules resolve as
   ``electrostatics_fd.<name>`` (several of them use package-relative imports),
   and the package directory itself so the flat ``import solver`` style used by
   the per-module test files keeps working.  Doing it here means the path shims
   at the top of the individual test modules are redundant -- they are harmless
   and self-contained, so they are left in place rather than churned.

2. Register the ``perf`` marker.  This replaces the earlier ``pytest.ini``,
   which only took effect when pytest's rootdir resolution happened to find it;
   ``pytest_configure`` in a conftest beside the tests always runs.
"""

from __future__ import annotations

import pathlib
import sys

_PKG_DIR = pathlib.Path(__file__).resolve().parent

for _p in (str(_PKG_DIR.parent), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config) -> None:
    """Register the suite's custom markers."""
    config.addinivalue_line(
        "markers",
        'perf: timing and speed-up budgets; deselect with -m "not perf"',
    )

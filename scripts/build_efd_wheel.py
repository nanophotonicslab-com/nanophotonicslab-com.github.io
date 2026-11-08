"""Build ``public/wasm/electrostatics_fd-<version>-py3-none-any.whl``.

The Electrostatics-FD Toolkit page runs the reference Python library *as-is* in
Pyodide, the same way ``lab/bem-solver`` runs ``nplab_bem``: the worker calls
``micropip.install()`` on a wheel served from ``public/wasm/``.  Running the
real library in the browser is the point -- a TypeScript port would be a second
implementation to keep in parity with the first, and the parity campaign of
2026-11-03 is quite enough of that.

Why a hand-rolled zip rather than ``python -m build``:

* No build backend, no network, no ``pyproject.toml`` needed beside a package
  that lives under ``scripts/`` precisely because it is *not* part of the site's
  npm/astro build.
* A pure-Python wheel is a zip with a ``.dist-info`` directory, and that is a
  format worth being explicit about here rather than delegating.
* Deterministic output: fixed timestamps and sorted members, so rebuilding
  without source changes produces a byte-identical wheel.  Needed because the
  wheel is committed, and a wheel whose bytes churn on every build turns every
  diff into noise.

Usage::

    py -3.11 scripts/build_efd_wheel.py            # writes public/wasm/
    py -3.11 scripts/build_efd_wheel.py --check     # verify, write nothing

``--check`` exits non-zero when the committed wheel is stale, which is the form
a CI step or a pre-publish check wants.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import zipfile
from pathlib import Path

VERSION = "0.1.0"
DIST = f"electrostatics_fd-{VERSION}"
WHEEL_NAME = f"{DIST}-py3-none-any.whl"

# The runtime modules, in the order ``__init__`` imports them.  bench.py
# (standalone timing harness) and conftest.py (pytest only) are deliberately
# left out: neither is reachable from the package root, and the wheel is a
# browser payload, so every kilobyte is a download.
MODULES = (
    "__init__.py",
    "units.py",
    "conductivity.py",
    "operators.py",
    "substrate.py",
    "sources.py",
    "solver.py",
    "observables.py",
    "dispersion.py",
    "nonlinear.py",
    "web.py",
)

# Fixed epoch for every member, so the wheel is reproducible.  1980-01-01 is the
# earliest a zip can represent.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

METADATA = f"""Metadata-Version: 2.1
Name: electrostatics-fd
Version: {VERSION}
Summary: Quasistatic finite-difference response of multiple 2D nanostructures
Author: Alvaro Rodriguez Echarri
License: MIT
Requires-Python: >=3.9
Requires-Dist: numpy
Requires-Dist: scipy

Reference implementation of the unified linear + nonlinear + substrate
quasistatic eigenmode formalism for self-interacting multiple graphene
nanoribbons, in the Gaussian eV-nm-fs unit system.

Built for NPLab's Electrostatics-FD Toolkit page, which runs it in Pyodide.
scripts/electrostatics_fd/README.md carries the module map, and what is and
is not externally validated.
"""

WHEEL = """Wheel-Version: 1.0
Generator: build_efd_wheel.py
Root-Is-Purelib: true
Tag: py3-none-any
"""


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build(pkg_dir: Path) -> bytes:
    """Return the wheel's bytes.  Raises FileNotFoundError on a missing module."""
    members: list[tuple[str, bytes]] = []

    for name in MODULES:
        src = pkg_dir / name
        if not src.is_file():
            raise FileNotFoundError(
                f"{src} is missing. MODULES in this script lists what the wheel "
                f"ships; add the file or remove it from that tuple."
            )
        # Read as bytes: the sources are UTF-8 and must land in the wheel
        # byte-for-byte, so no decode/encode round trip.
        members.append((f"electrostatics_fd/{name}", src.read_bytes()))

    members.append((f"{DIST}.dist-info/METADATA", METADATA.encode("utf-8")))
    members.append((f"{DIST}.dist-info/WHEEL", WHEEL.encode("utf-8")))
    members.append((f"{DIST}.dist-info/top_level.txt", b"electrostatics_fd\n"))

    record_lines = [f"{path},{_record_hash(data)},{len(data)}" for path, data in members]
    record_path = f"{DIST}.dist-info/RECORD"
    record_lines.append(f"{record_path},,")
    record = ("\n".join(record_lines) + "\n").encode("utf-8")
    members.append((record_path, record))

    buf = io.BytesIO()
    # ZIP_DEFLATED at the default level: micropip unpacks it in the browser, so
    # the trade is download size against a decompress that is already dwarfed by
    # Pyodide's own startup.
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, data in members:
            info = zipfile.ZipInfo(path, date_time=ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return buf.getvalue()


#: Where the build stamps the wheel's content hash, for the page to read.
#: The service worker caches ``.whl`` CACHE-FIRST keyed on the URL
#: (``public/sw.js``), and the wheel's name carries only the package version --
#: which does not move when the Python inside it does. So every browser that
#: had ever loaded the page kept serving the wheel it first saw, no matter how
#: many times the file on disk was rebuilt, and the page quietly ran old
#: physics. Putting the content hash in the query string gives each build its
#: own cache key, so a changed wheel simply misses the cache.
VERSION_TS = "src/scripts/electrostatics-fd/wheel-version.ts"


def version_source(wheel: bytes) -> str:
    """The generated TypeScript module that carries the hash."""
    digest = hashlib.sha256(wheel).hexdigest()[:12]
    return (
        "// GENERATED by scripts/build_efd_wheel.py -- do not edit by hand.\n"
        "//\n"
        "// The first 12 hex digits of the wheel's SHA-256, used as a cache\n"
        "// buster on its URL. `public/sw.js` caches `.whl` cache-first by URL,\n"
        "// and the wheel's filename only carries the package version, so\n"
        "// without this a rebuilt wheel keeps the old cache entry forever.\n"
        f"export const WHEEL_VERSION = '{digest}';\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed wheel matches the sources; write nothing",
    )
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    pkg_dir = repo / "scripts" / "electrostatics_fd"
    out_dir = repo / "public" / "wasm"
    out = out_dir / WHEEL_NAME

    wheel = build(pkg_dir)

    if args.check:
        if not out.is_file():
            print(f"MISSING  {out}", file=sys.stderr)
            return 1
        if out.read_bytes() != wheel:
            print(
                f"STALE    {out}\n         rebuild with: py -3.11 scripts/build_efd_wheel.py",
                file=sys.stderr,
            )
            return 1
        stamp = repo / VERSION_TS
        want = version_source(wheel)
        if not stamp.is_file() or stamp.read_text(encoding="utf-8") != want:
            print(
                f"STALE    {VERSION_TS}\n"
                f"         rebuild with: py -3.11 scripts/build_efd_wheel.py",
                file=sys.stderr,
            )
            return 1
        print(f"ok       {out.relative_to(repo)} matches the sources ({len(wheel):,} bytes)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    unchanged = out.is_file() and out.read_bytes() == wheel
    out.write_bytes(wheel)
    # Written every time, so the page's cache key cannot drift from the bytes.
    (repo / VERSION_TS).write_text(version_source(wheel), encoding="utf-8")
    with zipfile.ZipFile(io.BytesIO(wheel)) as zf:
        n = len(zf.namelist())
        raw = sum(i.file_size for i in zf.infolist())
    print(
        f"{'unchanged' if unchanged else 'wrote'}    {out.relative_to(repo)}\n"
        f"          {n} members, {raw:,} bytes of source -> {len(wheel):,} bytes compressed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

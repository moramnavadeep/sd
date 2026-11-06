"""Every frontend ES module must parse. `node --check file.js` is NOT enough: on Node 24 it returned
success for a module with a syntax error (an unescaped apostrophe that broke the live site on
2026-11-06). Checking a copy with the .mjs extension forces module parsing and catches it."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "frontend" / "js").glob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_module_parses(path, tmp_path):
    copy = tmp_path / (path.stem + ".mjs")
    copy.write_bytes(path.read_bytes())
    r = subprocess.run(["node", "--check", str(copy)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]

"""The original Twin Studio app is kept byte-identical to the archive."""
import hashlib
import json
from pathlib import Path

STUDIO = Path(__file__).resolve().parents[1] / "services" / "twin_studio" / "studio"


def test_studio_tree_is_byte_identical_to_archive():
    source = json.loads((STUDIO / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["archive_sha256"] == "e3d2e8df8bae997fee741d54923a50999ada261e1fcfff6320cd71649213f97d"
    on_disk = {p.relative_to(STUDIO).as_posix() for p in (STUDIO / "event_horizon_twin_studio").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    assert on_disk == set(source["files"])
    for name, entry in source["files"].items():
        data = (STUDIO / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["bytes"], name


def test_studio_engines_match_integrated_originals():
    engine = STUDIO.parent / "engine" / "original"
    assert (STUDIO / "event_horizon_twin_studio" / "simulation.py").read_bytes() == (engine / "simulation.py").read_bytes()
    assert (STUDIO / "event_horizon_twin_studio" / "dist" / "engine.js").read_bytes() == (engine / "engine.js").read_bytes()

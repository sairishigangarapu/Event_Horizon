"""
conftest.py – pytest path bootstrap for Person 4.

Inserts the person4/ directory (this file's parent) onto sys.path so that
`import event_horizon_p4` always resolves correctly, regardless of the
working directory from which pytest is invoked (e.g. repo root vs person4/).

This is the standard pattern for self-contained sub-packages that live inside
a larger monorepo and are not installed into the active environment's
site-packages.
"""
import sys
import pathlib

# Resolve person4/ (the directory containing this conftest.py)
_PERSON4_ROOT = pathlib.Path(__file__).parent.resolve()

if str(_PERSON4_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERSON4_ROOT))

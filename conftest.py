"""Root conftest.py for the Event Horizon project.

This conftest establishes a reproducible test-discovery policy:

1. It ensures the active ``studio/`` directory is at the front of ``sys.path``
   so that the top-level module name ``simulation`` resolves to
   ``studio/simulation.py`` (which defines ``live_steps``, ``simulate``,
   ``validate``, ``normalize_routes``, and ``path``) rather than to one of
   the archived ``simulation.py`` files under
   ``services/twin_studio/``.

2. It explicitly excludes legacy/archive directories from test collection
   to prevent import-name collisions on ``simulation``, ``observations``,
   and ``lab_server``.

To run the **maintained** suite (active application + tests + person4 + studio):

    PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider

To run the **archive** suite independently (note: may need additional setup):

    PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider \
      services/twin_studio/v3/archive \
      services/twin_studio/studio/event_horizon_twin_studio

To run only the **person4** suite:

    PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider person4/tests/

To run only the **studio** suite:

    PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider studio/

To run only the **main** suite (tests/):

    PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider tests/
"""
import sys
from pathlib import Path

# Ensure the active studio/ is first on sys.path so that 'simulation' and
# 'lab_server' resolve to the maintained implementations.
_STUDIO_DIR = str(Path(__file__).parent / "studio")
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

# Explicitly exclude legacy/archive directories from test collection.
# These directories define competing top-level modules (simulation.py,
# observations.py, lab_server.py) that collide with the maintained
# implementations in studio/ and crowd_reliability/.
collect_ignore = [
    "services/twin_studio/v3/archive",
    "services/twin_studio/studio/event_horizon_twin_studio",
]

# Also ignore during collection by glob so pytest never descends into them.
collect_ignore_glob = [
    "services/twin_studio/v3/archive/**",
    "services/twin_studio/studio/event_horizon_twin_studio/**",
]

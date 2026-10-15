"""
Test-session isolation for the main test suite.

crowd_reliability.database builds its SQLAlchemy engine at import time from
DATABASE_URL (default: ./data/app.db, the developer's demo database). Without
this file the tests read and write that real database, so results depended on
whether it had been seeded and running the suite mutated it.

pytest imports this conftest before any test module, i.e. before
crowd_reliability is imported, so setting the variable here is early enough.
An explicitly provided DATABASE_URL is respected.
"""
import os
import shutil
import tempfile

_TEST_DB_DIR = None

if not os.environ.get("DATABASE_URL"):
    _TEST_DB_DIR = tempfile.mkdtemp(prefix="eh_test_db_")
    _db_path = os.path.join(_TEST_DB_DIR, "test.db").replace("\\", "/")
    os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"


def pytest_sessionfinish(session, exitstatus):
    if _TEST_DB_DIR is None:
        return
    try:
        from crowd_reliability.database import engine
        engine.dispose()  # release the SQLite file handle (required on Windows)
    except Exception:
        pass
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)

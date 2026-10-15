"""The main dashboard renders headless and labels Stream B as experimental."""
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_with_readiness_warning():
    at = AppTest.from_file("dashboard/app.py", default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Experimental Stream B" in w.value for w in at.warning)

"""Phase 7 tests: dashboard reveal flow reads evaluator outputs only; no reference access."""
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from services.twin_studio import report_view as RV  # noqa: E402
from services.twin_studio.evaluation import STREAM_B_ID  # noqa: E402
from tests.test_twin_studio_phase5_6 import evaluate, store  # noqa: E402,F401  (fixture re-use)

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "dashboard" / "pages" / "1_Twin_Studio_Evaluation.py"


@pytest.fixture(scope="module")
def evaluation(store, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p7")
    rep = evaluate(store, tmp)
    return tmp / "eval", rep


def test_view_model(evaluation):
    root, rep = evaluation
    evs = RV.list_evaluations(root)
    assert [e["run_id"] for e in evs] == [rep["run_id"]]
    data = RV.load_evaluation(evs[0]["path"])
    obs = RV.observation_rows(data["report"])
    assert obs and all({"observed_at", "published_at", "received_at"} <= set(r) for r in obs)
    fc = RV.forecast_rows(data["forecasts"], stream=STREAM_B_ID, horizon=30)
    assert fc and all(r["horizon_seconds"] == 30 for r in fc)
    labels = RV.revealed_label_rows(data["report"], stream=STREAM_B_ID, horizon=30)
    assert labels and {r["label_status"] for r in labels} <= {"valid", "missing (undefined at target)", "not applicable"}
    assert all(r["label"] is None for r in labels if r["label_status"] != "valid")
    assert any("untrained" in w for w in RV.model_warnings(data["report"]))
    assert RV.metric_rows(data["report"])


def test_page_runs_and_reveals_only_on_request(evaluation, monkeypatch):
    root, _ = evaluation
    monkeypatch.setenv("EH_TWIN_EVALUATIONS_DIR", str(root))
    at = AppTest.from_file(str(PAGE), default_timeout=60).run()
    assert not at.exception
    assert any("untrained" in w.value for w in at.warning)
    assert any("Labels hidden" in i.value for i in at.info)
    n_before = len(at.dataframe)
    at.button[0].click().run()
    assert not at.exception and len(at.dataframe) == n_before + 2      # labels + metrics tables appear


def test_dashboard_has_no_reference_access():
    for path in (PAGE, ROOT / "services" / "twin_studio" / "report_view.py"):
        code = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
        code = re.sub(r'"""[\s\S]*?"""', "", code)        # docstrings may describe the boundary
        for forbidden in ("EH_REF", "reference_api_token", "/sessions", "reference-api", "httpx", "requests",
                          "boto3", "storage.readers", "ReferenceStore", "secrets", "eh-reference"):
            assert forbidden not in code, (path.name, forbidden)

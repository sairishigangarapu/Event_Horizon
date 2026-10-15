"""Integration fixtures: run ONLY inside the Compose `tests` service (EH_TWIN_INTEGRATION=1),
against the real MinIO deployment and the running observation/reference APIs."""
import os
import secrets
from pathlib import Path

import pytest

if os.environ.get("EH_TWIN_INTEGRATION") != "1":
    collect_ignore_glob = ["test_*.py"]

SECRETS = Path(os.environ.get("EH_SECRETS_DIR", "/run/secrets"))
ENDPOINT = os.environ.get("EH_S3_ENDPOINT", "http://minio:9000")


def secret(name: str) -> str:
    return (SECRETS / name).read_text(encoding="utf-8").strip()


def store_for(user: str):
    from services.twin_studio.storage.objects import s3_store
    if user == "root":
        return s3_store(ENDPOINT, secret("minio_root_user"), secret("minio_root_password"))
    return s3_store(ENDPOINT, secret(f"minio_{user}_access_key"), secret(f"minio_{user}_secret_key"))


@pytest.fixture(scope="session")
def admin():
    return store_for("root")


@pytest.fixture(scope="session")
def publisher():
    return store_for("publisher")


@pytest.fixture(scope="session")
def obs_reader():
    return store_for("obs_reader")


@pytest.fixture(scope="session")
def ref_reader():
    return store_for("ref_reader")


@pytest.fixture(scope="session")
def keyring():
    from services.twin_studio.keyed_ids import load_keyring
    return load_keyring()


@pytest.fixture(scope="session")
def session_tag():
    """Buckets persist across test runs (object lock forbids cleanup), so each run uses fresh inputs."""
    return secrets.token_hex(4)


def make_spec(tag: str, variant: str):
    from services.twin_studio.bundle import RunSpec
    scene = {
        "duration": 60, "scale": .1, "seed": 11, "randomCount": 1,
        "routes": [{"id": "r_e", "name": "Annex to E gate", "points": [{"x": 800, "y": 580}, {"x": 968, "y": 580}],
                    "count": 40, "speed": 1.4, "width": 3},
                   {"id": "r_s", "name": "Courtyard to south", "points": [{"x": 450, "y": 450}, {"x": 450, "y": 640}],
                    "count": 30, "speed": 1.2, "width": 3}],
        "events": [{"id": "inc_e", "name": "East gate shut", "type": "gate_closure", "gateId": "T_GATE_E",
                    "start": 15, "end": 40}],
    }
    return RunSpec(f"itest_{tag}_{variant}_east_gate_closure", scene, "custom", "2026-10-15T09:00:00Z")


@pytest.fixture(scope="session")
def published_run(publisher, keyring, session_tag, tmp_path_factory):
    from services.twin_studio.storage.publisher import publish_to_store
    spec = make_spec(session_tag, "main")
    res = publish_to_store(spec, publisher, keyring, tmp_path_factory.mktemp("bundles"))
    assert res.status == "published"
    return spec, res.run_id


@pytest.fixture(scope="session")
def obs_api():
    import httpx
    with httpx.Client(base_url=os.environ["EH_OBS_API_URL"], timeout=30) as c:
        yield c


@pytest.fixture(scope="session")
def ref_api():
    import httpx
    with httpx.Client(base_url=os.environ["EH_REF_API_URL"], timeout=30) as c:
        yield c


@pytest.fixture(scope="session")
def control_headers():
    return {"Authorization": f"Bearer {secret('observation_control_token')}"}


@pytest.fixture(scope="session")
def eval_headers():
    return {"Authorization": f"Bearer {secret('reference_api_token')}"}

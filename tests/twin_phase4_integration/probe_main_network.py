"""Runs in the `main-app-probe` container, attached ONLY to eh_observation (the main app's network).
Exit 0 iff the observation API is reachable and the reference API and MinIO are NOT reachable."""
import socket
import sys
import urllib.request


def reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


results = {
    "observation-api:8000 reachable": urllib.request.urlopen("http://observation-api:8000/health", timeout=5).status == 200,
    "reference-api:8000 unreachable": not reachable("reference-api", 8000),
    "minio:9000 unreachable": not reachable("minio", 9000),
}
for name, ok in results.items():
    print(("PASS " if ok else "FAIL ") + name)
sys.exit(0 if all(results.values()) else 1)

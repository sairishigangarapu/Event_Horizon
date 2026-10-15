"""Generates the local secrets for the Phase 4 Twin Studio Compose deployment (stdlib only).

  python deploy/twin_studio/init_secrets.py                    # create missing secrets, never overwrite
  python deploy/twin_studio/init_secrets.py --rotate-id-key    # add key version k(n+1) and make it active

Secrets are written to deploy/twin_studio/secrets/ (git-ignored, docker-ignored). Values are never
printed. Every secret is independently generated with `secrets`, and the script verifies that no
two are equal (in particular the MinIO root password is never reused as an HMAC key or API token).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "secrets"
USERS = ("publisher", "obs_reader", "ref_reader")
KEYRING = "id_keyring.json"
KEYRING_FORMAT = "eh-twin-id-keyring-v1"


def _write_new(path: Path, text: str) -> bool:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return True


def _replace(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _token_pair(d: Path, name: str, created: list) -> None:
    token = d / name
    if _write_new(token, secrets.token_urlsafe(32) + "\n"):
        created.append(name)
    digest = hashlib.sha256(token.read_text(encoding="utf-8").strip().encode()).hexdigest()
    if _write_new(d / f"{name}.sha256", digest + "\n"):
        created.append(f"{name}.sha256")


def init(d: Path) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    plan = {"minio_root_user": "ehroot" + secrets.token_hex(4), "minio_root_password": secrets.token_hex(24)}
    for user in USERS:
        plan[f"minio_{user}_access_key"] = f"eh{user.replace('_', '')}{secrets.token_hex(4)}"
        plan[f"minio_{user}_secret_key"] = secrets.token_hex(24)
    for name, value in plan.items():
        if _write_new(d / name, value + "\n"):
            created.append(name)
    keyring = {"format": KEYRING_FORMAT, "active": "k1", "keys": {"k1": secrets.token_hex(32)}}
    if _write_new(d / KEYRING, json.dumps(keyring, indent=2) + "\n"):
        created.append(KEYRING)
    _token_pair(d, "reference_api_token", created)
    _token_pair(d, "observation_control_token", created)
    check_distinct(d)
    return created


def rotate(d: Path) -> str:
    path = d / KEYRING
    doc = json.loads(path.read_text(encoding="utf-8"))
    nxt = f"k{max(int(v[1:]) for v in doc['keys']) + 1}"
    doc["keys"][nxt] = secrets.token_hex(32)
    doc["active"] = nxt                 # old versions are KEPT so earlier runs stay verifiable
    _replace(path, json.dumps(doc, indent=2) + "\n")
    check_distinct(d)
    return nxt


def check_distinct(d: Path) -> None:
    values = {}
    for p in d.iterdir():
        if p.is_file() and p.name != KEYRING and not p.name.endswith((".sha256", ".tmp", ".md")):
            values[p.name] = p.read_text(encoding="utf-8").strip()
    if (d / KEYRING).is_file():
        for v, key in json.loads((d / KEYRING).read_text(encoding="utf-8"))["keys"].items():
            values[f"{KEYRING}:{v}"] = key
    seen = {}
    for name, value in values.items():
        if value in seen:
            sys.exit(f"refusing: secrets {seen[value]} and {name} are identical")
        seen[value] = name
    root_pw = values.get("minio_root_password")
    if root_pw and any(root_pw in v for n, v in values.items() if n != "minio_root_password"):
        sys.exit("refusing: the MinIO root password appears inside another secret")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--rotate-id-key", action="store_true")
    args = ap.parse_args()
    if args.rotate_id_key:
        print(f"identity key rotated: new active version {rotate(args.dir)} (previous versions retained)")
        return
    created = init(args.dir)
    print("created: " + (", ".join(created) if created else "nothing (all secrets already exist)"))


if __name__ == "__main__":
    main()

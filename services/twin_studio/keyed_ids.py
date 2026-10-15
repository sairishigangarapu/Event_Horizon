"""Keyed (HMAC-SHA-256) run, report and public-scenario identifiers (Phase 4).

Why: the Phase 3 run ID is a prefix of SHA-256(canonical run inputs). Seeds are small integers, modes
and incident settings come from small sets, so anyone holding an observation bundle could recompute
candidate IDs offline and recover hidden settings. Keyed IDs remove that oracle.

Algorithm `hmac_sha256_v1` (versioned; any change needs a new scheme name):
  master key  K_v           32+ random bytes for key version v (e.g. "k1"), from the keyring secret
  subkeys     K_v,purpose = HMAC-SHA-256(K_v, "eh-twin/kdf/v1/" + purpose)
              purposes: "run-id", "report-id", "scenario-alias"   (domain separation by key)
  run ID      "ehts_" + hex(HMAC(K_v,run-id, "eh-twin/run-id/v1\\0" + canonical_json(run_inputs)))[:32]
  report ID   "R-" + hex(HMAC(K_v,report-id, "eh-twin/report-id/v1\\0" + run_id|source|zone|variable|t))[:20]
  alias       "scn_" + hex(HMAC(K_v,scenario-alias, "eh-twin/scenario-alias/v1\\0" + scenario_id))[:16]
(domain separation also by message prefix). run_inputs are exactly the Phase 3 canonical inputs.

Keyring secret (JSON, never in source, image, manifest, artifact or log):
  {"format": "eh-twin-id-keyring-v1", "active": "k2", "keys": {"k1": "<hex>", "k2": "<hex>"}}
Rotation is explicit: a new version is ADDED and made active; old versions stay for verification.
The same inputs under a new key version get a new run ID; the publisher refuses to create it silently
when the inputs were already published under an older version (see storage.publisher).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from pathlib import Path

RUN_ID_SCHEME = "hmac_sha256_v1"
LEGACY_RUN_ID_SCHEME = "sha256_unkeyed_v1"
KEYRING_FORMAT = "eh-twin-id-keyring-v1"
KEYRING_ENV = "EH_TWIN_ID_KEYRING_FILE"
KEY_VERSION_RE = re.compile(r"^k[1-9][0-9]{0,5}$")
MIN_KEY_BYTES = 32
PURPOSES = ("run-id", "report-id", "scenario-alias")


class KeyringError(RuntimeError):
    """The identity keyring is missing or invalid. Messages never contain key material."""


class IdentityKeyring:
    """Versioned HMAC master keys. repr/str never reveal key material."""

    __slots__ = ("_keys", "active_version")

    def __init__(self, keys: dict[str, bytes], active_version: str):
        if not keys:
            raise KeyringError("identity keyring contains no keys")
        for version, key in keys.items():
            if not KEY_VERSION_RE.match(version):
                raise KeyringError(f"invalid key version name {version!r} (expected k1, k2, ...)")
            if not isinstance(key, bytes) or len(key) < MIN_KEY_BYTES:
                raise KeyringError(f"key version {version} is shorter than {MIN_KEY_BYTES} bytes")
            if len(set(key)) < 8:
                raise KeyringError(f"key version {version} has too little variety to be a random key")
        if len(set(keys.values())) != len(keys):
            raise KeyringError("two key versions share the same key material")
        if active_version not in keys:
            raise KeyringError(f"active key version {active_version!r} is not in the keyring")
        self._keys = dict(keys)
        self.active_version = active_version

    def __repr__(self) -> str:
        return f"IdentityKeyring(active={self.active_version!r}, versions={self.versions})"

    __str__ = __repr__

    @property
    def versions(self) -> list[str]:
        return sorted(self._keys, key=lambda v: int(v[1:]))

    def has(self, version: str) -> bool:
        return version in self._keys

    def _subkey(self, version: str, purpose: str) -> bytes:
        if purpose not in PURPOSES:
            raise ValueError(f"unknown key purpose {purpose}")
        if version not in self._keys:
            raise KeyringError(f"key version {version!r} is not in the keyring (was it removed after rotation?)")
        return hmac.new(self._keys[version], b"eh-twin/kdf/v1/" + purpose.encode(), hashlib.sha256).digest()

    def _mac(self, version: str, purpose: str, message: bytes) -> str:
        domain = f"eh-twin/{purpose}/v1".encode() + b"\x00"
        return hmac.new(self._subkey(version, purpose), domain + message, hashlib.sha256).hexdigest()

    def run_id(self, canonical_inputs: str, version: str | None = None) -> str:
        return "ehts_" + self._mac(version or self.active_version, "run-id", canonical_inputs.encode())[:32]

    def report_id(self, run_id: str, source: str, zone: str, variable: str, t: int, version: str | None = None) -> str:
        message = f"{run_id}|{source}|{zone}|{variable}|{t}".encode()
        return "R-" + self._mac(version or self.active_version, "report-id", message)[:20]

    def scenario_alias(self, scenario_id: str, version: str | None = None) -> str:
        return "scn_" + self._mac(version or self.active_version, "scenario-alias", scenario_id.encode())[:16]

    def contains_secret(self, data: bytes) -> bool:
        """True if any master key or derived subkey (raw, hex or HEX) appears in `data`."""
        needles = []
        for version, key in self._keys.items():
            for k in [key] + [self._subkey(version, p) for p in PURPOSES]:
                needles += [k, k.hex().encode(), k.hex().upper().encode()]
        return any(n in data for n in needles)


def parse_keyring(text: str) -> IdentityKeyring:
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise KeyringError("identity keyring is not valid JSON") from None
    if not isinstance(doc, dict) or doc.get("format") != KEYRING_FORMAT:
        raise KeyringError(f"identity keyring format must be {KEYRING_FORMAT}")
    raw = doc.get("keys")
    if not isinstance(raw, dict):
        raise KeyringError("identity keyring 'keys' must be an object")
    keys = {}
    for version, value in raw.items():
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]+", value) or len(value) % 2:
            raise KeyringError(f"key version {version!r} is not lowercase hexadecimal")
        keys[version] = bytes.fromhex(value)
    return IdentityKeyring(keys, doc.get("active"))


def load_keyring(path: str | os.PathLike | None = None) -> IdentityKeyring:
    """Loads the keyring from `path` or $EH_TWIN_ID_KEYRING_FILE. Fails closed."""
    path = path or os.environ.get(KEYRING_ENV)
    if not path:
        raise KeyringError(f"no identity keyring configured (set {KEYRING_ENV})")
    p = Path(path)
    if not p.is_file():
        raise KeyringError("identity keyring file not found")
    return parse_keyring(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- public scenario aliases

# An explicit public alias must be neutral: lowercase letters/underscores only (no digits, so no
# seeds, times or gate numbers) and none of these hidden-configuration terms.
ALIAS_RE = re.compile(r"^[a-z][a-z_]{2,39}$")
ALIAS_FORBIDDEN_TERMS = ("gate", "close", "closure", "block", "surge", "collapse", "incident", "seed", "outage",
                         "evac", "emergenc", "crush", "fire", "alarm", "ideal", "custom", "random", "mode",
                         "drop", "noise", "fault", "fail", "exit", "entrance", "annex", "courtyard", "walk",
                         "connector", "pond", "north", "south", "east", "west", "nw", "sw")


def validate_public_alias(alias: str) -> str:
    if not isinstance(alias, str) or not ALIAS_RE.match(alias):
        raise ValueError("public scenario alias must match ^[a-z][a-z_]{2,39}$ (no digits)")
    hits = [t for t in ALIAS_FORBIDDEN_TERMS if t in alias]
    if hits:
        raise ValueError(f"public scenario alias contains hidden-configuration terms: {hits}")
    return alias

"""Validated, immutable local run bundles for Twin Studio runs (Phase 3).

One run = ONE simulation execution, from which both bundles are built:
  <root>/observations/<run_id>/   serving-safe: observations.jsonl, venue.json, manifest.json
  <root>/reference/<run_id>/      reference-only: reference_states.jsonl, accounting.jsonl,
                                  future_labels.jsonl, incidents.json, run_inputs.json, manifest.json
  <root>/published/<run_id>.json  publication-complete marker (written last, atomically)
  <root>/.staging/                private per-operation staging directories

A run is complete ONLY when its marker exists and verifies. Bundles are installed with
non-overwriting directory renames; nothing published is ever modified or deleted.
Re-publishing identical content is a no-op; different content for an existing run ID raises
BundleIntegrityError. Phase 4 (storage.publisher) uploads exactly these verified bytes to MinIO;
with an identity keyring the run/report IDs and public scenario id are keyed (keyed_ids).

Usage: python -m services.twin_studio.bundle --spec spec.json --root DIR
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION
from services.twin_studio.engine import simulation
from services.twin_studio.keyed_ids import (LEGACY_RUN_ID_SCHEME, RUN_ID_SCHEME, IdentityKeyring,
                                            validate_public_alias)
from services.twin_studio.labels import (DEFAULT_HORIZONS_SECONDS, DEFAULT_LABEL_DELAY_SECONDS,
                                         DEFAULT_LABEL_VARIABLES, INTERVAL_VARIABLES, future_labels,
                                         reference_records)
from services.twin_studio.observations import (CAMERA_VARIABLES, GATE_VARIABLES, ObservationConfig, RunContext,
                                               UNITS as OBS_UNITS, generate_observations)
from services.twin_studio.run_identity import (SCHEMA_VERSIONS, SUBSYSTEM, VENUE_FILE, canonical_json,
                                               code_fingerprint, normalise_epoch, run_id_from_digest,
                                               run_input_digest, run_inputs, venue_sha256)
from services.twin_studio.zone_states import UNSUPPORTED, derive_zone_states
from services.twin_studio.zones import load_venue

OBS_RECORD_FIELDS = ("report_id", "run_id", "scenario_id", "zone_id", "source_id", "variable", "value", "unit",
                     "simulation_second", "observed_at", "published_at", "received_at", "source_type",
                     "environment_id", "map_version", "schema_version")
OBS_MANIFEST_FIELDS = ("bundle_kind", "bundle_schema_version", "run_id", "environment_id", "map_version",
                       "venue_sha256", "scenario_id", "epoch_utc", "sample_seconds", "observation_schema_version",
                       "observation_every_seconds", "source_coverage", "variables", "files")
OBS_FILES = ("observations.jsonl", "venue.json", "manifest.json")
REF_FILES = ("reference_states.jsonl", "accounting.jsonl", "future_labels.jsonl", "incidents.json",
             "run_inputs.json", "manifest.json")
# Tokens that must never appear in any observation-bundle file
OBS_FORBIDDEN_TOKENS = ('"run_mode"', '"seed"', '"state_at"', '"label_available_at"', '"target_at"',
                        "SIMULATOR_HIDDEN_REFERENCE", "HIDDEN_FUTURE_LABEL", "gate_closure", "collapse", "surge",
                        "incidents", "reference_states", "future_labels", "run_inputs", "/reference/")
SCENARIO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
RUN_ID_RE = re.compile(r"^ehts_[0-9a-f]{32}$")
ALIAS_ID_RE = re.compile(r"^scn_[0-9a-f]{16}$")


class BundleValidationError(ValueError):
    """A generated bundle violates its contract; nothing is published."""


class BundleIntegrityError(RuntimeError):
    """Published content conflicts with the deterministic expectation for its run ID."""


@dataclass(frozen=True)
class RunSpec:
    scenario_id: str
    scene: dict
    mode: str
    epoch_utc: str
    sample_seconds: int = 5
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    label_horizons_seconds: tuple = DEFAULT_HORIZONS_SECONDS
    label_delay_seconds: int = DEFAULT_LABEL_DELAY_SECONDS
    label_variables: tuple = DEFAULT_LABEL_VARIABLES
    public_scenario_id: str | None = None   # optional approved neutral alias (Phase 4); default: keyed alias


def venue_gates(venue: dict) -> list[dict]:
    return [{"id": z["zone_id"], **{k: z["geometry"][k] for k in ("x", "y", "w", "d")}}
            for z in venue["zones"] if z["zone_type"] == "exit"]


def prepare_spec(spec: RunSpec, venue: dict) -> RunSpec:
    """Validates identity fields and injects the frozen venue gates into the scene."""
    if not SCENARIO_ID_RE.match(spec.scenario_id or ""):
        raise ValueError("scenario_id must match [A-Za-z0-9_.-]{1,64}")
    if spec.mode not in ("ideal", "custom", "random"):
        raise ValueError("mode must be ideal, custom or random")
    if spec.label_delay_seconds <= 0:
        raise ValueError("label_delay_seconds must be > 0")
    if spec.public_scenario_id is not None:
        validate_public_alias(spec.public_scenario_id)
    scene = copy.deepcopy(spec.scene)
    gates = venue_gates(venue)
    if "gates" in scene and scene["gates"] != gates:
        raise ValueError("scene gates must equal the frozen twin_venue_v1 gate zones")
    scene["gates"] = gates
    return RunSpec(spec.scenario_id, scene, spec.mode, normalise_epoch(spec.epoch_utc), spec.sample_seconds,
                   spec.observation_config, tuple(spec.label_horizons_seconds), spec.label_delay_seconds,
                   tuple(spec.label_variables), spec.public_scenario_id)


def _jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for r in records).encode()


def _json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _file_entry(data: bytes, records: int | None = None) -> dict:
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if records is not None:
        entry["records"] = records
    return entry


@dataclass
class RunOutputs:
    run_id: str
    run_input_sha256: str
    observation_files: dict[str, bytes]
    reference_files: dict[str, bytes]
    summary: dict[str, Any]


def build_run(spec: RunSpec, identity: IdentityKeyring | None = None) -> RunOutputs:
    """Executes the simulation ONCE and derives both bundles (in memory) from that single execution.

    identity=None: Phase 3 local scheme (unkeyed run/report IDs, scenario_id public as given).
    identity=keyring: Phase 4 publication scheme (keyed run/report IDs with the ACTIVE key version and a
    keyed public scenario alias unless an approved alias is given). Values, timings and every other
    field are identical under both schemes."""
    venue = load_venue()
    spec = prepare_spec(spec, venue)
    inputs = run_inputs(spec)
    digest = run_input_digest(inputs)
    if identity is None:
        run_id, scheme, key_version = run_id_from_digest(digest), LEGACY_RUN_ID_SCHEME, None
        public_sid = spec.public_scenario_id or spec.scenario_id
        report_id_fn = None
    else:
        run_id, scheme, key_version = identity.run_id(canonical_json(inputs)), RUN_ID_SCHEME, identity.active_version
        public_sid = spec.public_scenario_id or identity.scenario_alias(spec.scenario_id)
        report_id_fn = lambda source, zone, variable, t: identity.report_id(run_id, source, zone, variable, t)
    epoch = datetime.fromisoformat(spec.epoch_utc.replace("Z", "+00:00"))

    result = simulation.simulate(spec.scene, spec.mode)
    states = derive_zone_states(result, venue, spec.sample_seconds)
    ctx = RunContext(spec.scenario_id, spec.mode, spec.scene["seed"], epoch, spec.sample_seconds)
    raw_reports = generate_observations(states, venue, ctx, spec.observation_config, report_id_fn=report_id_fn)
    reports = [{**{k: r[k] for k in OBS_RECORD_FIELDS if k not in ("run_id", "scenario_id")},
                "run_id": run_id, "scenario_id": public_sid} for r in raw_reports]
    refs = reference_records(states, run_id=run_id, scenario_id=spec.scenario_id, epoch=epoch,
                             sample_seconds=spec.sample_seconds, label_delay_seconds=spec.label_delay_seconds)
    labels, excluded = future_labels(states, run_id=run_id, scenario_id=spec.scenario_id, epoch=epoch,
                                     sample_seconds=spec.sample_seconds, horizons_seconds=spec.label_horizons_seconds,
                                     variables=spec.label_variables, label_delay_seconds=spec.label_delay_seconds)
    accounting = [{"run_id": run_id, "simulation_second": st["simulation_second"],
                   "state_at": instant_iso(epoch, st["simulation_second"]), **st["accounting"]} for st in states]

    venue_bytes = (SUBSYSTEM / VENUE_FILE).read_bytes().replace(b"\r\n", b"\n")
    obs_files = {"observations.jsonl": _jsonl(reports), "venue.json": venue_bytes}
    oc = spec.observation_config
    zone_ids = [z["zone_id"] for z in venue["zones"]]
    coverage = {"sim_camera_A": sorted(oc.camera_a_zones) if oc.camera_a_zones is not None else sorted(zone_ids),
                "sim_camera_B": sorted(oc.camera_b_zones),
                "sim_gate_sensor": sorted(g["id"] for g in venue_gates(venue))}
    obs_manifest = {
        "bundle_kind": "observations", "bundle_schema_version": SCHEMA_VERSIONS["bundle"], "run_id": run_id,
        "environment_id": ENVIRONMENT_ID, "map_version": MAP_VERSION, "venue_sha256": venue_sha256(),
        "scenario_id": public_sid, "epoch_utc": spec.epoch_utc, "sample_seconds": spec.sample_seconds,
        "observation_schema_version": SCHEMA_VERSIONS["observations"],
        "observation_every_seconds": oc.observation_every_seconds,
        "source_coverage": coverage,
        "variables": {v: OBS_UNITS[v] for v in CAMERA_VARIABLES + GATE_VARIABLES},
        "files": {"observations.jsonl": _file_entry(obs_files["observations.jsonl"], len(reports)),
                  "venue.json": _file_entry(venue_bytes)},
    }
    obs_files["manifest.json"] = _json(obs_manifest)

    source = json.loads((SUBSYSTEM / "engine" / "SOURCE.json").read_text(encoding="utf-8"))
    ref_files = {
        "reference_states.jsonl": _jsonl(refs),
        "accounting.jsonl": _jsonl(accounting),
        "future_labels.jsonl": _jsonl(labels),
        "incidents.json": _json({"run_id": run_id, "mode": spec.mode, "events": result["events"],
                                 "note": "Reference-only incident schedule; never an operational input."}),
        "run_inputs.json": (canonical_json(inputs) + "\n").encode(),
    }
    ref_manifest = {
        "bundle_kind": "reference", "bundle_schema_version": SCHEMA_VERSIONS["bundle"], "run_id": run_id,
        "run_input_sha256": digest, "run_id_scheme": scheme, "id_key_version": key_version,
        "public_scenario_id": public_sid, "environment_id": ENVIRONMENT_ID, "map_version": MAP_VERSION,
        "venue_sha256": venue_sha256(), "scenario_id": spec.scenario_id, "mode": spec.mode,
        "seed": spec.scene["seed"], "epoch_utc": spec.epoch_utc, "sample_seconds": spec.sample_seconds,
        "schema_versions": SCHEMA_VERSIONS, "code_fingerprint": code_fingerprint(),
        "engine_source": {"original": source["original"], "modified": {k: source["modified"][k] for k in ("python", "javascript")}},
        "label_rule": {"label_available_at": "target_at + label_delay_seconds (reference states: state_at + label_delay_seconds)",
                       "label_delay_seconds": spec.label_delay_seconds,
                       "horizons_seconds": list(spec.label_horizons_seconds),
                       "target_variables": list(spec.label_variables),
                       "excluded_issue_instants_by_horizon": excluded},
        "unsupported": UNSUPPORTED,
        "observation_manifest_sha256": hashlib.sha256(obs_files["manifest.json"]).hexdigest(),
        "files": {name: _file_entry(data, data.count(b"\n") if name.endswith(".jsonl") else None)
                  for name, data in ref_files.items()},
    }
    ref_files["manifest.json"] = _json(ref_manifest)
    summary = {"states": len(states), "reports": len(reports), "reference_records": len(refs),
               "future_labels": len(labels), "excluded": excluded, "total_agents": result["total"]}
    return RunOutputs(run_id, digest, obs_files, ref_files, summary)


def instant_iso(epoch: datetime, seconds: int) -> str:
    from services.twin_studio.timebase import instant_to_utc
    return instant_to_utc(epoch, seconds).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- validation

def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def validate_outputs(out: RunOutputs, identity: IdentityKeyring | None = None) -> list[str]:
    """Returns the list of checks performed; raises BundleValidationError listing every problem.
    Keyed runs (Phase 4) can only be validated with a keyring holding their key version."""
    problems, checks = [], []
    venue = json.loads(out.observation_files["venue.json"])
    zone_ids = {z["zone_id"] for z in venue["zones"]}
    gates = {z["zone_id"] for z in venue["zones"] if z["zone_type"] == "exit"}
    obs_m = json.loads(out.observation_files["manifest.json"])
    ref_m = json.loads(out.reference_files["manifest.json"])
    obs = [json.loads(l) for l in out.observation_files["observations.jsonl"].splitlines()]
    refs = [json.loads(l) for l in out.reference_files["reference_states.jsonl"].splitlines()]
    labels = [json.loads(l) for l in out.reference_files["future_labels.jsonl"].splitlines()]
    dt = ref_m["sample_seconds"]
    epoch = _parse(ref_m["epoch_utc"])
    sec = lambda iso: int((_parse(iso) - epoch).total_seconds())

    def check(name, ok, detail=""):
        checks.append(name)
        if not ok:
            problems.append(f"{name}: {detail}")

    # identities and versions
    check("run_id format", bool(RUN_ID_RE.match(out.run_id)), out.run_id)
    inputs_text = out.reference_files["run_inputs.json"].decode().rstrip("\n")
    check("input digest matches run_inputs.json",
          hashlib.sha256(inputs_text.encode()).hexdigest() == ref_m["run_input_sha256"] == out.run_input_sha256)
    scheme, key_version = ref_m.get("run_id_scheme"), ref_m.get("id_key_version")
    public_sid = ref_m.get("public_scenario_id")
    keyed = scheme == RUN_ID_SCHEME
    if scheme == LEGACY_RUN_ID_SCHEME:
        check("run_id derives from input digest", out.run_id == run_id_from_digest(ref_m["run_input_sha256"]))
        check("legacy scheme has no key version", key_version is None)
    elif not keyed:
        check("run_id scheme known", False, scheme)
    elif identity is None or not identity.has(key_version or ""):
        check("keyed run_id verifiable", False, f"keyring lacks key version {key_version!r}")
    else:
        check("run_id is the keyed HMAC of run_inputs", out.run_id == identity.run_id(inputs_text, key_version))
        check("report ids are keyed HMACs",
              all(r["report_id"] == identity.report_id(out.run_id, r["source_id"], r["zone_id"], r["variable"],
                                                         r["simulation_second"], key_version) for r in obs))
        explicit = json.loads(inputs_text).get("public_scenario_id")
        check("public scenario id", public_sid == (explicit or identity.scenario_alias(ref_m["scenario_id"], key_version)))
        secret_hits = [n for n, d in {**out.observation_files, **out.reference_files}.items() if identity.contains_secret(d)]
        check("no key material in any artifact", not secret_hits, secret_hits)
    if keyed:
        check("public scenario id neutral",
              bool(public_sid) and (bool(ALIAS_ID_RE.match(public_sid)) or _alias_ok(public_sid)), public_sid)
    check("manifests share run identity", obs_m["run_id"] == ref_m["run_id"] == out.run_id
          and obs_m["scenario_id"] == public_sid and obs_m["epoch_utc"] == ref_m["epoch_utc"])
    check("manifest cross-link", ref_m["observation_manifest_sha256"] == hashlib.sha256(out.observation_files["manifest.json"]).hexdigest())
    check("schema versions", ref_m["schema_versions"] == SCHEMA_VERSIONS and obs_m["bundle_schema_version"] == SCHEMA_VERSIONS["bundle"])
    check("map version", venue["map_version"] == MAP_VERSION == obs_m["map_version"] == ref_m["map_version"])
    check("records carry run identity",
          all(r["run_id"] == out.run_id and r["scenario_id"] == public_sid for r in obs)
          and all(r["run_id"] == out.run_id and r["scenario_id"] == ref_m["scenario_id"] for r in refs + labels))
    # observation bundle safety
    check("observation record allowlist", all(set(r) == set(OBS_RECORD_FIELDS) for r in obs))
    check("observation manifest allowlist", set(obs_m) == set(OBS_MANIFEST_FIELDS), sorted(set(obs_m) ^ set(OBS_MANIFEST_FIELDS)))
    check("observation file set", set(out.observation_files) == set(OBS_FILES))
    incidents = json.loads(out.reference_files["incidents.json"])["events"]
    hidden = [f'"{e["id"]}"' for e in incidents] + [f'"{e["name"]}"' for e in incidents]
    hidden.append(ref_m["run_input_sha256"])                      # the unkeyed digest is reference-only
    if public_sid != ref_m["scenario_id"]:
        hidden.append(f'"{ref_m["scenario_id"]}"')                # the private scenario name stays private
    for name, data in out.observation_files.items():
        text = data.decode()
        leaked = [t for t in OBS_FORBIDDEN_TOKENS + tuple(hidden) if t in text]
        check(f"no hidden tokens in {name}", not leaked, leaked)
    # observation timing
    check("observation timestamps observed < published <= received",
          all(sec(r["observed_at"]) < sec(r["published_at"]) <= sec(r["received_at"]) for r in obs))
    check("observations on the state grid", all(sec(r["observed_at"]) == r["simulation_second"] and r["simulation_second"] % dt == 0 for r in obs))
    check("observation zones known", all(r["zone_id"] in zone_ids for r in obs))
    check("exit reports only for gates", all(r["zone_id"] in gates for r in obs if r["variable"] == "probability_exit_open"))
    check("unique report ids", len({r["report_id"] for r in obs}) == len(obs))
    # reference states
    keys = [(r["simulation_second"], r["zone_id"], r["variable"]) for r in refs]
    check("unique reference keys", len(set(keys)) == len(keys))
    grid = sorted({r["simulation_second"] for r in refs})
    check("reference grid contiguous from 0", grid == list(range(0, grid[-1] + 1, dt)) if grid else False)
    check("every zone at every instant", len(refs) == len(grid) * len(zone_ids) * len({r["variable"] for r in refs}))
    check("reference timestamps", all(sec(r["state_at"]) == r["simulation_second"]
                                      and sec(r["label_available_at"]) == r["simulation_second"] + ref_m["label_rule"]["label_delay_seconds"]
                                      for r in refs))
    check("interval windows [T-dt, T)", all(sec(r["interval_end"]) == r["simulation_second"]
                                           and sec(r["interval_start"]) == max(0, r["simulation_second"] - dt)
                                           for r in refs if r["measurement"] == "interval"))
    check("exit applicability", all((r["applicable"] == (r["zone_id"] in gates)) for r in refs if r["variable"] == "probability_exit_open"))
    check("missing values explicit", all(r["value_valid"] == (r["value"] is not None) for r in refs + labels))
    check("reference zones known", all(r["zone_id"] in zone_ids for r in refs + labels))
    # future labels
    ref_index = {k: r for k, r in zip(keys, refs)}
    delay = ref_m["label_rule"]["label_delay_seconds"]
    last = grid[-1] if grid else -1
    check("label targets inside run", all(r["target_second"] <= last for r in labels))
    check("label horizons", all(r["target_second"] == r["issue_second"] + r["horizon_seconds"]
                                and sec(r["target_at"]) == r["target_second"] and sec(r["issued_at"]) == r["issue_second"]
                                for r in labels))
    check("label availability", all(sec(r["label_available_at"]) == r["target_second"] + delay for r in labels))
    check("labels equal the reference state at target",
          all(ref_index[(r["target_second"], r["zone_id"], r["variable"])]["value"] == r["value"]
              and ref_index[(r["target_second"], r["zone_id"], r["variable"])]["applicable"] == r["applicable"] for r in labels))
    lkeys = [(r["issue_second"], r["horizon_seconds"], r["zone_id"], r["variable"]) for r in labels]
    check("unique label keys", len(set(lkeys)) == len(lkeys))
    # artifact hashes
    for kind, files, manifest in (("observations", out.observation_files, obs_m), ("reference", out.reference_files, ref_m)):
        for name, entry in manifest["files"].items():
            check(f"{kind}/{name} hash", name in files and hashlib.sha256(files[name]).hexdigest() == entry["sha256"]
                  and len(files[name]) == entry["bytes"])
    if problems:
        raise BundleValidationError("; ".join(problems))
    return checks


def _alias_ok(alias: str) -> bool:
    try:
        validate_public_alias(alias)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- publication

def _marker(root: Path, run_id: str) -> Path:
    return root / "published" / f"{run_id}.json"


def _final(root: Path, kind: str, run_id: str) -> Path:
    return root / kind / run_id


def _manifest_sha(files: dict[str, bytes]) -> str:
    return hashlib.sha256(files["manifest.json"]).hexdigest()


def _verify_dir(directory: Path, expected_names: tuple, manifest_sha: str | None = None) -> str:
    present = sorted(p.name for p in directory.iterdir())
    if present != sorted(expected_names):
        raise BundleIntegrityError(f"{directory}: unexpected file set {present}")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha is not None and sha != manifest_sha:
        raise BundleIntegrityError(f"{directory}: manifest differs from the expected deterministic content")
    for name, entry in json.loads(manifest_bytes)["files"].items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
            raise BundleIntegrityError(f"{directory / name}: hash does not match its manifest")
    return sha


def verify_published(root: str | Path, run_id: str) -> dict:
    """Raises BundleIntegrityError unless the run is completely and consistently published."""
    root = Path(root)
    marker = _marker(root, run_id)
    if not marker.is_file():
        raise BundleIntegrityError(f"{run_id} is not published (no completion marker)")
    m = json.loads(marker.read_text(encoding="utf-8"))
    _verify_dir(_final(root, "observations", run_id), OBS_FILES, m["observation_manifest_sha256"])
    _verify_dir(_final(root, "reference", run_id), REF_FILES, m["reference_manifest_sha256"])
    return m


def is_published(root: str | Path, run_id: str) -> bool:
    try:
        verify_published(root, run_id)
        return True
    except (BundleIntegrityError, FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def list_published(root: str | Path) -> list[str]:
    d = Path(root) / "published"
    return sorted(p.stem for p in d.glob("ehts_*.json")) if d.is_dir() else []


@dataclass
class PublishResult:
    run_id: str
    status: str            # "published" | "already_published" | "recovered"
    root: Path
    summary: dict


def _write_tree(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir(parents=True)
    for name, data in files.items():
        with open(directory / name, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())


def _install(staged: Path, final: Path, expected_names: tuple, manifest_sha: str) -> bool:
    """Moves a staged bundle into place without ever replacing an existing directory.
    Returns True if installed, False if an identical bundle was already there."""
    final.parent.mkdir(parents=True, exist_ok=True)
    if not final.exists():
        try:
            os.rename(staged, final)   # fails instead of replacing if `final` appeared meanwhile
            return True
        except OSError:
            if not final.exists():
                raise
    _verify_dir(final, expected_names, manifest_sha)  # raises BundleIntegrityError on any difference
    return False


def publish_run(spec: RunSpec, root: str | Path, *, verify_determinism: bool = True,
                identity: IdentityKeyring | None = None) -> PublishResult:
    root = Path(root)
    out = build_run(spec, identity)
    if verify_determinism:
        again = build_run(spec, identity)
        if (again.observation_files, again.reference_files) != (out.observation_files, out.reference_files):
            raise BundleIntegrityError(f"{out.run_id}: identical inputs produced different outputs (nondeterminism)")
    validate_outputs(out, identity)
    obs_sha, ref_sha = _manifest_sha(out.observation_files), _manifest_sha(out.reference_files)

    marker = _marker(root, out.run_id)
    if marker.exists():
        existing = verify_published(root, out.run_id)
        if (existing["observation_manifest_sha256"], existing["reference_manifest_sha256"]) != (obs_sha, ref_sha):
            raise BundleIntegrityError(f"{out.run_id}: published content differs from the deterministic result")
        return PublishResult(out.run_id, "already_published", root, out.summary)

    staging = root / ".staging" / f"{out.run_id}.{uuid.uuid4().hex}"
    try:
        _write_tree(staging / "observations", out.observation_files)
        _write_tree(staging / "reference", out.reference_files)
        _verify_dir(staging / "observations", OBS_FILES, obs_sha)
        _verify_dir(staging / "reference", REF_FILES, ref_sha)
        installed = [
            _install(staging / "observations", _final(root, "observations", out.run_id), OBS_FILES, obs_sha),
            _install(staging / "reference", _final(root, "reference", out.run_id), REF_FILES, ref_sha),
        ]
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(f".tmp-{uuid.uuid4().hex}")
        tmp.write_text(json.dumps({"run_id": out.run_id, "observation_manifest_sha256": obs_sha,
                                   "reference_manifest_sha256": ref_sha,
                                   "published_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                  indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, marker)          # the single step that makes the run visible as complete
        verify_published(root, out.run_id)
        return PublishResult(out.run_id, "published" if all(installed) else "recovered", root, out.summary)
    finally:
        shutil.rmtree(staging, ignore_errors=True)   # only this operation's staging directory


def spec_from_json(data: dict) -> RunSpec:
    oc = data.get("observation_config", {})
    oc = {**oc, **{k: tuple(v) for k, v in oc.items() if k in ("camera_a_zones", "camera_b_zones") and v is not None}}
    if "outages" in oc:
        oc["outages"] = tuple(oc["outages"])
    return RunSpec(scenario_id=data["scenario_id"], scene=data["scene"], mode=data["mode"], epoch_utc=data["epoch_utc"],
                   sample_seconds=data.get("sample_seconds", 5), observation_config=ObservationConfig(**oc),
                   label_horizons_seconds=tuple(data.get("label_horizons_seconds", DEFAULT_HORIZONS_SECONDS)),
                   label_delay_seconds=data.get("label_delay_seconds", DEFAULT_LABEL_DELAY_SECONDS),
                   public_scenario_id=data.get("public_scenario_id"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish one deterministic Twin Studio run as validated local bundles")
    parser.add_argument("--spec", required=True, type=Path, help="Run specification JSON")
    parser.add_argument("--root", required=True, type=Path, help="Local bundle root directory")
    args = parser.parse_args()
    res = publish_run(spec_from_json(json.loads(args.spec.read_text(encoding="utf-8"))), args.root)
    print(json.dumps({"run_id": res.run_id, "status": res.status, **res.summary}, indent=2))

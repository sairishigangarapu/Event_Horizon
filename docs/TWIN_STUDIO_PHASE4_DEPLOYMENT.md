# Twin Studio Phase 4: storage, APIs and keyed identifiers

This is a **local prototype**. It is not production-ready. It runs on one host with Docker Compose, generated local secrets and GOVERNANCE-mode object lock. Nothing here was assessed for production use.

## 1. Components

| Piece | Code | Purpose |
|---|---|---|
| Keyed identifiers | `services/twin_studio/keyed_ids.py` | HMAC run IDs, report IDs and public scenario aliases |
| Bundle builder (Phase 3) | `services/twin_studio/bundle.py` | The single serializer; it now takes an optional identity keyring |
| Object store interface | `services/twin_studio/storage/objects.py` | boto3/MinIO adapter, plus an in-memory double used only by unit tests |
| Markers | `services/twin_studio/storage/markers.py` | Deterministic publication markers that pin objects by version |
| Publisher | `services/twin_studio/storage/publisher.py` | Phase 3 local publication, then conditional upload, then markers last |
| Verified readers | `services/twin_studio/storage/readers.py` | Fail-closed verification before any run is exposed |
| Observation API | `services/twin_studio/api/observation_app.py` | Serving-safe reads behind a server-controlled replay watermark |
| Reference API | `services/twin_studio/api/reference_app.py` | Authenticated evaluation sessions with a server-side watermark |
| Deployment | `deploy/twin_studio/` | Compose file, Dockerfiles, MinIO bootstrap, policies, secret initializer |

## 2. Keyed identifiers (`hmac_sha256_v1`)

**Master keys.** The keyring holds one master key `K_v` per version (`k1`, `k2`, …), each at least 32 random bytes.

**Subkeys.** Each use gets its own subkey:

```
K_v,p = HMAC-SHA256(K_v, "eh-twin/kdf/v1/" + p)
```

where `p` is one of `run-id`, `report-id` or `scenario-alias`.

**Identifiers.** Each identifier is an HMAC under the matching subkey. The message starts with a domain prefix and a NUL byte:

| Identifier | Construction |
|---|---|
| Run ID | `"ehts_" + hex(HMAC(K_v,run-id, "eh-twin/run-id/v1\0" + canonical_json(run_inputs)))[:32]` |
| Report ID | `"R-" + hex(HMAC(K_v,report-id, "eh-twin/report-id/v1\0" + run_id + "|" + source + "|" + zone + "|" + variable + "|" + t))[:20]` |
| Public scenario alias | `"scn_" + hex(HMAC(K_v,scenario-alias, "eh-twin/scenario-alias/v1\0" + scenario_id))[:16]` |

**Run-ID inputs.** `run_inputs` is exactly the Phase 3 canonical identity. It covers:
- environment, map version and venue hash
- code fingerprint and schema versions
- scenario, scene and incidents
- mode, seed, epoch and sampling
- observation config and label config

If an explicit public alias is used, `public_scenario_id` is added. Wall-clock time, UUIDs, file times and output hashes are never inputs.

**Report IDs never feed the noise RNG.** Values, dropouts and delays are identical under the keyed and unkeyed schemes. A test asserts this.

**Where the unkeyed digest lives.** The unkeyed SHA-256 of `run_inputs` appears only in the reference bundle (`run_input_sha256`, `run_inputs.json`). Validation fails if it appears in any observation file.

**Scheme and key version.** `run_id_scheme` and `id_key_version` are recorded in the reference manifest and the reference marker. They never appear on the observation side.

**Public scenario alias.** By default the public scenario identifier is the keyed alias.
- An operator may supply an approved neutral alias (`public_scenario_id`).
- An approved alias must match `^[a-z][a-z_]{2,39}$`, so it contains no digits (no seeds, times or gate numbers).
- It must not contain incident, gate, mode or zone terms.
- The original scenario name stays in the reference provenance only.

**Limit.** HMAC identifiers hide *configuration*, not *observations*. A sensor report showing a gate closed still shows the gate closed. The observation data is not anonymous.

### Key storage and rotation

- **Location.** The keyring is the file `deploy/twin_studio/secrets/id_keyring.json`. It is git-ignored and docker-ignored, and is mounted only into the `publisher` container (and the test harness) as a Compose secret.
- **Never present** in source, images, manifests, markers, artifacts, API responses or logs. Tests scan for raw, hex and uppercase hex forms of every master key and subkey.
- **Format.**
  ```json
  {"format": "eh-twin-id-keyring-v1", "active": "k2", "keys": {"k1": "<hex>", "k2": "<hex>"}}
  ```
- **Fails closed.** Publication refuses to run if:
  - the keyring is missing or malformed;
  - a key is short or has too little variety;
  - two versions share a key;
  - the active version is unknown.

  Error messages never contain key material.
- **Rotation** is explicit: `python deploy/twin_studio/init_secrets.py --rotate-id-key`.
  - It adds `k(n+1)` and makes it active. Old versions are kept.
  - Readers and APIs need no keys, so published runs stay readable. Keyed validation of an old run needs its key version, which is why old versions are retained.
  - If the same inputs were already published under an older version, the publisher returns `already_published_under_key_version` with the existing run ID.
  - A second identity is created only with `--allow-new-identity-after-rotation`.
- **No legacy runs.** No run was ever published with the Phase 3 unkeyed scheme outside temporary test directories, so no migration was needed. The unkeyed scheme (`sha256_unkeyed_v1`) remains available only for local Phase 3 bundles. It is never uploaded.

## 3. Storage

### Buckets and layout

All three buckets are created with object lock, which forces versioning. Each has a default retention, `GOVERNANCE` 30 days by default (`EH_RETENTION_MODE`/`EH_RETENTION_DAYS`).

```
eh-observations  runs/<run_id>/{observations.jsonl, venue.json, manifest.json}
eh-reference     runs/<run_id>/{reference_states.jsonl, accounting.jsonl, future_labels.jsonl,
                                incidents.json, run_inputs.json, manifest.json}
                 published/<run_id>.json      reference marker: pins every object (version, sha256, size)
eh-catalog       runs/<run_id>.json           catalog marker (sanitized), written LAST
```

**Catalog marker fields (allowlist):**
- `marker_kind`, `marker_schema_version`
- `run_id`, `scenario_id` (public alias)
- `environment_id`, `map_version`, `epoch_utc`, `sample_seconds`
- `bundle_schema_version`, `observation_schema_version`
- `observation_manifest_sha256`, `observation_objects` (pins)
- `reference_marker_sha256`

It contains no input digest, mode, seed, private scenario name, key version or reference paths.

### Identities and policies (`deploy/twin_studio/minio/policies/`)

| Identity | Used by | Allowed | Denied |
|---|---|---|---|
| root (bootstrap) | `minio`, `minio-init` (and the test harness) | everything | – |
| `eh-publisher` | `publisher` | `PutObject`/`GetObject`/`GetObjectVersion` on the artifact and marker prefixes of all three buckets; `ListBucket`/`ListBucketVersions` | explicit **Deny** on: `DeleteObject`, `DeleteObjectVersion`, `BypassGovernanceRetention`, `PutObjectRetention`, `PutObjectLegalHold`, `DeleteBucket`, bucket policy, versioning, lock and lifecycle changes |
| `eh-obs_reader` | `observation-api` | `GetObject(Version)` on `eh-observations/runs/*` and `eh-catalog/runs/*`; list on `eh-catalog` | explicit **Deny** `s3:*` on `eh-reference`; implicit deny on all writes |
| `eh-ref_reader` | `reference-api` | `GetObject(Version)` on `eh-reference/runs/*`, `eh-reference/published/*` and `eh-catalog/runs/*`; list on `eh-reference`, `eh-catalog` | explicit **Deny** `s3:*` on `eh-observations`; implicit deny on all writes |

The root credential is mounted only into `minio`, `minio-init` and the test-only `tests` service.

### Immutability: what is and isn't guaranteed

Hashes in a manifest alone prove nothing, because anyone who can rewrite both the object and the manifest can defeat them. The protections come from the storage layer:

1. **Conditional create.** Every object and marker is written with `If-None-Match: *`, so an existing key is never replaced by the publisher's code path.
2. **Object lock retention plus versioning.** No stored version can be deleted during retention:
   - The publisher is explicitly denied delete, retention and bypass actions.
   - Deleting a specific version without the governance bypass fails even for root.
3. **Version pinning.** Markers pin every artifact by version ID, SHA-256 and size. Readers fetch **by version ID** and re-hash, so a later version at the same key is ignored.
4. **First write wins.** A marker must have exactly one stored version. If anything adds a version (for example a plain PUT with the publisher credential, which conditional writes cannot forbid), the run **fails closed** for both APIs, and a republish raises an integrity error.

**Not protected against:**
- root (it can bypass GOVERNANCE);
- anyone with access to the Docker volume or the host;
- `docker compose down -v`.

COMPLIANCE mode would stop root from deleting through the API, but it would not stop host-level deletion either.

### Publication protocol (`storage/publisher.py`)

1. `build_run` executes one simulation and builds both bundles. The run is built a second time, and identical bytes are required.
2. `validate_outputs` runs with the keyring: the keyed run and report IDs are recomputed, the alias is checked, and the bundles are scanned for leaks and key material.
3. Phase 3 local staged publication (non-overwriting, marker last), then `verify_published`.
4. Each of the 9 objects is uploaded with a conditional create and read back by version ID. If an object already exists, it is reused only if it has a single version whose bytes are identical; otherwise the result is `BundleIntegrityError`.
5. The reference marker is written (conditional, verified).
6. The catalog marker is written **last**. Until it exists, neither API exposes the run.
7. Republishing an identical run returns `already_published`, after full re-verification. A run interrupted at any step is retried as `recovered`, because markers are deterministic.

### What the readers check before exposing a run

1. The catalog marker exists and has exactly one version, with the approved fields.
2. Every pinned object matches its pin (fetched by version) and its bundle manifest.
3. Manifest identities agree, and the schema versions are supported.
4. For the reference reader, additionally:
   - the reference marker has one version;
   - its SHA-256 equals the value in the catalog;
   - its pins match the catalog's observation pins.

On any failure the reader returns a generic "run not available".

## 4. Networks (`deploy/twin_studio/compose.yaml`)

| Network | internal | Members |
|---|---|---|
| `eh_publish_storage` | yes | minio, minio-init, publisher, (tests) |
| `eh_obs_storage` | yes | minio, observation-api, (tests) |
| `eh_ref_storage` | yes | minio, reference-api, (tests) |
| `eh_observation` | no | observation-api, main-app-probe, (tests), **and the future main application** |
| `eh_reference` | yes | reference-api, (tests), authorized evaluators only |

- **Main application.** It must join `eh_observation` only. The `main-app-probe` service sits exactly there and checks that the observation API is reachable while the reference API and MinIO are not.
- **Observation API.** It is published on `127.0.0.1:${EH_OBS_API_PORT:-8081}` only.
- **Reference API and MinIO.** By default they publish no host ports.
- **Host access to the reference API** requires the explicit opt-in file `compose.reference-host.yaml`. It binds `127.0.0.1:${EH_REF_API_PORT:-8082}`, and the token is still required.

**Container hardening:**
- read-only root filesystem with tmpfs `/tmp`;
- `no-new-privileges`;
- all capabilities dropped;
- the application runs as a non-root user.

## 5. APIs

### Observation API

| Route | Auth | Notes |
|---|---|---|
| `GET /health` | – | Liveness. |
| `GET /runs` | – | Fully verified runs only, with fields `run_id, scenario_id, environment_id, map_version, epoch_utc, sample_seconds, observation_schema_version`. |
| `GET /runs/{run_id}/venue` | – | Venue allowlist: `map_version, environment_id, coordinate_system, zones, obstacles, connections`. |
| `POST /replays` | control token | Creates a replay at watermark 0. |
| `POST /replays/{id}/advance` | control token | Moves forward 1–12 grid steps, capped at the run's last `received_at`. |
| `GET /replays/{id}` | – | Shows the replay state. |
| `GET /runs/{run_id}/reports?replay_id=&as_of=` | – (replay_id) | Reports with `received_at <= cutoff`. The cutoff is the watermark, or an earlier `as_of`. An `as_of` past the watermark returns 409. |
| `GET /runs/{run_id}/snapshot?replay_id=&as_of=` | – (replay_id) | Freshest received report per zone and variable, with `applicable` (exit only at gates), `covered`, `present`, `value`, `age_seconds`, `observed_at` and `received_at`. |

- **Control token.** It belongs to the operator or evaluation harness, never to the forecaster. Only its SHA-256 is mounted into the API.
- **Response fields.** All responses are built from explicit allowlists.
- **Errors.** They are generic:
  - 404 `run not available` / `replay not found`
  - 409 `incompatible schema version` / `as_of is beyond the replay watermark`
  - 422 `invalid request` (field names only, no echoed input)

### Reference API

Every route except `/health` requires `Authorization: Bearer <evaluator token>`. Only the token's SHA-256 is mounted into the service.

| Route | Semantics |
|---|---|
| `POST /sessions {run_id}` | Creates a session with `current_second = 0`, run horizons, `label_delay_seconds`, `last_state_second` and `final_second = last + delay`. |
| `POST /sessions/{id}/predictions {issue_second, predictions[]}` | Rejects the batch unless all of these hold:<ul><li>`issue_second == current_second`</li><li>the horizon is a configured horizon</li><li>`target <= last_state_second`</li><li>zone and variable are known</li><li>the value is finite</li><li>no duplicate key</li></ul>Accepted predictions are written durably (fsync + atomic replace) **before** the 201 response. |
| `POST /sessions/{id}/advance {steps 1..12}` | `current_second += steps * 5`, capped at `final_second`. There is no way to set a time. |
| `GET /sessions/{id}/labels` | Releases a label only when its `label_available_at` (= `target_at` + 60 s, from the reference bundle) is at or before the current watermark (`current_second`) **and** the prediction was registered at a watermark below that. |
| `GET /sessions/{id}/reference-states?simulation_second=&zone_id=&variable=` | One record, released only when `state_at + delay <= current_second`; otherwise 403. |
| `GET /sessions/{id}/validation` | MAE per variable and horizon over released labels only. |
| `GET /runs/{run_id}/validation` | Integrity re-verification from storage: check names and pass/fail only, with no hashes, values or digests. |

No client timestamp, header or request wall-clock time takes part in any eligibility decision. Tests send `as_of`, `current_second` and a `Date` header and confirm they are ignored.

## 6. Operating it

### Requirements

- **Disk.** Docker Desktop stores its engine disk (`docker_data.vhdx`) on C: by default.
  - Building MinIO from source needs roughly 3–5 GB of free space inside that disk: the Go toolchain image, the module cache and the build.
  - Keep at least **10 GB free** on the drive that holds the disk image.
  - On 2026-10-15, C: had 0.26 GB free. The build failed with "no space left on device", and afterwards the daemon no longer answered.
- **Network for the first build:** GitHub (MinIO and mc source) and the Go module proxy. Go verifies modules against `go.sum` and the checksum database.
- **Host tests on a full C: drive.** Pass `--basetemp=<dir on another drive>` to pytest, because the default temp directory is on C:.

### Docker recovery (operator steps; never edit the VHDX by hand)

1. Check whether the daemon responds: `docker version`. It should print a `Server:` section within a few seconds.
2. If it hangs:
   - Free space on C: (Disk Cleanup, emptying the Recycle Bin, removing large personal files), **or**
   - In Docker Desktop, go to **Settings → Resources → Advanced → Disk image location** and choose a folder on D:, then **Apply & restart**. Docker Desktop moves the disk itself. If the UI is unresponsive, quit Docker Desktop, free a little space on C:, and start it again first.
3. Optionally, once the daemon responds, `docker builder prune` removes build cache only, including the failed MinIO build's cache. It does not touch volumes or published runs.
4. Re-check with `docker version` and `docker system df` before rebuilding.

### Deployment, publication and tests (run one at a time, in this order)

```powershell
python deploy/twin_studio/init_secrets.py                                     # once; never overwrites
docker compose -f deploy/twin_studio/compose.yaml up -d --build --wait        # minio -> minio-init -> APIs
docker compose -f deploy/twin_studio/compose.yaml --profile publish run --rm publisher
docker compose -f deploy/twin_studio/compose.yaml --profile test run --rm tests
docker compose -f deploy/twin_studio/compose.yaml --profile test run --rm main-app-probe
```

### Evaluation and dashboard

- **Evaluator.** Run it from the evaluator side only: the reference token is required. Either use the opt-in loopback override, or a container on `eh_reference` + `eh_observation`.

  ```powershell
  docker compose -f deploy/twin_studio/compose.yaml -f deploy/twin_studio/compose.reference-host.yaml up -d
  $env:EH_OBS_API_URL="http://127.0.0.1:8081"; $env:EH_REF_API_URL="http://127.0.0.1:8082"
  $env:EH_OBS_CONTROL_TOKEN_FILE="deploy/twin_studio/secrets/observation_control_token"
  $env:EH_REF_TOKEN_FILE="deploy/twin_studio/secrets/reference_api_token"
  python -m services.twin_studio.evaluation --run-id <ehts_...> --out data/twin_evaluations
  ```

- **Dashboard.** Run `streamlit run dashboard/app.py` and open the **Twin Studio Evaluation** page. It reads `data/twin_evaluations` (or `EH_TWIN_EVALUATIONS_DIR`) only, and never gets a token.

### Secrets: initialisation and recovery

- **Lost tokens.** API tokens and MinIO service passwords are local, dev-only values.
  1. Stop the stack.
  2. Delete only the affected secret file **and** its `.sha256`.
  3. Rerun `init_secrets.py`.
  4. For MinIO users, rerun `minio-init`, which re-applies the users with the new secret.
- **Never delete or regenerate `id_keyring.json` for a deployment with published runs.** Rotate it instead with `--rotate-id-key`. Old versions are kept so earlier runs remain verifiable.
- **Root credentials.** Changing `minio_root_*` for an existing `minio_data` volume requires MinIO's own credential-rotation procedure.

### Cleanup: safe vs destructive

| Command | Effect |
|---|---|
| `docker compose -f deploy/twin_studio/compose.yaml stop` / `down` | **Safe.** Stops and removes containers and networks; volumes (published runs, sessions) are kept. |
| `docker compose ... rm -f tests main-app-probe` | **Safe.** Removes test containers. |
| `docker builder prune` | **Safe for data.** Removes build cache only; the next build is slower. |
| `docker compose -f deploy/twin_studio/compose.yaml down -v` | **DESTRUCTIVE.** Deletes `minio_data` (all published runs, which object lock cannot protect from volume deletion) and all replay and session state. Use only for a deliberate local reset. |
| `docker system prune --volumes` | **DESTRUCTIVE**, and affects other projects too. Do not use. |

### Pinned images and their evidence

- **MinIO server.** `minio/minio` and `minio/mc` were removed from Docker Hub (September 2026), and quay.io answers 401 for every tag. The server is therefore built from the official tag `RELEASE.2025-10-15T17-29-55Z`. The build checks that the tag resolves to commit `9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a`.
- **CVE-2025-62506 (GHSA-jjjj-jwhf-8rgr, high).**
  - The GitHub advisory lists the first patched version as fix commit `c1a49490c78e9c3ebcad86ba0662319138ace190` ("fix: check sub-policy properly when present", which changes `cmd/iam.go`).
  - The GitHub compare API shows the pinned commit is **1 commit ahead and 0 behind** that fix. The one extra commit is a documentation change. The pinned **source** therefore contains the fix.
  - Not yet checked: the built binary's version string, because the image build has not completed (Docker blocked).
  - This is evidence for that one CVE, not a general claim that the release is secure.
- **mc client.** Built from tag `RELEASE.2025-08-13T08-35-41Z` (commit `7394ce0dd2a80935aded936b09fa12cbb3cb8096`). It is used only by the one-shot `minio-init` container.
- **Base images:** `golang:1.24.9-bookworm` (build) and `python:3.12.10-slim-bookworm` (runtime; pulled successfully on 2026-10-15).
- **Python dependencies** are pinned in `deploy/twin_studio/requirements.txt`.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `minio-init` exits 3 or retention errors | The bucket was created without object lock, which cannot be added afterwards. Reset with `down -v`. |
| A run is missing from `/runs` | Not published, publication interrupted, or failed verification. Rerun the publisher (it resumes); check the API logs, which give only the run ID and the failing step. |
| `already_published_under_key_version` | The inputs were published before a key rotation. Use the existing run ID, or pass `--allow-new-identity-after-rotation` deliberately. |
| Docker build fails with "no space left on device" | Docker Desktop's disk image (`docker_data.vhdx`) is full. Free space, or move the image (Settings → Resources → Advanced). |

## 7. Known limits of the prototype

- Single-node MinIO, GOVERNANCE retention, a local volume. Not durable or tamper-proof against the host or root.
- Session and replay state are JSON files on a container volume. There is no multi-instance coordination. Do not scale the APIs horizontally.
- Tokens are static bearer tokens: no expiry, no per-evaluator identity, and no TLS inside the Compose networks.
- The observation API's reader routes are unauthenticated and rely on network placement plus the replay capability. The replay ID is a random capability, not a secret credential.
- The integrity of a run depends on the first published marker. An attacker holding the publisher credential cannot replace a run, but can make it unavailable by adding marker versions; that is detected and the run fails closed.
- MinIO is built from source because no official pinned image is pullable. Rebuilding requires GitHub and the Go module proxy.

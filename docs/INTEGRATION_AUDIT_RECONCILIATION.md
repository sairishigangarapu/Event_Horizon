# Main / Parallel / live frontend integration

Date: 15 October 2026. This is the current reconciliation record; the original audit reports remain historical evidence, not a list of currently reproducible defects.

## Inputs and preservation

- Current main: `8f7cf07ac9aac529d7ce852ee32f713a3ec7eccf` (Twin Studio v3 and A* integration).
- Audit branch parallel: `214fc6a3a60070083d1a38e4da0fe6174e26d8ac`.
- Shared base: `e997fdd32f96097e2b6521050f67f6034ce9157d`.
- Earlier combined dashboard on bugbounty: `6fe88e574f8789130af2a3c88dc43961966601ec`, plus the locally reviewed random-flow controls, provider and evaluator.

The clean integration checkout reproduced current main's exact tree `9d23764e2e59ed1d1eb34412c30100aa8e87b404` before applying audit/frontend changes. Every materialized remote file was checked against its Git blob SHA. Main's v3 source engine, archive provenance, observation/reference storage boundary, trained environment artifacts, staleness policy and A* implementation were retained. The v3 adapter received one UTC-boundary compatibility fix; archived source copies and model weights were not rewritten.

The audit's `docs/project_report.pdf` is byte-identical to the already committed workload plan. Audit prose and duplicate report copies are retained for provenance. Their old severity/status assertions must be checked against this document and current tests.

## Changes that matter at integration boundaries

| Boundary | Reconciled behavior |
| --- | --- |
| Shared API authentication | Optional API-key mode fails closed when the key is missing/unconfigured. Inbound APP_API_KEY is independent of upstream CENTRAL_API_KEY. Core, runtime/world, Studio proxy and v3 API routes share the guard; readiness/config/health stay accessible. |
| Browser access | Session-only access control sends keys in headers, never URLs. CSV downloads use authenticated fetch. The legacy two-service Studio proxy forwards the header. |
| Resource bounds | Production requests use a bounded process-local token bucket. API write bodies are bounded before model work, including bodies without Content-Length. Demo/test/development remain account-free. |
| Forecast pagination | SQL pages distinct scenario/zone/variable/unit/issue/target keys and selects the latest revision per stream. It no longer loads an entire scenario before Python slicing. |
| Existing DBs | Forecast channel masks, missing-data masks and applicability are added idempotently without deleting old rows. Legacy absent masks remain unknown. SQLite foreign keys are enabled; non-SQLite URLs are not treated as filesystem paths. |
| Model integrity | Missing artifacts fail explicitly; serving enforces eval mode. Deterministic forecast identities include model provenance. Tree cache keys include manifest hash/file fingerprint and use a lock. The manifest API reports the actual serving artifact version and separately flags any stale DB registry version. |
| Evidence | The public path delegates to the canonical five-label reliability engine. Coverage is the fresh fraction of the historically received source roster, not an age-decay substitute. Conflict provenance/counts are available separately. A full expected deployment roster still requires source configuration. |
| Channel preservation | All six D channels remain direct temporal inputs and spatial inputs. Six-element forecast masks and model versions survive runtime serialization and CSV export. |
| Person 4 timestamps | UTC normalization, issue/target/horizon/unit-aware pairing, deterministic revisions, future-received exclusion, unit-aware labels and released-label checks. Sequence input also enforces the shared observation contract and scenario scope. |
| Twin adapter UTC | The older Phase 5 adapter now normalizes both epoch and pipeline target timestamps before subtraction. This fixed failures exposed when previously skipped Phase 4/5/7 tests were enabled. |
| CSV provider inputs | Required columns, identities, units, timestamps, finite values, six binary masks and immutable forecast IDs are checked. Empty exports retain their schema. |
| Unknown capacities | Main's explicit unknown-capacity graph behavior is retained. New route zones no longer invent a physical capacity from area. |
| New frontend | `/dashboard/` runs curved walkable routes, density/direction/duration controls, hidden disturbances, delayed reports, real forecasts, incident scoring and bounded clearance comparisons. Studio stays at `/`. |
| Packaging | The hosted image now includes services and environment-model paths required by main's v3 endpoints. The independently pinned v3 observation/reference service requirements remain intact. |

## Architecture decisions and implementation scope

1. **D is never one trust score.** It remains six ordered channels; the UI evidence index is separate and is not an input to forecasting.
2. **The checkpoint layout is fixed.** U is raw(6), age(6), observation-mask(6), D(6), H(16), totaling 40; history is 12 five-second steps. Tree/applicability masks are explicit spatial/output metadata, not silently added GRU slots. Changing the layout requires retraining/versioning.
3. **Node/edge representations are generated.** The hypernetwork does not generate a different GNN architecture per tick. The GNN/GRU serving weights do not change during replay.
4. **Truth is not inference input.** The live provider sends only released reports into inference, then current labels separately to retrospective evaluation. One hosted operator process is logical separation; the v3 observation/reference services offer a different access-controlled deployment boundary. Neither is misrepresented as universal process isolation.
5. **The two forecast streams stay separate.** Stream A is an observation-only synthetic EWMA/persistence provider unless an external feed is actually configured. Comparisons cannot mix issue times, targets, units, scenarios or horizons.
6. **Model environments stay separate.** Route-context mc_v2, original production artifacts and canonical-zone v3 models retain their own feature/graph contracts and manifests. The new route frontend is not claimed to consume the v3 58-feature bank, and its clearance trials are not claimed to be the separate v3 A* router.
7. **Monte Carlo scope is clarified.** The historical mc_v2 neural/tree joint search is preserved as an experiment. Current offline training selects tree depth/cluster count only, cross-fits whole training scenarios for neural inputs, uses a predeclared neural configuration, and reports final test metrics without using them for selection. A failed validation gate leaves the new candidate trained_not_validated and inactive. Initial supervised bootstrapping is distinct from a live feedback loop.
8. **Feedback/promotion remains pending.** Error-driven training of a copied HNet/GNN pair with trees and GRU frozen, an active compatibility/quality promotion gate and rollback are not implemented by these audit fixes. Pair checkpoint serialization alone is not that workflow. Documentation now labels this gap explicitly.
9. **No safety/optimality claim.** Synthetic threshold incidents, false positives/misses and best-among-tested clearance assumptions remain visible. The new curved stress recordings showed no advance warning before their first incidents. No retraining or threshold tuning was used to hide that result.

## Audit disposition

- F-001/002/003/012/014: readiness labels, deterministic inference, missing-artifact refusal and observation-only Stream A were already substantially fixed at the shared base. Preserve and regression-test them; do not claim newly discovered truth leakage or newly validated real-world models.
- F-004/007/008/015/016/023/028/031/032/033/036: authentication, bounded requests, SQL pagination, locked cache, FK enforcement, additive DB upgrade/explicit production initialization, inactive candidate defaults, typed artifact errors and same-origin mutation checks are integrated or completed here.
- F-006/017: carry the legacy dashboard's escaping/None geometry fixes; verify its headless render.
- F-009/010/011/025/026/029/038: wire canonical evidence assessment; retain scaler/masks; validate CSV; record deterministic version-aware identities; keep numerical normalization bounded. Missing values are never asserted to be observed zeros.
- F-005/018/020/037/039/041: retain Parallel's CLI entry point, seed handling and explicit regeneration/overwrite behavior. Local CLI file access is operator-authorized; it is not a remote file-upload vulnerability merely because absolute paths are accepted.
- F-019/021/022/024/027/030/034/035/040: document scope/configuration rather than overstate blanket remediation. Upstream HTTP provider integration is not implemented merely by reading a key; environment extras remain compatibility-tolerant; local HTTP is intentional and Railway terminates public TLS. Generic server errors do not return exception internals. DB initialization preserves existing records instead of silently clearing them.

## Verification record

See the final validation counts below. Tests cover both preserved Twin paths and the new live frontend, not only the main branch's old API.

- Full main suite, including legacy Streamlit, Twin Phase 1–7 and v3: 414 passed; no skips (full in-process suite).
- Person 4 suite: 57 passed.
- Studio Python: 18 passed.
- Studio JavaScript: 17 engine + 11 regression + 11 UI/math + 1 ordered-transport check passed.
- Actual two-service HTTP roundtrip: passed, through 35 s with released observations, 144 comparisons, matured metrics, UTF-8 report and hidden future actuals.
- Actual single-port hosted HTTP roundtrip: passed, including render, model inference, matured labels and UTF-8 report.
- Authenticated frontend/real-model transport: passed random curve generation, 35 s runtime, forecast rows, action comparison and authenticated CSV download. This is a DOM/transport test, not a visual browser screenshot test.
- Staged hosted-image filesystem: passed app startup, static/access assets and preserved v3 API import/routing from only Docker-copied paths.
- Offline trainer smoke: two training scenarios, disjoint validation/test sets, one epoch; tree-only selection, scenario OOS folds, failed gate marked trained_not_validated, candidate inactive. No committed weights were modified or promoted.

Installing the missing YAML test dependency exposed seven older integration failures/errors; normalizing the Phase 5 adapter UTC boundary corrected them. The complete Phase 4/5/7 targeted suite then passed 70 tests.

Actual Docker image build and live MinIO/Compose network/authorization integration were not run on this host because Docker is unavailable. Their test harness remains under tests/twin_phase4_integration and requires the explicit documented Compose environment. GitHub workflow execution is separate from local verification; this document does not claim a green remote CI run. Browser visual verification remains pending.

## Publication boundary

The requested integration is committed to main. Railway remains attached to bugbounty; this integration does not update that branch or redeploy the live app. Main's GitHub workflow now runs integration checks only and has no deployment job. Promoting this frontend to the public service remains a separate approval/action.

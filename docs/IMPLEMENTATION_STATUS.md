# Integrated implementation status — 15 October 2026

This supersedes the pre-integration status snapshots in the walkthrough, tree handoff and spatial notes. The architecture contract is [ARCHITECTURE.md](ARCHITECTURE.md); branch provenance, audit disposition and test evidence are in [INTEGRATION_AUDIT_RECONCILIATION.md](INTEGRATION_AUDIT_RECONCILIATION.md).

| Component | Current behavior | Boundary / remaining gap |
| --- | --- | --- |
| Studio | User routes, three rendered scenarios, separate observable/truth outputs | Exact analysis exports are operator/evaluation data, not inference input |
| Live random-flow dashboard | Seeded curved walkable routes, initial density, direction/duration controls, hidden random disturbances, one-second ticks | Synthetic kinematics; route-corridor density is not venue-wide occupancy |
| Observation ingestion | Finite/unit/timestamp checks, immutable report identities, released reports only | Caller/provider owns the replay clock; full archives contain later reports |
| Contextual trees | Six channels, 42-feature observed local/neighbor context, training-fitted cluster routing/fallback | Not pedestrian-identity clustering; no online tree updates |
| Spatial models | Hypernetwork generates node/edge representations; edge-aware GNN; immutable serving state | No per-tick GNN architecture generation; custom geometry transfer unvalidated |
| Temporal model | 40-D U, 12 steps, +30/+60/+120 s forecasts every 5 s | Not individual trajectory prediction or a learned exact incident-onset time |
| Comparison | Separate EWMA baseline and architecture streams; exact issue/target/unit alignment | EWMA is a synthetic provider, not an external authority |
| Information assessment | Canonical five-label engine, independent-source conflict graph, coverage of received source roster | UI evidence index is a heuristic, not calibrated correctness or accuracy |
| Evaluation | Matured labels only, MAE/RMSE, synthetic sustained-incident outcomes, misses/false alarms | New curved-flow stress recordings show missed advance warnings and false positives |
| Clearance planner | Tests continue/meter/clear/combined from current observable estimates | Best among tested assumptions; clearing requires inspection; outside queues remain counted |
| Twin v3 | Preserved source-of-truth engine, sensing, replay, trained models, staleness and A* modules | Separate canonical-zone/model contract; neural gate is not passed; persisted-output APIs are read-only |
| API / DB | Shared optional fail-closed key boundary, production rate limit, SQL pagination, additive forecast-mask upgrade | Local demo is deliberately account-free; no claim of multi-user tenant isolation |
| Offline trainer | Tree-only parameter selection, whole-scenario cross-fit tree inputs, fixed neural bootstrap, validation gate/status | Historical mc_v2 artifact is unchanged and retains its original experimental provenance |
| Feedback / promotion | Serving frozen; checkpoints hash-verified and versioned | Error-driven pair training, gated active promotion and rollback remain pending |
| Deployment | Railway continues to follow bugbounty | Updating main does not approve or publish this frontend to the live branch |

Model directories are never silently interchanged: original `models/production`, historical `models/candidates/mc_v2` and `models/environments/eh_twin_studio/v3` remain distinct. A newly trained candidate is not made active just because its files exist.

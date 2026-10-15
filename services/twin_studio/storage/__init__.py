"""Object-storage publication and verified readers for Twin Studio runs (Phase 4).

Buckets (MinIO, object lock + versioning, see deploy/twin_studio):
  eh-observations  runs/<run_id>/{observations.jsonl, venue.json, manifest.json}
  eh-reference     runs/<run_id>/{reference_states.jsonl, accounting.jsonl, future_labels.jsonl,
                                  incidents.json, run_inputs.json, manifest.json}
                   published/<run_id>.json            reference-side publication marker
  eh-catalog       runs/<run_id>.json                 sanitized catalog marker, written LAST
"""

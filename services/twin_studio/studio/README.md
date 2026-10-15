# Event Horizon Twin Studio (original app, unmodified)

`event_horizon_twin_studio/` is a byte-identical extraction of `Event_Horizon_Twin_Studio.zip`
(SHA-256 and per-file hashes in `SOURCE.json`). It is the original 3D studio: route drawing,
incidents, ideal/custom/random renders, CSV export. Nothing in it is changed.

Start it (binds 127.0.0.1 only, as the original server does):

```powershell
cd services/twin_studio/studio/event_horizon_twin_studio
python lab_server.py --port 8000          # or START_WINDOWS.bat
```

Open http://127.0.0.1:8000/.

Relationship to the integrated subsystem:
- The studio runs its own original engines (`simulation.py`, `dist/engine.js`). They have no
  `gate_closure` incident; the integrated engines in `services/twin_studio/engine/` add it and are
  otherwise output-identical (fidelity audit: 123/123 runs, Python and JavaScript).
- Published runs, keyed identifiers, APIs, labels and the evaluation dashboard page are the
  integration layers (Phases 3-7); the studio does not read or write them.

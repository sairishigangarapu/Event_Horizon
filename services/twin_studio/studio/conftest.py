# The exact archive tree is kept unmodified, including its own test scripts (which start a local
# server). They are run deliberately, not collected by a repository-wide pytest run.
collect_ignore_glob = ["event_horizon_twin_studio/*"]

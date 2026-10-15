#!/bin/sh
# One-time (idempotent) MinIO bootstrap for Twin Studio Phase 4. Runs in the minio-init container,
# the ONLY place the root credential is used besides the MinIO server itself. Prints no secrets.
set -eu
S="${EH_SECRETS_DIR:-/run/secrets}"
r() { tr -d '\r\n' < "$S/$1"; }
export MC_CONFIG_DIR="${MC_CONFIG_DIR:-/tmp/mc}"
MODE="${EH_RETENTION_MODE:-GOVERNANCE}"
DAYS="${EH_RETENTION_DAYS:-30}"
case "$MODE" in GOVERNANCE|COMPLIANCE) ;; *) echo "EH_RETENTION_MODE must be GOVERNANCE or COMPLIANCE" >&2; exit 2;; esac

mc alias set admin "${EH_MINIO_URL:-http://minio:9000}" "$(r minio_root_user)" "$(r minio_root_password)" >/dev/null

for b in eh-observations eh-reference eh-catalog; do
  mc mb --ignore-existing --with-lock "admin/$b" >/dev/null
  # Fail if an existing bucket was created without object lock (cannot be added afterwards).
  mc retention set --default "$MODE" "${DAYS}d" "admin/$b" >/dev/null
  mc version info "admin/$b" | grep -q "versioning is enabled" || { echo "bucket $b is not versioned" >&2; exit 3; }
  echo "bucket $b: object lock default $MODE ${DAYS}d, versioning enabled"
done

for u in publisher obs_reader ref_reader; do
  mc admin policy create admin "eh-$u" "${EH_POLICY_DIR:-/init/policies}/$u.json" >/dev/null
  mc admin user add admin "$(r minio_${u}_access_key)" "$(r minio_${u}_secret_key)" >/dev/null
  out=$(mc admin policy attach admin "eh-$u" --user "$(r minio_${u}_access_key)" 2>&1) || \
    echo "$out" | grep -qi "already" || { echo "policy attach failed for $u" >&2; exit 4; }
  echo "user for $u: policy eh-$u only"
done
echo "minio-init complete"

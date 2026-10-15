# MinIO server + mc client built from official source at pinned release tags (Phase 4).
#
# Why from source: minio/minio and minio/mc were removed from Docker Hub (Sept 2026) and quay.io/minio
# now answers 401 for every tag, so no official pinned image is pullable. Building the tagged source is
# the only option that is both official and version-pinned. Each tag is checked against its commit so a
# moved tag fails the build. Commit 9e49d5e7 is a descendant of c1a49490, the fix commit named by
# GHSA-jjjj-jwhf-8rgr (CVE-2025-62506); see docs/TWIN_STUDIO_PHASE4_DEPLOYMENT.md section 6.
FROM golang:1.24.9-bookworm AS build
ARG MINIO_RELEASE=RELEASE.2025-10-15T17-29-55Z
ARG MINIO_COMMIT=9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a
ARG MC_RELEASE=RELEASE.2025-08-13T08-35-41Z
ARG MC_COMMIT=7394ce0dd2a80935aded936b09fa12cbb3cb8096
ENV CGO_ENABLED=0 GOTOOLCHAIN=local GOFLAGS=-trimpath
RUN git clone --quiet --depth 1 --branch "$MINIO_RELEASE" https://github.com/minio/minio /src/minio \
 && cd /src/minio && test "$(git rev-parse HEAD)" = "$MINIO_COMMIT" \
 && go build -ldflags "-s -w $(go run buildscripts/gen-ldflags.go)" -o /out/minio .
RUN git clone --quiet --depth 1 --branch "$MC_RELEASE" https://github.com/minio/mc /src/mc \
 && cd /src/mc && test "$(git rev-parse HEAD)" = "$MC_COMMIT" \
 && go build -ldflags "-s -w $(go run buildscripts/gen-ldflags.go)" -o /out/mc .

FROM python:3.12.10-slim-bookworm
COPY --from=build /out/minio /out/mc /usr/local/bin/
RUN useradd --system --uid 10002 --home-dir /data minio && mkdir -p /data && chown minio /data
USER minio
EXPOSE 9000
ENTRYPOINT ["/usr/local/bin/minio"]

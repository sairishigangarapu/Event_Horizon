"""Minimal object-store interface used by the publisher and the readers.

`S3ObjectStore` wraps a boto3 S3 client pointed at MinIO. `MemoryObjectStore` mimics the same
semantics in memory (versioning, conditional create, no delete) for fast unit tests only; the
storage guarantees themselves are tested against the real MinIO deployment
(tests/twin_phase4_integration).
"""
from __future__ import annotations

import base64
import hashlib
import os
import uuid
from pathlib import Path

OBSERVATION_BUCKET = os.environ.get("EH_OBSERVATION_BUCKET", "eh-observations")
REFERENCE_BUCKET = os.environ.get("EH_REFERENCE_BUCKET", "eh-reference")
CATALOG_BUCKET = os.environ.get("EH_CATALOG_BUCKET", "eh-catalog")


class StorageError(RuntimeError):
    """Base class; messages contain bucket/key names only, never credentials."""


class ObjectNotFound(StorageError):
    pass


class ObjectExists(StorageError):
    """A conditional create found an existing object (If-None-Match: * refused)."""


class AccessDenied(StorageError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore:
    def put_if_absent(self, bucket: str, key: str, data: bytes) -> str:
        """Creates the object only if no object exists at `key`; returns its version ID."""
        raise NotImplementedError

    def get(self, bucket: str, key: str, version_id: str | None = None) -> tuple[bytes, str]:
        raise NotImplementedError

    def versions(self, bucket: str, key: str) -> list[str]:
        """All version IDs (and delete markers) stored exactly at `key`."""
        raise NotImplementedError

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        raise NotImplementedError


class S3ObjectStore(ObjectStore):
    def __init__(self, client):
        self.client = client

    @staticmethod
    def _code(exc) -> str:
        return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))

    def _map(self, exc, bucket: str, key: str):
        code = self._code(exc)
        if code in ("PreconditionFailed", "ConditionalRequestConflict", "412"):
            return ObjectExists(f"{bucket}/{key} already exists")
        if code in ("NoSuchKey", "NoSuchVersion", "404", "NotFound", "NoSuchBucket"):
            return ObjectNotFound(f"{bucket}/{key} not found")
        if code in ("AccessDenied", "403", "Forbidden", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
            return AccessDenied(f"access denied for {bucket}/{key}")
        return StorageError(f"storage error {code or type(exc).__name__} for {bucket}/{key}")

    def put_if_absent(self, bucket, key, data):
        from botocore.exceptions import ClientError
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode()
        try:
            resp = self.client.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*",
                                          ChecksumSHA256=checksum, ContentType="application/octet-stream",
                                          Metadata={"sha256": sha256_hex(data)})
        except ClientError as exc:
            raise self._map(exc, bucket, key) from None
        version = resp.get("VersionId")
        if not version or version == "null":
            raise StorageError(f"{bucket} is not versioned; refusing to publish without version pinning")
        return version

    def get(self, bucket, key, version_id=None):
        from botocore.exceptions import ClientError
        kwargs = {"Bucket": bucket, "Key": key}
        if version_id:
            kwargs["VersionId"] = version_id
        try:
            resp = self.client.get_object(**kwargs)
            return resp["Body"].read(), resp.get("VersionId")
        except ClientError as exc:
            raise self._map(exc, bucket, key) from None

    def versions(self, bucket, key):
        from botocore.exceptions import ClientError
        out, kwargs = [], {"Bucket": bucket, "Prefix": key}
        try:
            while True:
                resp = self.client.list_object_versions(**kwargs)
                out += [v["VersionId"] for v in resp.get("Versions", []) if v["Key"] == key]
                out += ["delete-marker:" + v["VersionId"] for v in resp.get("DeleteMarkers", []) if v["Key"] == key]
                if not resp.get("IsTruncated"):
                    return out
                kwargs.update(KeyMarker=resp["NextKeyMarker"], VersionIdMarker=resp["NextVersionIdMarker"])
        except ClientError as exc:
            raise self._map(exc, bucket, key) from None

    def list_keys(self, bucket, prefix):
        from botocore.exceptions import ClientError
        keys, kwargs = [], {"Bucket": bucket, "Prefix": prefix}
        try:
            while True:
                resp = self.client.list_objects_v2(**kwargs)
                keys += [o["Key"] for o in resp.get("Contents", [])]
                if not resp.get("IsTruncated"):
                    return keys
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except ClientError as exc:
            raise self._map(exc, bucket, prefix) from None


class MemoryObjectStore(ObjectStore):
    """Unit-test double with versioned, create-only semantics. Not a substitute for MinIO tests."""

    def __init__(self, readable: set[str] | None = None):
        self.objects: dict[tuple[str, str], list[tuple[str, bytes]]] = {}
        self.readable = readable          # None = all buckets; otherwise simulated per-bucket access
        self.fail_after_puts: int | None = None
        self.puts = 0

    def _check(self, bucket):
        if self.readable is not None and bucket not in self.readable:
            raise AccessDenied(f"access denied for {bucket}")

    def put_if_absent(self, bucket, key, data):
        if self.fail_after_puts is not None and self.puts >= self.fail_after_puts:
            raise StorageError("simulated interruption")
        self.puts += 1
        if (bucket, key) in self.objects:
            raise ObjectExists(f"{bucket}/{key} already exists")
        version = uuid.uuid4().hex
        self.objects[(bucket, key)] = [(version, bytes(data))]
        return version

    def force_put(self, bucket, key, data):
        """Simulates an unconditional overwrite: a NEW version on top of the existing ones."""
        version = uuid.uuid4().hex
        self.objects.setdefault((bucket, key), []).append((version, bytes(data)))
        return version

    def get(self, bucket, key, version_id=None):
        self._check(bucket)
        if (bucket, key) not in self.objects:
            raise ObjectNotFound(f"{bucket}/{key} not found")
        versions = self.objects[(bucket, key)]
        if version_id is None:
            return versions[-1][1], versions[-1][0]
        for v, d in versions:
            if v == version_id:
                return d, v
        raise ObjectNotFound(f"{bucket}/{key} version not found")

    def versions(self, bucket, key):
        self._check(bucket)
        return [v for v, _ in self.objects.get((bucket, key), [])]

    def list_keys(self, bucket, prefix):
        self._check(bucket)
        return sorted(k for b, k in self.objects if b == bucket and k.startswith(prefix))


# ---------------------------------------------------------------- credentials / client

def read_secret_file(path: str | os.PathLike) -> str:
    p = Path(path)
    if not p.is_file():
        raise StorageError(f"credential file {p.name} not found")
    value = p.read_text(encoding="utf-8").strip()
    if not value:
        raise StorageError(f"credential file {p.name} is empty")
    return value


def s3_store_from_env(prefix: str) -> S3ObjectStore:
    """Builds a client from <prefix>_ACCESS_KEY_FILE / <prefix>_SECRET_KEY_FILE and EH_S3_ENDPOINT."""
    import boto3
    from botocore.config import Config
    endpoint = os.environ.get("EH_S3_ENDPOINT", "http://minio:9000")
    access = read_secret_file(os.environ[f"{prefix}_ACCESS_KEY_FILE"])
    secret = read_secret_file(os.environ[f"{prefix}_SECRET_KEY_FILE"])
    return s3_store(endpoint, access, secret)


def s3_store(endpoint: str, access_key: str, secret_key: str) -> S3ObjectStore:
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                          region_name="us-east-1",
                          config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                        retries={"max_attempts": 2, "mode": "standard"},
                                        connect_timeout=5, read_timeout=30))
    return S3ObjectStore(client)

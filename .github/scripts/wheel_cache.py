"""S3 wheel manifests with a conditional-write lease for one producer per key.

The lease is longer than the entire build job's timeout. A cancelled/failed job
releases it in an always() step; a lost runner is recovered after lease expiry.
Publishing uses the lease's ETag, so an old producer cannot replace a newer one.
Readers only consume ready manifests, published after validation and upload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote

from botocore.exceptions import ClientError

BUCKET = "gha-artifacts"
PREFIX = "tritonci-artifacts/wheel-cache/v1"
LEASE_SECONDS = 4 * 60 * 60  # Must exceed build.yml's 180-minute job timeout.
WAIT_SECONDS = 90 * 60
RETENTION_SECONDS = 14 * 24 * 60 * 60


def error_code(exc):
    return exc.response["Error"]["Code"]


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def build_identity():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    recipe = hashlib.sha256()
    # Tooling comes from the workflow revision, even when building an older ref.
    tooling = Path(__file__).resolve().parents[1]
    for path in (tooling / "workflows/build.yml", Path(__file__)):
        recipe.update(path.read_bytes())
    return {
        "repository": os.environ["GITHUB_REPOSITORY"],
        "sha": sha,
        "abi": "cp312",
        "platform": f"{os.environ['RUNNER_OS']}-{os.environ['RUNNER_ARCH']}",
        "recipe": recipe.hexdigest(),
    }


class WheelCache:

    def __init__(self, s3, identity, clock=time.time, sleep=time.sleep):
        self.s3 = s3
        self.identity = identity
        self.clock = clock
        self.sleep = sleep
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.key = f"{PREFIX}/{identity['repository']}/{identity_hash}.json"

    def read(self):
        try:
            response = self.s3.get_object(Bucket=BUCKET, Key=self.key)
        except ClientError as exc:
            if error_code(exc) in ("NoSuchKey", "404"):
                return None, None
            raise
        return json.loads(response["Body"].read()), response["ETag"]

    def write(self, value, etag):
        condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            response = self.s3.put_object(Bucket=BUCKET, Key=self.key, Body=json.dumps(value).encode(),
                                          ContentType="application/json", **condition)
        except ClientError as exc:
            if error_code(exc) in ("PreconditionFailed", "ConditionalRequestConflict", "412", "409"):
                return None
            raise
        return response["ETag"]

    def ready(self, state):
        if not state or state.get("status") != "ready" or state.get("identity") != self.identity:
            return False
        # Leave enough lifetime for downstream tests to queue and download.
        if state["expires_at"] <= self.clock() + LEASE_SECONDS:
            return False
        wheel = state["wheel"]
        if not re.fullmatch(r"[a-f0-9]{64}", wheel["sha256"]):
            return False
        try:
            obj = self.s3.head_object(Bucket=BUCKET, Key=wheel["key"])
        except ClientError as exc:
            if error_code(exc) in ("NoSuchKey", "404", "NotFound"):
                return False
            raise
        return obj["ContentLength"] == wheel["size"]

    def acquire(self, owner):
        deadline = self.clock() + WAIT_SECONDS
        while True:
            state, etag = self.read()
            if self.ready(state):
                return state, None
            if not state or state.get("status") != "building" or state["expires_at"] <= self.clock():
                lease = {
                    "status": "building", "identity": self.identity, "owner": owner, "expires_at":
                    self.clock() + LEASE_SECONDS
                }
                lease_etag = self.write(lease, etag)
                if lease_etag:
                    return None, {"key": self.key, "etag": lease_etag, "identity": self.identity}
            if self.clock() >= deadline:
                raise TimeoutError(f"Timed out waiting for wheel producer: {self.key}")
            print(f"Waiting for the wheel producer: {self.key}", flush=True)
            self.sleep(30)

    def publish(self, lease, wheel):
        state = {
            "status": "ready", "identity": self.identity, "expires_at": self.clock() + RETENTION_SECONDS, "wheel": wheel
        }
        if not self.ready(state):
            raise RuntimeError("Uploaded wheel is missing or has the wrong size")
        if not self.write(state, lease["etag"]):
            raise RuntimeError("Lost the wheel build lease; refusing to publish")
        return state

    def release(self, lease):
        # A successful publish changes the ETag, making this a harmless no-op.
        self.write({"status": "failed", "identity": self.identity, "expires_at": 0}, lease["etag"])


def emit(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def emit_wheel(state):
    wheel = state["wheel"]
    url = f"https://{BUCKET}.s3.us-east-1.amazonaws.com/{quote(wheel['key'], safe='/')}"
    emit(wheel_url=url, wheel_filename=wheel["key"].rsplit("/", 1)[-1], wheel_sha256=wheel["sha256"])


def main():
    import boto3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "publish", "release"))
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    state_path = Path(os.environ["RUNNER_TEMP"]) / "triton-wheel-lease.json"
    s3 = boto3.client("s3", region_name="us-east-1")
    if args.command == "acquire":
        cache = WheelCache(s3, build_identity())
        owner = f"{os.environ['GITHUB_RUN_ID']}/{os.environ['GITHUB_RUN_ATTEMPT']}"
        state, lease = cache.acquire(owner)
        if lease:
            state_path.write_text(json.dumps(lease))
            emit(hit="false")
            print(f"Acquired wheel build lease: {cache.key}")
        else:
            emit(hit="true")
            emit_wheel(state)
            print(f"Reusing validated wheel: {state['wheel']['key']}")
    elif state_path.exists():
        lease = json.loads(state_path.read_text())
        cache = WheelCache(s3, lease["identity"])
        if args.command == "release":
            cache.release(lease)
            state_path.unlink()
        else:
            objects = json.loads(os.environ["UPLOADED_OBJECTS"])
            if len(objects) != 1 or args.wheel is None:
                raise RuntimeError("Expected exactly one uploaded wheel")
            key = next(iter(objects))
            if key.rsplit("/", 1)[-1] != args.wheel.name:
                raise RuntimeError("Uploaded wheel does not match the validated wheel")
            wheel = {"key": key, "sha256": digest(args.wheel), "size": args.wheel.stat().st_size}
            emit_wheel(cache.publish(lease, wheel))
    elif args.command == "publish":
        raise RuntimeError("Cannot publish a wheel without a lease")


if __name__ == "__main__":
    main()

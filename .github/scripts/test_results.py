"""Compact, positive recovery evidence for CI issue reconciliation."""
import base64
import json
import zlib


def encode_recovery(passed=(), resolved=(), complete=False):
    payload = json.dumps({"passed": sorted(set(passed)), "resolved": sorted(set(resolved)), "complete": complete})
    return base64.b64encode(zlib.compress(payload.encode())).decode()


def decode_recovery(value):
    if not value:
        return {"passed": [], "resolved": [], "complete": False}
    try:
        return json.loads(zlib.decompress(base64.b64decode(value, validate=True)))
    except (ValueError, zlib.error) as e:
        raise ValueError("Invalid recovery evidence") from e

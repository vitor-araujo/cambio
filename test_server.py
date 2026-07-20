"""
Server API tests — validates all endpoints return correct shapes.

  python test_server.py              # local test against 127.0.0.1:8765
  python test_server.py --port 3001  # custom port

Requires `python server.py` running in another terminal.
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def get(url: str) -> tuple[int, dict | list]:
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return resp.status, data
    except URLError as e:
        return 0, {"error": str(e.reason)}


def post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": str(e.reason)}
    except URLError as e:
        return 0, {"error": str(e.reason)}


def ok(msg: str) -> str:
    return f"  ✓ {msg}"


def fail(msg: str) -> str:
    return f"  ✗ {msg}"


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(ok(f"{name} {detail}"))
        return True
    print(fail(f"{name} — {detail}"))
    return False


def main():
    parser = argparse.ArgumentParser(description="Cambio API tests")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}/api"
    passed = 0
    total = 0

    print("\n  cambio · API tests\n")

    # health
    total += 1
    status, data = get(f"{base}/health")
    if check("GET /health", status == 200, f"status={status}"):
        if isinstance(data, dict):
            total += 1
            passed += check("  healthy", data.get("healthy"), str(data.get("healthy")))
            total += 1
            passed += check("  uptime", data.get("uptime_seconds") is not None)
            total += 1
            passed += check(
                "  db", data.get("db", {}).get("ok") is True, str(data.get("db"))
            )
            passed += 1
        else:
            passed += 0

    # dashboard
    total += 1
    status, data = get(f"{base}/dashboard")
    if check("GET /dashboard", status == 200, f"status={status}"):
        if isinstance(data, dict):
            total += 1
            passed += check(
                "  total_signals", isinstance(data.get("total_signals"), int)
            )
            total += 1
            passed += check(
                "  recent_signals", isinstance(data.get("recent_signals"), list)
            )
            total += 1
            passed += check("  thresholds", isinstance(data.get("thresholds"), dict))
            passed += 1
        else:
            passed += 0

    # journal
    total += 1
    status, data = get(f"{base}/journal")
    if check("GET /journal", status == 200, f"status={status}"):
        if isinstance(data, list):
            total += 1
            passed += check("  is array", True, f"{len(data)} entries")
            if data:
                total += 1
                passed += check("  has ts", "ts" in data[0])
                total += 1
                passed += check("  has decision", "decision" in data[0])
                passed += 1  # parent GET /journal check
            else:
                passed += 1  # parent GET /journal check
        else:
            passed += 0

    # thresholds
    total += 1
    status, data = get(f"{base}/thresholds")
    if check("GET /thresholds", status == 200, f"status={status}"):
        if isinstance(data, dict):
            required = {"watch_interval_min", "notify_threshold", "dca_floor"}
            total += 1
            passed += check("  keys ok", required <= set(data.keys()))
            passed += 1
        else:
            passed += 0

    # notifier
    total += 1
    status, data = get(f"{base}/notifier")
    if check("GET /notifier", status == 200, f"status={status}"):
        if isinstance(data, dict):
            total += 1
            passed += check("  has provider", "provider" in data)
            total += 1
            passed += check("  has is_configured", "is_configured" in data)
            passed += 1  # parent GET /notifier check
        else:
            passed += 0

    # state
    total += 1
    status, data = get(f"{base}/state")
    passed += check("GET /state", status == 200, f"status={status}")

    # POST thresholds read-back (idempotent — save current values as-is)
    total += 1
    _, current = get(f"{base}/thresholds")
    if isinstance(current, dict):
        status, data = post(f"{base}/thresholds", current)
        passed += check(
            "POST /thresholds", status == 200 and data.get("ok"), f"status={status}"
        )
    else:
        print(fail("POST /thresholds — could not read current"))
        total -= 1

    # Invalid controls are rejected instead of silently corrupting policy.
    total += 1
    status, data = post(f"{base}/thresholds", {"cadence_days": 0})
    passed += check(
        "POST /thresholds validation",
        status == 400 and bool(data.get("error")),
        f"status={status}",
    )

    # Execution lifecycle is reversible.
    total += 1
    status, data = post(f"{base}/executions", {})
    marked = status == 200 and data.get("execution", {}).get("executed") is True
    passed += check("POST /executions", marked, f"status={status}")

    total += 1
    status, data = post(f"{base}/executions/undo", {})
    undone = status == 200 and data.get("execution", {}).get("executed") is False
    passed += check("POST /executions/undo", undone, f"status={status}")

    print(f"\n  {passed}/{total} passed\n")

    if passed == total:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()

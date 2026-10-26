"""Live semantic test client.

Starts a test server (guard ON or OFF, fresh DB), seeds 5 canonical
questions, then asks 6 paraphrases (expect HIT) and 6 near-misses
(expect MISS). Reports a table plus a guard ON vs OFF similarity column.

Run:
  python live_semantic_test.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"

SEEDS = [
    "Who wrote Hamlet?",
    "What is the capital of France?",
    "How tall is Mount Everest?",
    "Who founded Apple?",
    "What is the boiling point of water in Celsius?",
]
PARAPHRASES = [
    "Who is the author of Hamlet?",
    "Capital city of France?",
    "What is the height of Mount Everest?",
    "Which person started Apple?",
    "At what temperature in degrees Celsius does water boil?",
    "Can you tell me who wrote the play Hamlet?",
]
NEAR_MISSES = [
    "Who wrote Macbeth?",
    "What is the capital of Germany?",
    "How tall is K2?",
    "Who founded Microsoft?",
    "What is the boiling point of water in Fahrenheit?",
    "Who founded Apple in 2020?",
]


def ask(prompt: str) -> dict:
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{BASE}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def start_server(guard: bool, db_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["TEST_GUARD"] = "true" if guard else "false"
    env["TEST_DB_PATH"] = db_path
    env["TEST_PORT"] = str(PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "test_server:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for readiness
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return proc
        except Exception:
            time.sleep(0.3)
    proc.terminate()
    out = proc.stdout.read() if proc.stdout else ""
    raise RuntimeError(f"Server did not start:\n{out}")


def run(guard: bool, db_path: str) -> list[dict]:
    proc = start_server(guard, db_path)
    try:
        rows: list[dict] = []
        for q in SEEDS:
            r = ask(q)
            rows.append({"question": q, "expected": "SEED", **r})
        for q in PARAPHRASES:
            r = ask(q)
            rows.append({"question": q, "expected": "HIT", **r})
        for q in NEAR_MISSES:
            r = ask(q)
            rows.append({"question": q, "expected": "MISS", **r})
        return rows
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def print_table(rows: list[dict], label: str) -> None:
    print(f"\n=== {label} ===\n")
    print(f"{'question':<58s} {'expected':>7s} {'hit':>5s} {'sim':>7s}  response")
    print("-" * 135)
    for r in rows:
        actual = "HIT" if r["cache_hit"] else "MISS"
        print(
            f"{r['question'][:57]:<58s} {r['expected']:>7s} {actual:>5s} "
            f"{r['similarity']:7.4f}  {r['response'][:55]}"
        )


def main() -> None:
    on_rows = run(guard=True, db_path=":memory:")
    off_rows = run(guard=False, db_path=":memory:")

    print_table(on_rows, "GUARD ON")
    print_table(off_rows, "GUARD OFF")

    print("\n=== ANALYSIS (guard ON) ===\n")
    problems = [r for r in on_rows if r["expected"] != ("HIT" if r["cache_hit"] else "MISS")]
    if not problems:
        print("No mismatches: all paraphrases HIT, all near-misses MISS.")
    else:
        for r in problems:
            actual = "HIT" if r["cache_hit"] else "MISS"
            print(
                f"  MISMATCH: {r['question']!r} expected={r['expected']} "
                f"actual={actual} sim={r['similarity']:.4f}"
            )

    print("\n=== SIMILARITY COMPARISON (guard ON vs OFF) ===\n")
    print(f"{'question':<58s} {'guard_on':>10s} {'guard_off':>10s}")
    print("-" * 82)
    for on, off in zip(on_rows, off_rows):
        assert on["question"] == off["question"]
        print(
            f"{on['question'][:57]:<58s} {on['similarity']:10.4f} "
            f"{off['similarity']:10.4f}"
        )


if __name__ == "__main__":
    main()
import json
import urllib.request

BASE = "http://127.0.0.1:8765/ask"

questions = [
    "Who wrote Hamlet?",
    "What is the boiling point of water in Celsius?",
]


def ask(prompt: str) -> dict:
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(
        BASE,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


for q in questions:
    print(f"=== {q} ===")
    for i in range(2):
        r = ask(q)
        print(
            f"  call {i + 1}: cache_hit={r['cache_hit']} "
            f"similarity={r['similarity']} response={r['response']!r}"
        )
    print()
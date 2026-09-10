import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

# Demo 1
r1 = client.post("/api/english-to-sign", json={"text": "Hello, can you help me?"})
print("=== DEMO 1: 'Hello, can you help me?' ===")
print("HTTP Status:", r1.status_code)
print(json.dumps(r1.json(), indent=2))

# Demo 2
r2 = client.post("/api/english-to-sign", json={"text": "Please take your medicine after lunch."})
print("\n=== DEMO 2: 'Please take your medicine after lunch.' ===")
print("HTTP Status:", r2.status_code)
print(json.dumps(r2.json(), indent=2))

# Demo 3: Synonym and lemma test
r3 = client.post("/api/english-to-sign", json={"text": "The physician is helping me today."})
print("\n=== DEMO 3: 'The physician is helping me today.' ===")
print("HTTP Status:", r3.status_code)
print(json.dumps(r3.json(), indent=2))

# Demo 4: Negation test
r4 = client.post("/api/english-to-sign", json={"text": "Do not eat food."})
print("\n=== DEMO 4: 'Do not eat food.' ===")
print("HTTP Status:", r4.status_code)
print(json.dumps(r4.json(), indent=2))

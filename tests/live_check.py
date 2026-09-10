import urllib.request
import json
import asyncio
import websockets

print("1. Testing GET /health")
with urllib.request.urlopen("http://127.0.0.1:8000/health") as r:
    health = json.loads(r.read())
    print("   Response:", health)

print("\n2. Testing GET /api/signs")
with urllib.request.urlopen("http://127.0.0.1:8000/api/signs") as r:
    signs = json.loads(r.read())
    print(f"   Registered signs count: {signs['count']}")

print("\n3. Testing POST /api/english-to-sign (Demo 1: Hello, can you help me?)")
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/english-to-sign",
    data=json.dumps({"text": "Hello, can you help me?"}).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as r:
    d1 = json.loads(r.read())
    print("   Status:", d1["translation_status"])
    print("   Signs:", [s["gloss"] for s in d1["isl"]["signs"]])
    print("   Assets:", [a["gloss"] for a in d1["realization"]["assets"]])
    print("   Missing:", d1["missing_concepts"])

print("\n4. Testing POST /api/english-to-sign (Demo 2: Please take your medicine after lunch.)")
req2 = urllib.request.Request(
    "http://127.0.0.1:8000/api/english-to-sign",
    data=json.dumps({"text": "Please take your medicine after lunch."}).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req2) as r:
    d2 = json.loads(r.read())
    print("   Status:", d2["translation_status"])
    print("   Signs:", [s["gloss"] for s in d2["isl"]["signs"]])
    print("   Assets:", [a["gloss"] for a in d2["realization"]["assets"]])
    print("   Missing:", d2["missing_concepts"])
    print("   Fallback:", d2["fallback_text"])

print("\n5. Testing video streaming endpoint /media/isl/clips/hello.mp4")
with urllib.request.urlopen("http://127.0.0.1:8000/media/isl/clips/hello.mp4") as r:
    headers = dict(r.getheaders())
    print("   HTTP Status:", r.status, "Content-Length:", headers.get("Content-Length"), "Content-Type:", headers.get("Content-Type"))

print("\n6. Testing Frontend root /")
with urllib.request.urlopen("http://127.0.0.1:8000/") as r:
    html = r.read().decode("utf-8")
    print("   HTTP Status:", r.status, "HTML length:", len(html), "Contains Title:", "<title>Vebra</title>" in html)


async def test_ws():
    print("\n7. Testing WebSocket /ws/session")
    async with websockets.connect("ws://127.0.0.1:8000/ws/session?scenario=hospital") as ws:
        hello = json.loads(await ws.recv())
        print("   Received WS hello: model =", hello["model"], "clips =", len(hello["clips"]))

        await ws.send(json.dumps({"type": "text_in", "text": "Hello, can you help me?", "lang": "en"}))
        seq = json.loads(await ws.recv())
        print("   Received WS sign_sequence: status =", seq.get("translation_status"), "available =", seq.get("available_signs"))
        print("   Items with video clips:", [(it["gloss"], it["clip"]) for it in seq.get("items", [])])


asyncio.run(test_ws())
print("\nALL LIVE INTEGRATION CHECKS PASSED!")

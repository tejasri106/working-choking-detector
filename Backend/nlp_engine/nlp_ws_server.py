# nlp_ws_server.py
import asyncio, json, websockets
from gemini_pipeline import start_emergency_guidance

async def cv_listener(websocket):
    print("🌐 Waiting for CV input...")
    async for message in websocket:
        event = json.loads(message)
        evt_type = event.get("type", "")
        conf = event.get("confidence", 0)
        print(f"📥 Received: {evt_type} (conf={conf})")

        if evt_type == "choking":
            result = start_emergency_guidance(evt_type)
            print(f"🤖 Gemini: {result['response']}")

async def main():
    ws_server = await websockets.serve(cv_listener, "localhost", 8765)
    print("🤖 Gemini WebSocket server running at ws://localhost:8765")
    await asyncio.Future()  # keep it alive

if __name__ == "__main__":
    asyncio.run(main())

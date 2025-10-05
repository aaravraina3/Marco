#!/usr/bin/env python3
"""
Simple test client for CDP WebSocket streaming
Tests the WebSocket connection and displays frame info
"""

import asyncio
import websockets
import json
import base64
from datetime import datetime

async def test_cdp_stream():
    uri = "ws://localhost:8000/ws"
    
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to CDP WebSocket!")
            
            frame_count = 0
            start_time = datetime.now()
            
            # Send a test navigation command
            await websocket.send(json.dumps({
                "type": "goto",
                "url": "https://www.google.com"
            }))
            print("📍 Sent navigation command")
            
            # Listen for frames
            while frame_count < 10:  # Receive 10 frames then exit
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "frame":
                    frame_count += 1
                    # Frame data is base64 encoded JPEG
                    frame_size = len(data["data"]) * 3 / 4  # Approximate decoded size
                    print(f"📹 Frame {frame_count}: ~{frame_size/1024:.1f} KB")
                elif data["type"] == "status":
                    print(f"📊 Status: {data}")
                else:
                    print(f"📨 Message: {data.get('type', 'unknown')}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n✅ Received {frame_count} frames in {elapsed:.1f}s (~{fps:.1f} FPS)")
            
            # Test interaction
            print("\n🖱️ Testing click interaction...")
            await websocket.send(json.dumps({
                "type": "click",
                "x": 640,
                "y": 360
            }))
            print("✅ Click sent to center of screen")
            
            # Wait for a few more frames
            for _ in range(3):
                message = await websocket.recv()
                data = json.loads(message)
                if data["type"] == "frame":
                    print("📹 Received frame after click")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 CDP WebSocket Test Client")
    print("=" * 40)
    asyncio.run(test_cdp_stream())

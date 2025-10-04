#!/usr/bin/env python3
"""
Quick test to verify the fixes for the FastAPI server
"""

import requests
import time

BASE_URL = "http://localhost:8000"

def test_fixes():
    print("🔍 Quick Test for Fixes")
    print("=" * 50)
    
    # Test 1: Check server health
    try:
        resp = requests.get(f"{BASE_URL}/")
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Server is running")
            print(f"  - Stagehand Ready: {data.get('stagehand_ready')}")
            print(f"  - Gemini Ready: {data.get('gemini_configured')}")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure server is running: python src/fastapi_stagehand_server.py")
        return
    
    # Test 2: Navigate to a simple page
    print("\n📍 Test Navigation...")
    try:
        resp = requests.post(f"{BASE_URL}/navigate", params={"url": "https://www.example.com"})
        if resp.status_code == 200:
            print("✅ Navigation successful")
        else:
            print(f"❌ Navigation failed: {resp.text}")
    except Exception as e:
        print(f"❌ Navigation error: {e}")
    
    time.sleep(1)
    
    # Test 3: Analyze screenshot
    print("\n📸 Test Gemini Analysis...")
    try:
        resp = requests.post(
            f"{BASE_URL}/analyze",
            json={"query": "What is the main text on this page?"}
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                print("✅ Gemini analysis works!")
                print(f"  - Response: {data.get('analysis')[:100]}...")
            else:
                print(f"❌ Analysis failed: {data.get('error')}")
        else:
            print(f"❌ Analysis request failed: {resp.text}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")
    
    time.sleep(1)
    
    # Test 4: Execute with Computer Use
    print("\n🤖 Test Computer Use Agent...")
    try:
        resp = requests.post(
            f"{BASE_URL}/execute",
            json={"instruction": "Click on the 'More information...' link"}
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                print("✅ Computer Use agent works!")
            else:
                print(f"❌ Execution failed: {data.get('error')}")
        else:
            print(f"❌ Execution request failed: {resp.text}")
    except Exception as e:
        print(f"❌ Execution error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick test complete!")

if __name__ == "__main__":
    test_fixes()

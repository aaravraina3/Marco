#!/usr/bin/env python3
"""
Test script for the FastAPI Stagehand server
Demonstrates how to use the API endpoints
"""

import requests
import base64
import json
import time
from typing import Optional

# API base URL
BASE_URL = "http://localhost:8000"

def check_health():
    """Check if the server is running and ready"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Server Status:")
            print(f"  - Service: {data.get('service')}")
            print(f"  - Browser Connected: {data.get('browser_connected')}")
            print(f"  - Stagehand Ready: {data.get('stagehand_ready')}")
            print(f"  - Gemini Configured: {data.get('gemini_configured')}")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Start it with: python src/fastapi_stagehand_server.py")
        return False
    except Exception as e:
        print(f"❌ Error checking server health: {e}")
        return False

def test_navigate(url: str):
    """Test navigation to a URL"""
    print(f"\n📍 Testing navigation to: {url}")
    try:
        response = requests.post(f"{BASE_URL}/navigate", params={"url": url})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Navigated successfully!")
            print(f"  - URL: {data.get('url')}")
            print(f"  - Title: {data.get('title')}")
            return True
        else:
            print(f"❌ Navigation failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error navigating: {e}")
        return False

def test_analyze(query: str, url: Optional[str] = None):
    """Test the analyze endpoint"""
    print(f"\n🔍 Testing analysis with query: {query}")
    
    payload = {"query": query}
    if url:
        payload["url"] = url
        print(f"  - URL: {url}")
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Analysis successful!")
                print(f"  - Analysis: {data.get('analysis')[:200]}...")  # First 200 chars
                
                # Save screenshot if you want
                if data.get("screenshot_base64"):
                    with open("test_screenshot.png", "wb") as f:
                        f.write(base64.b64decode(data["screenshot_base64"]))
                    print("  - Screenshot saved as: test_screenshot.png")
                return True
            else:
                print(f"❌ Analysis failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def test_execute(instruction: str):
    """Test the execute endpoint"""
    print(f"\n🤖 Testing execution with instruction: {instruction}")
    
    payload = {"instruction": instruction}
    
    try:
        response = requests.post(f"{BASE_URL}/execute", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Execution successful!")
                print(f"  - Result: {data.get('result')}")
                return True
            else:
                print(f"❌ Execution failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return False

def test_status():
    """Get the current browser status"""
    print("\n📊 Getting browser status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Browser Status:")
            print(f"  - Status: {data.get('status')}")
            print(f"  - Current URL: {data.get('current_url')}")
            print(f"  - Page Title: {data.get('page_title')}")
            print(f"  - Viewport: {data.get('viewport')}")
            return True
        else:
            print(f"❌ Failed to get status: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FastAPI Stagehand Server")
    print("=" * 50)
    
    # Check if server is running
    if not check_health():
        return
    
    # Give server a moment to fully initialize
    time.sleep(2)
    
    # Test navigation
    test_navigate("https://www.example.com")
    time.sleep(2)
    
    # Test status
    test_status()
    
    # Test analysis (requires GOOGLE_API_KEY)
    test_analyze(
        query="What is the main heading on this page?",
        url="https://www.wikipedia.org"
    )
    time.sleep(2)
    
    # Test execute (simple navigation command)
    test_execute("Go to google.com and search for 'healthcare automation'")
    time.sleep(3)
    
    # Final status check
    test_status()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nNote: Check the browser window to see the actions performed.")

if __name__ == "__main__":
    main()

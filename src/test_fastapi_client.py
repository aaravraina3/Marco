#!/usr/bin/env python3
"""
Test client for the FastAPI Stagehand server
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
            print(f"  - Agent Ready: {data.get('agent_ready')}")
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
    """Test the execute endpoint with Computer Use"""
    print(f"\n🤖 Testing Computer Use execution with instruction: {instruction}")
    
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

def interactive_mode():
    """Interactive mode to test the API"""
    print("\n🎮 Interactive Mode - Computer Use Agent")
    print("=" * 50)
    print("Commands:")
    print("  1. Navigate to URL")
    print("  2. Analyze current page (Gemini)")
    print("  3. Execute browser action (Claude Computer Use)")
    print("  4. Get browser status")
    print("  5. Run all tests")
    print("  0. Exit")
    print("=" * 50)
    
    while True:
        choice = input("\nSelect option (0-5): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            url = input("Enter URL: ").strip()
            test_navigate(url)
        elif choice == "2":
            query = input("Enter analysis query: ").strip()
            url = input("Enter URL (optional, press Enter to skip): ").strip()
            test_analyze(query, url if url else None)
        elif choice == "3":
            instruction = input("Enter instruction for the Computer Use agent: ").strip()
            test_execute(instruction)
        elif choice == "4":
            test_status()
        elif choice == "5":
            run_all_tests()
        else:
            print("❌ Invalid option. Please select 0-5")

def run_all_tests():
    """Run all automated tests"""
    print("\n🧪 Running All Tests")
    print("=" * 50)
    
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
    
    # Test execute with Computer Use (simple navigation command)
    test_execute("Go to google.com and search for 'healthcare automation'")
    time.sleep(3)
    
    # Final status check
    test_status()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nNote: Check the browser window to see the actions performed.")

def main():
    """Main function"""
    import sys
    
    print("🧪 FastAPI Stagehand Server Test Client")
    print("=" * 50)
    
    # Check if server is running
    if not check_health():
        return
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            run_all_tests()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            print("Usage:")
            print("  python src/test_fastapi_client.py           # Run all tests")
            print("  python src/test_fastapi_client.py --all     # Run all tests")
            print("  python src/test_fastapi_client.py --interactive  # Interactive mode")
    else:
        # Default: run all tests
        run_all_tests()

if __name__ == "__main__":
    main()

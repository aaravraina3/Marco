#!/usr/bin/env python3
"""
Simple test for the voice-controlled agent
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_agent_tools():
    """Test the agent tools independently"""
    from voice_controlled_agent import execute_browser_action, analyze_current_page
    
    print("🧪 Testing Agent Tools")
    print("=" * 50)
    
    # Test execute_browser_action
    print("\n1️⃣ Testing execute_browser_action...")
    result = await execute_browser_action("Navigate to google.com")
    print(f"   Result: {result}")
    
    # Test analyze_current_page
    print("\n2️⃣ Testing analyze_current_page...")
    result = await analyze_current_page("What is the main heading on this page?")
    print(f"   Result: {result}")
    
    print("\n✅ Tools test complete!")

async def test_agent_processing():
    """Test the agent's text processing"""
    from voice_controlled_agent import VoiceControlledAgent
    
    print("\n🧪 Testing Agent Text Processing")
    print("=" * 50)
    
    try:
        voice_agent = VoiceControlledAgent()
        
        test_queries = [
            "What's on this page?",
            "Search for Python tutorials",
            "hmm",  # Should be ignored as noise
            "Navigate to news website"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing: '{query}'")
            result = await voice_agent.process_with_agent(query)
            print(f"   Response: {result}")
        
        print("\n✅ Agent processing test complete!")
    except Exception as e:
        print(f"❌ Error in agent processing test: {e}")

async def main():
    """Run all tests"""
    print("🎯 Voice-Controlled Agent Test Suite")
    print("=" * 70)
    
    # Check environment variables
    missing_vars = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_vars.append("ANTHROPIC_API_KEY")
    if not os.getenv("ELEVEN_LABS_API_KEY"):
        missing_vars.append("ELEVEN_LABS_API_KEY")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return
    
    print("✅ Environment variables OK")
    
    # Check server connection
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/", timeout=aiohttp.ClientTimeout(total=2)) as response:
                data = await response.json()
                print(f"✅ Server connected: {data.get('service')}")
    except:
        print("❌ FastAPI server is not running!")
        print("Please start it with: python src/fastapi_stagehand_server.py")
        return
    
    # Run tests
    await test_agent_tools()
    await test_agent_processing()
    
    print("\n" + "=" * 70)
    print("🎉 All tests complete!")
    print("\n💡 To run the full voice agent:")
    print("   python src/voice_controlled_agent.py")

if __name__ == "__main__":
    asyncio.run(main())

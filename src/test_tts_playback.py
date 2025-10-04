#!/usr/bin/env python3
"""
Simple test to verify TTS audio generation and playback
"""

import asyncio
import os
import sys
from pathlib import Path

# Add ffmpeg to PATH if it exists at the known location
FFMPEG_PATH = r"C:\Program Files\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ['PATH']:
    os.environ['PATH'] = FFMPEG_PATH + os.pathsep + os.environ['PATH']
    print(f"✅ Added ffmpeg to PATH: {FFMPEG_PATH}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_tts_playback():
    """Test TTS generation and playback"""
    # Import after setting up path
    from voice_controlled_agent import VoiceControlledAgent
    
    print("=" * 70)
    print("Testing TTS Audio Playback")
    print("=" * 70)
    print()
    
    # Create agent with TTS enabled
    agent = VoiceControlledAgent(enable_tts=True)
    
    # Test messages
    test_messages = [
        "Hello, this is a test of the text to speech system.",
        "The audio should play through your speakers now.",
        "If you hear this message, the TTS is working correctly!"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Test {i}/{len(test_messages)}]")
        print(f"📝 Text: {message}")
        print("-" * 50)
        
        # Generate and play audio
        await agent.text_to_speech(message)
        
        # Wait a bit between messages
        if i < len(test_messages):
            print("\n⏳ Waiting 2 seconds before next test...")
            await asyncio.sleep(2)
    
    print("\n" + "=" * 70)
    print("✅ TTS Playback Test Complete!")
    print("=" * 70)
    
    # Cleanup
    agent.cleanup()

if __name__ == "__main__":
    print("Starting TTS Playback Test...")
    print("You should hear 3 test messages through your speakers.")
    print()
    
    try:
        asyncio.run(test_tts_playback())
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

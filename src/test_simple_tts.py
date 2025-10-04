#!/usr/bin/env python3
"""
Minimal test for Eleven Labs TTS and Windows audio playback
"""

import os
import sys
import tempfile
import time

# Add ffmpeg to PATH if it exists at the known location
FFMPEG_PATH = r"C:\Program Files\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ['PATH']:
    os.environ['PATH'] = FFMPEG_PATH + os.pathsep + os.environ['PATH']
    print(f"✅ Added ffmpeg to PATH: {FFMPEG_PATH}")

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.play import play

# Load environment variables
load_dotenv()

def test_simple_tts():
    """Simple TTS test without async complexity"""
    
    api_key = os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        print("❌ ELEVEN_LABS_API_KEY not found!")
        return
    
    print("🔧 Initializing Eleven Labs client...")
    client = ElevenLabs(api_key=api_key)
    
    text = "Hello! This is a test of the Eleven Labs text to speech system. Can you hear me?"
    print(f"📝 Text: {text}")
    
    try:
        print("🔊 Generating speech...")
        
        # Generate speech
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # George voice
            model_id="eleven_turbo_v2_5",
            output_format="mp3_22050_32",
        )
        
        # Convert to bytes
        audio_bytes = b"".join(audio_generator)
        print(f"✅ Generated {len(audio_bytes)} bytes of audio")
        
        # Play using ElevenLabs' built-in play function
        print("🎵 Playing audio with ElevenLabs play() function...")
        try:
            play(audio_bytes)
            print("✅ Audio playback complete!")
        except Exception as play_error:
            print(f"⚠️ Could not play audio: {play_error}")
            print("   This might mean ffmpeg or mpv needs to be installed.")
            
            # Save as fallback
            fallback_path = "test_audio.mp3"
            with open(fallback_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"💾 Audio saved to {fallback_path} - you can play it manually")
        
        print("\n✅ Test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 70)
    print("Simple TTS Playback Test")
    print("=" * 70)
    print()
    test_simple_tts()

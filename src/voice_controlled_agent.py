#!/usr/bin/env python3
"""
Voice-Controlled Web Agent using LangChain and Speech Transcription
Combines VAD speech transcription with LangChain agent for web automation
"""

import os
import sys
import io
import wave
import pyaudio
import time
import torch
import numpy as np
import asyncio
import aiohttp
import json
from collections import deque
from dotenv import load_dotenv

# Add ffmpeg to PATH if it exists at the known location
FFMPEG_PATH = r"C:\Program Files\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ['PATH']:
    os.environ['PATH'] = FFMPEG_PATH + os.pathsep + os.environ['PATH']
    print(f"✅ Added ffmpeg to PATH: {FFMPEG_PATH}")
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from typing import Optional, Any, Type
from pydantic import BaseModel, Field

# LangChain v1-alpha imports
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Audio configuration (same as transcribe_speech.py)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
VAD_WINDOW_SIZE = 512
ACCUMULATE_CHUNKS = 2
SPEECH_THRESHOLD = 0.5
SILENCE_DURATION_THRESHOLD = 1.0
MIN_SPEECH_DURATION = 0.5

# FastAPI server URL
API_BASE_URL = "http://localhost:8000"

# ===== TOOL DEFINITIONS =====

@tool
async def execute_browser_action(instruction: str) -> str:
    """
    Execute a browser action for general web queries and navigation.
    Use this when you need to search for something on the web or navigate to websites.
    IMPORTANT: Always navigate to google.com first before searching.
    
    Args:
        instruction: The action to perform in the browser
        
    Returns:
        Result of the browser action
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE_URL}/execute",
                json={"instruction": instruction},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                data = await response.json()
                if data.get("success"):
                    return f"✅ Successfully executed: {instruction}"
                else:
                    return f"❌ Failed to execute: {data.get('error', 'Unknown error')}"
        except asyncio.TimeoutError:
            return "❌ Execution timed out (60 seconds)"
        except Exception as e:
            return f"❌ Error executing browser action: {str(e)}"

@tool
async def analyze_current_page(query: str) -> str:
    """
    Analyze the current page visible in the browser.
    Use this when the user asks questions about what's currently on the screen.
    
    Args:
        query: The question to answer about the current page
        
    Returns:
        Analysis of the current page based on the query
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE_URL}/analyze",
                json={"query": query},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                data = await response.json()
                if data.get("success"):
                    return data.get("analysis", "No analysis available")
                else:
                    return f"❌ Failed to analyze page: {data.get('error', 'Unknown error')}"
        except asyncio.TimeoutError:
            return "❌ Analysis timed out (30 seconds)"
        except Exception as e:
            return f"❌ Error analyzing page: {str(e)}"

# ===== LANGCHAIN AGENT SETUP =====

def create_voice_agent():
    """Create and configure the LangChain agent with Claude 4 Sonnet"""
    
    # Initialize Claude 4 Sonnet model
    model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # Claude 4 Sonnet model
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
        max_tokens=1000
    )
    
    # Define the tools
    tools = [execute_browser_action, analyze_current_page]
    
    # System prompt with clear instructions
    prompt = """You are a helpful voice-controlled web assistant. You have two main capabilities:

1. **execute_browser_action**: Use this for general web queries, searches, and navigation.
   - ALWAYS navigate to google.com first before searching
   - Use for: "Search for...", "Find...", "Look up...", "Navigate to..."
   
2. **analyze_current_page**: Use this to answer questions about what's currently visible on screen.
   - Use for: "What's on this page?", "What do you see?", "Read the...", "Tell me about what's shown"

IMPORTANT RULES:
- If the user's input seems to be silence, random noise, or not a real command (like "hmm", "uh", breathing sounds), respond with "Waiting for command..." and don't take any action.
- Be concise in your responses since this is voice-controlled.
- Always confirm what action you're taking.
- Return most of the full response of the execute tool especially if the sub agent in the execute tool seems to be asking for something.

"""
    
    # Create the agent using the v1-alpha format
    agent = create_agent(
        model=model,
        tools=tools,
        prompt=prompt
    )
    
    return agent

# ===== SPEECH TRANSCRIPTION WITH VAD =====

class VoiceControlledAgent:
    def __init__(self, enable_tts: bool = True):
        """Initialize the voice-controlled agent
        
        Args:
            enable_tts: Whether to enable text-to-speech for agent responses
        """
        # Initialize Eleven Labs for transcription and TTS
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        if not self.eleven_labs_api_key:
            raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")
        
        self.client = ElevenLabs(api_key=self.eleven_labs_api_key)
        self.enable_tts = enable_tts
        self.is_running = True
        self.audio = pyaudio.PyAudio()
        self.is_playing_audio = False  # Track if TTS audio is currently playing
        
        # Initialize the LangChain agent
        print("🤖 Initializing LangChain Agent with Claude 4 Sonnet...")
        self.agent = create_voice_agent()
        print("✅ Agent initialized successfully!")
        
        # Chat history for context
        self.chat_history = []
        
        # Initialize Silero VAD
        print("🔧 Loading Silero VAD model...")
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
            print("✅ Silero VAD model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading VAD model: {e}")
            raise
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(RATE * 10))
        self.vad_accumulator = []
        self.speech_frames = []
        self.is_speech_active = False
        self.silence_start_time = None
    
    def detect_speech_in_frame(self, audio_frame):
        """Use Silero VAD to detect speech in audio frame"""
        try:
            if len(audio_frame) != VAD_WINDOW_SIZE:
                if len(audio_frame) < VAD_WINDOW_SIZE:
                    padded = np.zeros(VAD_WINDOW_SIZE, dtype=np.int16)
                    padded[:len(audio_frame)] = audio_frame
                    audio_frame = padded
                else:
                    audio_frame = audio_frame[:VAD_WINDOW_SIZE]
            
            audio_tensor = torch.from_numpy(audio_frame).float()
            
            if audio_tensor.abs().max() > 0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            confidence = self.vad_model(audio_tensor, RATE).item()
            return confidence > SPEECH_THRESHOLD
        except Exception as e:
            print(f"⚠️  VAD detection error: {e}")
            return False
    
    def record_with_vad(self):
        """Record audio using VAD to detect speech boundaries"""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("🎤 Voice-controlled agent listening... (say 'exit' or 'quit' to stop)")
        print("💡 Try saying: 'Search for weather', 'What's on this page?', 'Navigate to news'")
        
        try:
            while self.is_running:
                # Wait if audio is currently playing to avoid recording our own TTS output
                if self.is_playing_audio:
                    time.sleep(0.1)
                    # Clear any accumulated audio and buffers while we were playing
                    try:
                        stream.read(CHUNK, exception_on_overflow=False)
                    except:
                        pass
                    # Clear VAD buffers to start fresh after TTS playback
                    self.audio_buffer = []
                    self.vad_accumulator = []
                    self.speech_frames = []
                    self.is_speech_active = False
                    self.silence_start_time = None
                    continue
                
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                self.audio_buffer.extend(audio_chunk)
                self.vad_accumulator.extend(audio_chunk)
                
                if len(self.vad_accumulator) >= VAD_WINDOW_SIZE * ACCUMULATE_CHUNKS:
                    vad_frame = np.array(self.vad_accumulator[-VAD_WINDOW_SIZE:])
                    is_speech = self.detect_speech_in_frame(vad_frame)
                    
                    current_time = time.time()
                    
                    if is_speech:
                        if not self.is_speech_active:
                            print("🗣️  Listening...")
                            self.is_speech_active = True
                            self.speech_frames = []
                        
                        self.speech_frames.extend(self.vad_accumulator)
                        self.silence_start_time = None
                        
                    else:
                        if self.is_speech_active:
                            if self.silence_start_time is None:
                                self.silence_start_time = current_time
                            
                            self.speech_frames.extend(self.vad_accumulator)
                            
                            silence_duration = current_time - self.silence_start_time
                            if silence_duration >= SILENCE_DURATION_THRESHOLD:
                                print("🔇 Processing speech...")
                                
                                if len(self.speech_frames) > 0:
                                    speech_duration = len(self.speech_frames) / RATE
                                    if speech_duration >= MIN_SPEECH_DURATION:
                                        yield self.create_audio_buffer(self.speech_frames)
                                    else:
                                        print(f"⏭️  Too short ({speech_duration:.1f}s)")
                                
                                self.is_speech_active = False
                                self.speech_frames = []
                                self.silence_start_time = None
                    
                    self.vad_accumulator = []
                
                time.sleep(0.01)
                
        finally:
            stream.stop_stream()
            stream.close()
    
    def create_audio_buffer(self, frames):
        """Create WAV audio buffer from frames"""
        wav_buffer = io.BytesIO()
        wf = wave.open(wav_buffer, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(np.array(frames, dtype=np.int16).tobytes())
        wf.close()
        wav_buffer.seek(0)
        return wav_buffer
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Eleven Labs API"""
        try:
            print("🔄 Transcribing...")
            transcription = self.client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=False
            )
            return transcription
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            return None
    
    async def text_to_speech(self, text: str):
        """Convert text to speech using Eleven Labs TTS and play it"""
        try:
            print("🔊 Generating speech...")
            self.is_playing_audio = True  # Set flag to prevent recording
            
            # Generate speech audio from text
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",  # Using a default voice (George)
                model_id="eleven_turbo_v2_5",  # Fast model for low latency
                output_format="mp3_44100_128",  # Higher quality for better playback
            )
            
            # Convert generator to bytes for duration calculation
            audio_bytes = b"".join(audio_generator)
            
            # Estimate duration for waiting
            words = len(text.split())
            estimated_duration = max((words / 150) * 60, 2) + 1  # At least 2 seconds + 1 buffer
            
            try:
                print(f"🎵 Playing audio ({len(audio_bytes)} bytes)...")
                
                # Use ElevenLabs' built-in play function
                # This handles all OS-specific playback automatically
                play(audio_bytes)
                
                # The play function is synchronous, so when it returns, playback is done
                print("✅ Audio playback complete, ready for next command")
                
            except Exception as play_error:
                print(f"⚠️ Could not play audio: {play_error}")
                
                # Save to file as fallback
                fallback_path = "agent_response.mp3"
                with open(fallback_path, 'wb') as f:
                    f.write(audio_bytes)
                print(f"💾 Audio saved to {fallback_path} - you can play it manually")
                
                # Wait the estimated duration even on error
                print(f"⏳ Waiting {estimated_duration:.1f} seconds...")
                await asyncio.sleep(estimated_duration)
                
        except Exception as e:
            print(f"⚠️ TTS failed (continuing without audio): {e}")
            # Wait a bit to avoid immediate recording
            await asyncio.sleep(2)
        finally:
            # Always clear the flag when done
            self.is_playing_audio = False
    
    async def process_with_agent(self, text: str):
        """Process transcribed text with the LangChain agent
        
        Returns:
            tuple: (response_text, should_speak) where should_speak indicates if TTS should be used
        """
        try:
            print(f"🤖 Agent processing: '{text}'")
            
            # Check for exit commands
            if any(word in text.lower() for word in ['exit', 'quit', 'stop']):
                print("👋 Exit command detected")
                self.is_running = False
                return "Goodbye! Stopping voice control.", True  # Always speak goodbye
            
            # Build messages list with context
            messages = []
            
            # Add recent chat history for context (if any)
            if self.chat_history:
                messages.extend(self.chat_history[-10:])  # Keep last 10 messages
            
            # Add current user message
            messages.append({"role": "user", "content": text})
            
            # Invoke the agent using v1-alpha format
            result = await self.agent.ainvoke({
                "messages": messages
            })
            
            # Check if tools were called by examining the message history
            tools_called = False
            if result and "messages" in result:
                # Check all messages in the result for tool calls
                for message in result["messages"]:
                    # Check for tool call indicators
                    if hasattr(message, 'additional_kwargs'):
                        if message.additional_kwargs.get('tool_calls'):
                            tools_called = True
                            break
                    # Also check message type
                    if message.__class__.__name__ in ['ToolMessage', 'FunctionMessage']:
                        tools_called = True
                        break
                    # Check content for tool execution patterns
                    if hasattr(message, 'content') and isinstance(message.content, str):
                        content = message.content.lower()
                        if any(indicator in content for indicator in [
                            'executing', 'navigating', 'clicking', 'typing',
                            'screenshot', 'analyzed', 'completed', 'performed'
                        ]):
                            tools_called = True
                            break
            
            # Extract the response from the last message
            if result and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    output = last_message.content
                else:
                    output = str(last_message)
            else:
                output = "No response"
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": text})
            self.chat_history.append({"role": "assistant", "content": output})
            
            # Determine if we should speak based on tools being called or substantive response
            should_speak = tools_called or len(output.split()) > 10  # Speak if tools called or long response
            
            if tools_called:
                print(f"🔧 Tools were called - will speak response")
            else:
                print(f"💬 Simple response - {'will speak' if should_speak else 'text only'}")
            
            print(f"✅ Agent response: {output}")
            return output, should_speak
            
        except Exception as e:
            print(f"❌ Error processing with agent: {e}")
            return f"Error: {str(e)}", True  # Speak errors
    
    async def run(self):
        """Main loop - record, transcribe, and process with agent"""
        print("🎯 Voice-Controlled Web Agent with Claude 4 Sonnet")
        print("=" * 70)
        
        try:
            for audio_data in self.record_with_vad():
                if not self.is_running:
                    break
                
                # Transcribe the audio
                transcription = self.transcribe_audio(audio_data)
                
                if transcription and transcription.text.strip():
                    text = transcription.text.strip()
                    print(f"📝 You said: {text}")
                    
                    # Process with agent (returns response text and whether to speak)
                    response, should_speak = await self.process_with_agent(text)
                    
                    # Speak the response using TTS only if:
                    # 1. TTS is enabled
                    # 2. We're still running
                    # 3. The agent determined speech is appropriate (tools were called or important response)
                    if response and self.enable_tts and self.is_running and should_speak:
                        await self.text_to_speech(response)
                else:
                    print("🔇 No speech detected")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.is_running = False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()
        print("✅ Cleanup complete")

# ===== MAIN FUNCTION =====

async def main():
    """Main function to run the voice-controlled agent"""
    try:
        # Check environment variables
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        if not os.getenv("ELEVEN_LABS_API_KEY"):
            raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")
        
        # Check if server is running
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{API_BASE_URL}/", timeout=aiohttp.ClientTimeout(total=2)) as response:
                    data = await response.json()
                    print(f"✅ Server connected: {data.get('service')}")
            except:
                print("❌ FastAPI server is not running!")
                print("Please start it with: python src/fastapi_stagehand_server.py")
                return
        
        # Create and run the voice-controlled agent
        # Set enable_tts=False to disable text-to-speech responses
        agent = VoiceControlledAgent(enable_tts=True)
        await agent.run()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please set required API keys in your .env file")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
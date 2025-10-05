#!/usr/bin/env python3
"""
Healthcare Intake Assistant - Speech Transcription Script with VAD
Continuously transcribes speech using Eleven Labs Speech-to-Text API with Voice Activity Detection
Uses Silero VAD to automatically detect speech start/end instead of fixed time intervals
"""

import os
import io
import wave
import pyaudio
import threading
import time
import torch
import numpy as np
from collections import deque
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

# Audio configuration
CHUNK = 1024  # Larger chunks for better audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz for optimal VAD performance
VAD_WINDOW_SIZE = 512  # EXACTLY 512 samples required for Silero VAD at 16kHz
ACCUMULATE_CHUNKS = 2  # Accumulate chunks before VAD processing

# VAD configuration
SPEECH_THRESHOLD = 0.5  # Confidence threshold for speech detection
SILENCE_DURATION_THRESHOLD = 1.0  # Seconds of silence before considering speech ended
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration to process

class VADSpeechTranscriber:
    def __init__(self):
        self.api_key = os.getenv("ELEVEN_LABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables")
        
        self.client = ElevenLabs(api_key=self.api_key)
        self.transcriptions = []
        self.is_running = True
        self.audio = pyaudio.PyAudio()
        
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
        
        # Audio buffer for VAD processing
        self.audio_buffer = deque(maxlen=int(RATE * 10))  # 10-second buffer
        self.vad_accumulator = []  # Accumulate chunks for VAD processing
        self.speech_frames = []
        self.is_speech_active = False
        self.silence_start_time = None
        
    def detect_speech_in_frame(self, audio_frame):
        """Use Silero VAD to detect speech in audio frame"""
        try:
            # Silero VAD requires EXACTLY 512 samples for 16kHz
            if len(audio_frame) != VAD_WINDOW_SIZE:
                # Pad or truncate to exact size
                if len(audio_frame) < VAD_WINDOW_SIZE:
                    # Pad with zeros
                    padded = np.zeros(VAD_WINDOW_SIZE, dtype=np.int16)
                    padded[:len(audio_frame)] = audio_frame
                    audio_frame = padded
                else:
                    # Truncate to exact size
                    audio_frame = audio_frame[:VAD_WINDOW_SIZE]
                
            # Convert to float32 and normalize
            audio_tensor = torch.from_numpy(audio_frame).float()
            
            # VAD expects audio to be normalized between -1 and 1
            if audio_tensor.abs().max() > 0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Get VAD confidence
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
        
        print("🎤 Listening with VAD... (speak naturally, pauses will be detected automatically)")
        
        try:
            while self.is_running:
                # Read audio chunk
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Add to buffer and accumulator
                self.audio_buffer.extend(audio_chunk)
                self.vad_accumulator.extend(audio_chunk)
                
                # Process VAD when we have enough accumulated data
                if len(self.vad_accumulator) >= VAD_WINDOW_SIZE * ACCUMULATE_CHUNKS:
                    # Take exactly VAD_WINDOW_SIZE samples for analysis
                    vad_frame = np.array(self.vad_accumulator[-VAD_WINDOW_SIZE:])
                    
                    # Detect speech
                    is_speech = self.detect_speech_in_frame(vad_frame)
                    
                    current_time = time.time()
                    
                    if is_speech:
                        if not self.is_speech_active:
                            print("🗣️  Speech detected - recording...")
                            self.is_speech_active = True
                            self.speech_frames = []
                        
                        # Add to speech buffer (add all accumulated data)
                        self.speech_frames.extend(self.vad_accumulator)
                        self.silence_start_time = None
                        
                    else:  # No speech detected
                        if self.is_speech_active:
                            if self.silence_start_time is None:
                                self.silence_start_time = current_time
                            
                            # Continue collecting a bit more audio during silence
                            self.speech_frames.extend(self.vad_accumulator)
                            
                            # Check if silence duration exceeded threshold
                            silence_duration = current_time - self.silence_start_time
                            if silence_duration >= SILENCE_DURATION_THRESHOLD:
                                print("🔇 Speech ended - processing...")
                                
                                # Process the collected speech
                                if len(self.speech_frames) > 0:
                                    speech_duration = len(self.speech_frames) / RATE
                                    if speech_duration >= MIN_SPEECH_DURATION:
                                        yield self.create_audio_buffer(self.speech_frames)
                                    else:
                                        print(f"⏭️  Speech too short ({speech_duration:.1f}s) - skipping")
                                
                                # Reset state
                                self.is_speech_active = False
                                self.speech_frames = []
                                self.silence_start_time = None
                    
                    # Clear accumulator after processing
                    self.vad_accumulator = []
                
                # Small delay to prevent excessive CPU usage
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
    
    def check_for_quit(self, text):
        """Check if the transcribed text contains 'quit' command"""
        if text and "quit" in text.lower():
            return True
        return False
    
    def run(self):
        """Main loop - continuously record and transcribe using VAD"""
        print("🎯 Healthcare Intake Assistant - VAD Speech Transcriber")
        print("🔊 Say 'quit' to stop recording and see all transcriptions")
        print("🤖 Using Silero VAD for automatic speech detection")
        print("=" * 70)
        
        try:
            # Start VAD-based recording
            for audio_data in self.record_with_vad():
                if not self.is_running:
                    break
                
                # Transcribe the audio
                transcription = self.transcribe_audio(audio_data)
                
                if transcription and transcription.text.strip():
                    print(f"📝 Transcribed: {transcription.text}")
                    self.transcriptions.append(transcription.text.strip())
                    
                    # Check if user said quit
                    if self.check_for_quit(transcription.text):
                        print("🛑 'Quit' detected. Stopping transcription...")
                        self.is_running = False
                        break
                else:
                    print("🔇 No speech detected or transcription failed")
                
                # Small pause between processing
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.is_running = False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()
    
    def print_all_transcriptions(self):
        """Print all collected transcriptions"""
        print("\n" + "=" * 70)
        print("📋 ALL TRANSCRIPTIONS:")
        print("=" * 70)
        
        if not self.transcriptions:
            print("No transcriptions recorded.")
        else:
            for i, transcription in enumerate(self.transcriptions, 1):
                print(f"{i:2d}. {transcription}")
        
        print("=" * 70)
        print(f"Total transcriptions: {len(self.transcriptions)}")

def main():
    """Main function"""
    try:
        transcriber = VADSpeechTranscriber()
        transcriber.run()
        transcriber.print_all_transcriptions()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please make sure ELEVEN_LABS_API_KEY is set in your .env file")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    main()
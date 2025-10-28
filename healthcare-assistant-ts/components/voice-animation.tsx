'use client';

import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Mic, MicOff, Volume2, AlertCircle, Loader2 } from 'lucide-react';
import { getVoiceRecognition } from '@/lib/speech-recognition';

interface VoiceAnimationProps {
  onTranscript?: (text: string) => void;
  isListening?: boolean;
  onListeningChange?: (listening: boolean) => void;
  isProcessing?: boolean;  // Add this to know when agent is processing
}

export function VoiceAnimation({ 
  onTranscript, 
  isListening = false,
  onListeningChange,
  isProcessing = false
}: VoiceAnimationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [interimTranscript, setInterimTranscript] = useState<string>('');
  const [audioInputs, setAudioInputs] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const processingRef = useRef(false); // Track processing state internally
  const [secureContextError, setSecureContextError] = useState<string | null>(null);

  // Keep device list fresh (hot‑plugging mics, OS changes, etc.)
  useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const refresh = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs = devices.filter(d => d.kind === 'audioinput');
        setAudioInputs(inputs);
        if (!selectedDeviceId && inputs[0]?.deviceId) setSelectedDeviceId(inputs[0].deviceId);
      } catch {}
    };
    refresh();
    const handler = () => refresh();
    navigator.mediaDevices.addEventListener?.('devicechange', handler);
    return () => navigator.mediaDevices.removeEventListener?.('devicechange', handler);
  }, [selectedDeviceId]);

  // Stop and replace current mic stream
  const replaceMicStream = (stream: MediaStream | null) => {
    if (micStreamRef.current) {
      try { micStreamRef.current.getTracks().forEach(t => { t.stop(); t.enabled = false; }); } catch {}
    }
    micStreamRef.current = stream;
  };

  // Warm-up permission + robust stream acquisition
  const acquireMicStream = async (): Promise<MediaStream> => {
    // 1) Permission + warm-up on any available device
    let warm: MediaStream | null = null;
    try {
      warm = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e: any) {
      throw e; // Surface NotAllowedError etc.
    }

    // 2) Refresh device list (labels are populated only after permission)
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter(d => d.kind === 'audioinput');
      setAudioInputs(inputs);
      if (!selectedDeviceId && inputs[0]?.deviceId) setSelectedDeviceId(inputs[0].deviceId);
    } catch {}

    // 3) Try preferred device (exact). If it fails, keep warm stream.
    if (selectedDeviceId) {
      try {
        const preferred = await navigator.mediaDevices.getUserMedia({
          audio: { deviceId: { exact: selectedDeviceId }, echoCancellation: true, noiseSuppression: true } as MediaTrackConstraints,
        });
        try { warm.getTracks().forEach(t => t.stop()); } catch {}
        return preferred;
      } catch (e: any) {
        // fall through to warm
      }
    }
    return warm;
  };

  // Initialize audio context, analyser, and speech recognition
  const setupAudio = async () => {
    // Don't setup if we're processing
    if (processingRef.current || isProcessing) {
      return;
    }
    try {
      // Must be HTTPS/localhost for mic
      if (!window.isSecureContext && location.hostname !== 'localhost') {
        setSecureContextError('Microphone requires HTTPS. Use Vercel link or https://localhost.');
        return;
      }
      // Ensure mediaDevices support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Browser does not support microphone APIs. Use Chrome.');
        return;
      }

      // Request microphone permission and setup audio visualization
      const stream = await acquireMicStream();
      replaceMicStream(stream);
      
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      // Setup speech recognition - always create a new one, don't reuse
      // Clear any existing recognition first
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch (e) {}
        recognitionRef.current = null;
      }
      
      try {
        const recognition = getVoiceRecognition({
          continuous: true,
          interimResults: true,
        });
          
          recognition.onResult((transcript, isFinal) => {
            // Don't process any results if we're processing (check both prop and ref)
            if (processingRef.current || isProcessing) {
              recognition.abort(); // Force abort
              setInterimTranscript('');
              return;
            }
            
            if (isFinal) {
              // Send final transcript to parent
              onTranscript?.(transcript);
              setInterimTranscript('');
              // Stop recognition immediately after getting final transcript
              recognition.stop();
              setIsRecording(false);
            } else {
              // Update interim transcript for display
              setInterimTranscript(transcript);
            }
          });
          
          recognition.onError((errorMsg) => {
            setError(errorMsg);
            setIsRecording(false);
            onListeningChange?.(false);
          });
          
          recognition.onEnd(() => {
            setIsRecording(false);
            // Don't restart if we're processing
            if (isListening && !processingRef.current && !isProcessing) {
              // Only restart if we should still be listening AND not processing
              recognition.start();
            } else {
              // Clear the listening state if we ended during processing
              if (processingRef.current || isProcessing) {
                onListeningChange?.(false);
              }
            }
          });
        
        recognitionRef.current = recognition;
      } catch (err) {
        console.error('Speech recognition not supported:', err);
        setError('Speech recognition not supported in this browser');
      }
      
      // Start speech recognition (ensure AudioContext is running)
      if (recognitionRef.current) {
        try {
          if (audioContextRef.current.state === 'suspended') {
            await audioContextRef.current.resume();
          }
          recognitionRef.current.start();
        } catch (e) {
          console.error('Failed to start recognition:', e);
        }
      }
      
      setIsRecording(true);
      setError(null);
      startVisualization();
    } catch (error: any) {
      console.error('Failed to setup audio:', error);
      const name = error?.name || '';
      const msg = name === 'NotAllowedError' ? 'Microphone blocked. Allow mic access in the browser.'
        : name === 'NotFoundError' ? 'No microphone found. Choose a device below or check input in Chrome settings.'
        : 'Failed to access microphone. Please check permissions.';
      setError(msg);
    }
  };

  const stopAudio = () => {
    // Stop speech recognition FIRST and clear the ref
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort(); // Force abort instead of stop
        recognitionRef.current = null; // Clear the reference
      } catch (e) {
        console.log('Recognition already stopped');
      }
    }
    
    // Stop audio stream
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => {
        track.stop();
        track.enabled = false; // Also disable the track
      });
      micStreamRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    
    setIsRecording(false);
    setInterimTranscript('');
    setError(null);
  };

  const startVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas || !analyserRef.current) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      
      analyserRef.current!.getByteFrequencyData(dataArray);
      
      // Clear canvas
      ctx.fillStyle = 'rgb(15, 23, 42)'; // Dark background
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw waveform bars
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.8;
        
        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - barHeight);
        
        if (isListening || isRecording) {
          // Active state - purple/blue gradient
          gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
          gradient.addColorStop(1, 'rgba(118, 75, 162, 0.8)');
        } else {
          // Inactive state - gray
          gradient.addColorStop(0, 'rgba(100, 116, 139, 0.5)');
          gradient.addColorStop(1, 'rgba(71, 85, 105, 0.5)');
        }
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
      
      // Check if speaking (audio level detection)
      const average = dataArray.reduce((a, b) => a + b, 0) / bufferLength;
      setIsSpeaking(average > 30); // Threshold for speaking detection
    };
    
    draw();
  };

  useEffect(() => {
    // Update internal processing ref
    processingRef.current = isProcessing;
    
    // Always stop audio if processing starts, regardless of listening state
    if (isProcessing) {
      // Force stop all audio and recognition
      stopAudio();
      if (isListening) {
        onListeningChange?.(false);
      }
      return; // Don't setup audio while processing
    }
    
    // Only setup/stop based on listening state when not processing
    if (isListening && !isProcessing) {
      setupAudio();
    } else {
      stopAudio();
    }
    
    return () => {
      stopAudio();
    };
  }, [isListening, isProcessing]);

  const handleToggleListening = () => {
    // Don't allow toggling while processing
    if (isProcessing) return;
    
    const newState = !isListening;
    // Resume AudioContext immediately on user gesture
    if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume().catch(() => {});
    }
    onListeningChange?.(newState);
  };

  return (
    <Card className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="text-lg font-semibold">Voice Control</h3>
        <Badge variant={isProcessing ? "default" : isSpeaking ? "default" : "secondary"}>
          {isProcessing ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : isSpeaking ? <Volume2 className="w-3 h-3 mr-1" /> : null}
          {isProcessing ? 'Processing' : isSpeaking ? 'Speaking' : isRecording ? 'Listening' : 'Ready'}
        </Badge>
      </div>
      
      <div className="flex-1 p-4 flex flex-col items-center justify-center">
        <canvas
          ref={canvasRef}
          width={280}
          height={150}
          className="w-full max-w-sm rounded"
        />
        
        <div className="mt-6 flex flex-col items-center gap-4">
          {/* Microphone device selector */}
          {audioInputs.length > 0 && (
            <div className="w-full max-w-sm">
              <label className="text-xs text-muted-foreground">Microphone</label>
              <select
                className="w-full mt-1 border rounded px-2 py-1 bg-background"
                value={selectedDeviceId}
                onChange={(e) => setSelectedDeviceId(e.target.value)}
                disabled={isRecording}
              >
                {audioInputs.map((d) => (
                  <option key={d.deviceId} value={d.deviceId}>{d.label || 'Microphone'}</option>
                ))}
              </select>
            </div>
          )}

          <Button
            onClick={handleToggleListening}
            variant={isListening ? "destructive" : isProcessing ? "secondary" : "default"}
            size="lg"
            className="rounded-full w-20 h-20"
            disabled={isProcessing}
          >
            {isProcessing ? (
              <Loader2 className="w-8 h-8 animate-spin" />
            ) : isListening ? (
              <MicOff className="w-8 h-8" />
            ) : (
              <Mic className="w-8 h-8" />
            )}
          </Button>
          
          <p className="text-sm text-muted-foreground text-center">
            {isProcessing
              ? 'Processing your request...'
              : isListening 
              ? 'Listening for commands...' 
              : 'Click to start voice control'}
          </p>
        </div>
        
        {/* Show interim transcript */}
        {interimTranscript && (
          <div className="mt-4 w-full p-3 bg-muted rounded-lg">
            <p className="text-sm italic">📝 {interimTranscript}</p>
          </div>
        )}
        
        {/* Show error if any */}
        {(error || secureContextError) && (
          <div className="mt-4 w-full p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-destructive" />
            <p className="text-sm text-destructive">{secureContextError || error}</p>
          </div>
        )}

        {/* Mic quick test */}
        <div className="mt-2 w-full flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={async () => {
              try {
                setError(null);
                setSecureContextError(null);
                // Force permission prompt and refresh devices
                await navigator.mediaDevices.getUserMedia({ audio: true });
                const devices = await navigator.mediaDevices.enumerateDevices();
                const inputs = devices.filter(d => d.kind === 'audioinput');
                setAudioInputs(inputs);
                if (!selectedDeviceId && inputs[0]?.deviceId) setSelectedDeviceId(inputs[0].deviceId);
              } catch (e: any) {
                const name = e?.name || '';
                const msg = name === 'NotAllowedError' ? 'Microphone blocked. Allow in the URL bar.'
                  : name === 'NotFoundError' ? 'No microphone found. Choose a device in Chrome settings.'
                  : 'Failed to access microphone.';
                setError(msg);
              }
            }}
          >
            Test Mic
          </Button>
        </div>
        
        <div className="mt-4 w-full">
          <p className="text-xs text-muted-foreground mb-2">Quick Commands:</p>
          <div className="flex flex-wrap gap-1">
            {[
              'Navigate to [website]',
              "What's on screen?",
              'Search for [query]',
              'Click on [element]',
              'Analyze the page'
            ].map((cmd) => (
              <Badge key={cmd} variant="outline" className="text-xs">
                {cmd}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

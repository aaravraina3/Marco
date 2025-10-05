/**
 * Speech recognition utilities for voice input
 * Uses Web Speech API for browser-based speech recognition
 */

export interface SpeechRecognitionConfig {
  continuous?: boolean;
  interimResults?: boolean;
  language?: string;
  maxAlternatives?: number;
}

// Extend the Window interface to include webkitSpeechRecognition
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

export class VoiceRecognition {
  private recognition: any;
  private isListening: boolean = false;
  private onResultCallback?: (transcript: string, isFinal: boolean) => void;
  private onErrorCallback?: (error: string) => void;
  private onStartCallback?: () => void;
  private onEndCallback?: () => void;

  constructor(config?: SpeechRecognitionConfig) {
    // Check for browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      throw new Error('Speech recognition is not supported in this browser');
    }

    this.recognition = new SpeechRecognition();
    
    // Configure recognition
    this.recognition.continuous = config?.continuous ?? true;
    this.recognition.interimResults = config?.interimResults ?? true;
    this.recognition.language = config?.language ?? 'en-US';
    this.recognition.maxAlternatives = config?.maxAlternatives ?? 1;

    // Set up event handlers
    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    this.recognition.onstart = () => {
      console.log('Speech recognition started');
      this.isListening = true;
      this.onStartCallback?.();
    };

    this.recognition.onend = () => {
      console.log('Speech recognition ended');
      this.isListening = false;
      this.onEndCallback?.();
    };

    this.recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      this.isListening = false;
      
      let errorMessage = 'Speech recognition error';
      switch (event.error) {
        case 'no-speech':
          errorMessage = 'No speech detected';
          break;
        case 'audio-capture':
          errorMessage = 'No microphone found';
          break;
        case 'not-allowed':
          errorMessage = 'Microphone permission denied';
          break;
        case 'network':
          errorMessage = 'Network error';
          break;
      }
      
      this.onErrorCallback?.(errorMessage);
    };

    this.recognition.onresult = (event: any) => {
      const last = event.results.length - 1;
      const transcript = event.results[last][0].transcript;
      const isFinal = event.results[last].isFinal;

      console.log(`Transcript (${isFinal ? 'final' : 'interim'}):`, transcript);
      
      this.onResultCallback?.(transcript, isFinal);
    };

    this.recognition.onnomatch = () => {
      console.log('No speech match');
    };
  }

  public start() {
    if (!this.isListening) {
      try {
        this.recognition.start();
      } catch (error) {
        console.error('Failed to start recognition:', error);
        this.onErrorCallback?.('Failed to start recognition');
      }
    }
  }

  public stop() {
    if (this.isListening) {
      try {
        this.recognition.stop();
      } catch (e) {
        console.log('Recognition already stopped');
      }
    }
  }

  public abort() {
    // Force abort - more aggressive than stop
    try {
      this.recognition.abort();
      this.isListening = false;
    } catch (e) {
      console.log('Recognition already aborted');
    }
  }

  public onResult(callback: (transcript: string, isFinal: boolean) => void) {
    this.onResultCallback = callback;
  }

  public onError(callback: (error: string) => void) {
    this.onErrorCallback = callback;
  }

  public onStart(callback: () => void) {
    this.onStartCallback = callback;
  }

  public onEnd(callback: () => void) {
    this.onEndCallback = callback;
  }

  public getIsListening(): boolean {
    return this.isListening;
  }
}

// Create a new instance each time - NOT a singleton
// This ensures we can properly stop and restart recognition
export function getVoiceRecognition(config?: SpeechRecognitionConfig): VoiceRecognition {
  return new VoiceRecognition(config);
}

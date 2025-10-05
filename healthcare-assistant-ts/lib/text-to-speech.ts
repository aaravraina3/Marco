/**
 * Text-to-Speech utilities using Web Speech API
 * For production, replace with Eleven Labs SDK
 */

export interface TTSConfig {
  rate?: number;      // 0.1 to 10
  pitch?: number;     // 0 to 2
  volume?: number;    // 0 to 1
  voice?: string;     // Voice name
  language?: string;  // Language code
}

export class TextToSpeech {
  private synth: SpeechSynthesis | null = null;
  private voices: SpeechSynthesisVoice[] = [];
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private config: TTSConfig;
  private isInitialized: boolean = false;

  constructor(config?: TTSConfig) {
    // Only access window if we're in the browser
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      this.synth = window.speechSynthesis;
      this.initialize();
    }
    
    this.config = {
      rate: config?.rate ?? 1,
      pitch: config?.pitch ?? 1,
      volume: config?.volume ?? 1,
      voice: config?.voice ?? '',
      language: config?.language ?? 'en-US'
    };
  }

  private initialize() {
    if (!this.synth) return;
    
    // Load voices
    const loadVoices = () => {
      if (!this.synth) return;
      this.voices = this.synth.getVoices();
      this.isInitialized = true;
      console.log(`Loaded ${this.voices.length} TTS voices`);
    };

    // Some browsers load voices async
    if (this.synth.getVoices().length > 0) {
      loadVoices();
    } else {
      this.synth.addEventListener('voiceschanged', loadVoices);
    }
  }

  public speak(text: string, onEnd?: () => void): void {
    if (!this.synth || typeof window === 'undefined') return;
    
    // Cancel any ongoing speech
    this.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    
    // Apply configuration
    utterance.rate = this.config.rate!;
    utterance.pitch = this.config.pitch!;
    utterance.volume = this.config.volume!;
    utterance.lang = this.config.language!;

    // Set voice if specified and available
    if (this.config.voice && this.voices.length > 0) {
      const voice = this.voices.find(v => v.name === this.config.voice);
      if (voice) {
        utterance.voice = voice;
      } else {
        // Use default voice for the language
        const langVoice = this.voices.find(v => v.lang.startsWith(this.config.language!.split('-')[0]));
        if (langVoice) {
          utterance.voice = langVoice;
        }
      }
    }

    // Set up event handlers
    utterance.onend = () => {
      this.currentUtterance = null;
      onEnd?.();
    };

    utterance.onerror = (event) => {
      // Don't log "interrupted" errors as they are expected when we cancel
      if (event.error !== 'interrupted') {
        console.error('TTS error:', event.error);
      }
      this.currentUtterance = null;
    };

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
  }

  public pause(): void {
    if (this.synth && this.synth.speaking && !this.synth.paused) {
      this.synth.pause();
    }
  }

  public resume(): void {
    if (this.synth && this.synth.paused) {
      this.synth.resume();
    }
  }

  public cancel(): void {
    if (this.synth) {
      this.synth.cancel();
    }
    this.currentUtterance = null;
  }

  public isSpeaking(): boolean {
    return this.synth ? this.synth.speaking : false;
  }

  public isPaused(): boolean {
    return this.synth ? this.synth.paused : false;
  }

  public getVoices(): SpeechSynthesisVoice[] {
    return this.voices;
  }

  public setVoice(voiceName: string): void {
    this.config.voice = voiceName;
  }

  public setRate(rate: number): void {
    this.config.rate = Math.max(0.1, Math.min(10, rate));
  }

  public setPitch(pitch: number): void {
    this.config.pitch = Math.max(0, Math.min(2, pitch));
  }

  public setVolume(volume: number): void {
    this.config.volume = Math.max(0, Math.min(1, volume));
  }
}

// Singleton instance
let ttsInstance: TextToSpeech | null = null;

export function getTextToSpeech(config?: TTSConfig): TextToSpeech {
  if (!ttsInstance) {
    ttsInstance = new TextToSpeech(config);
  }
  return ttsInstance;
}

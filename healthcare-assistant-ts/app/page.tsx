'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { BrowserDisplay } from '@/components/browser-display';
import { VoiceAnimation } from '@/components/voice-animation';
import { AgentResponses, Message } from '@/components/agent-responses';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Send, Terminal, Volume2, VolumeX } from 'lucide-react';
import { apiClient } from '@/lib/api';
import { getTextToSpeech } from '@/lib/text-to-speech';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [commandInput, setCommandInput] = useState('');
  const [serverStatus, setServerStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [isTTSEnabled, setIsTTSEnabled] = useState(true);
  const [isTTSSpeaking, setIsTTSSpeaking] = useState(false);
  const ttsRef = useRef<{ isSpeaking: () => boolean; cancel: () => void; speak: (text: string, callback: () => void) => void } | null>(null);

  // Initialize TTS on client side
  useEffect(() => {
    ttsRef.current = getTextToSpeech();
  }, []);
  
  // Check server health on mount
  useEffect(() => {
    const checkServer = async () => {
      setServerStatus('checking');
      const isOnline = await apiClient.checkHealth();
      setServerStatus(isOnline ? 'online' : 'offline');
    };
    
    checkServer();
    // Check every 5 seconds
    const interval = setInterval(checkServer, 5000);
    return () => clearInterval(interval);
  }, []);

  const addMessage = useCallback((role: 'user' | 'assistant' | 'system', content: string, options?: Partial<Message>) => {
    const newMessage: Message = {
      id: `msg-${Date.now()}`,
      role,
      content,
      timestamp: new Date(),
      ...options
    };
    setMessages(prev => [...prev, newMessage]);
    
    // Speak assistant messages if TTS is enabled
    if (role === 'assistant' && isTTSEnabled && !options?.error && ttsRef.current) {
      // Cancel any existing speech first
      if (ttsRef.current.isSpeaking()) {
        ttsRef.current.cancel();
      }
      
      // Remove markdown and clean up text for speech
      const cleanText = content
        .replace(/[*_~`#]/g, '')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/```[\s\S]*?```/g, 'code block')
        .replace(/✅|❌|🔧/g, '') // Remove emojis
        .trim();
      
      if (cleanText && cleanText.length > 0) {
        // Small delay to ensure previous speech is fully cancelled
        setTimeout(() => {
          if (ttsRef.current && isTTSEnabled) {
            setIsTTSSpeaking(true);
            ttsRef.current.speak(cleanText, () => {
              // Called when TTS finishes
              setIsTTSSpeaking(false);
            });
          }
        }, 100);
      }
    }
    
    return newMessage;
  }, [isTTSEnabled]);

  const handleTranscript = useCallback(async (transcript: string) => {
    if (!transcript.trim()) return;
    
    // Stop listening immediately while processing
    setIsListening(false);
    
    // Add user message
    addMessage('user', transcript);
    
    // Process the command
    await processCommand(transcript);
    
    // Note: Don't automatically restart listening - let user control it
    // Processing state will be cleared by processCommand
  }, []);

  const processCommand = async (command: string) => {
    setIsProcessing(true);
    
    try {
      // Start streaming if not already
      if (!isStreaming) {
        setIsStreaming(true);
      }
      
      // Process through LangChain agent
      const response = await apiClient.processWithAgent(command);
      
      if (response.success) {
        // Add the agent's response
        addMessage('assistant', response.response, { 
          toolsCalled: response.tools_used && response.tools_used.length > 0 
        });
      } else {
        addMessage('assistant', `Error: ${response.error || 'Failed to process command'}`, { 
          error: true 
        });
      }
    } catch (error: unknown) {
      // If agent endpoint fails, fallback to simpler logic
      console.error('Agent endpoint failed, falling back:', error);
      
      // Simple fallback for basic commands
      try {
        if (command.toLowerCase().includes('navigate to') || command.toLowerCase().includes('go to')) {
          const urlMatch = command.match(/(?:navigate to|go to)\s+(https?:\/\/\S+|\S+\.com|\S+\.org|\S+\.net)/i);
          if (urlMatch) {
            const url = urlMatch[1].trim();
            const fullUrl = url.startsWith('http') ? url : `https://${url}`;
            await apiClient.navigateTo(fullUrl);
            addMessage('assistant', `Navigating to ${fullUrl}...`, { toolsCalled: true });
            return;
          }
        }
        
        // Try analysis for questions
        if (command.toLowerCase().includes('what') || command.toLowerCase().includes('describe')) {
          const response = await apiClient.analyzeScreenshot(command);
          addMessage('assistant', response.analysis, { toolsCalled: true });
        } else {
          // Try direct execution
          const response = await apiClient.executeCommand(command);
          if (response.success) {
            addMessage('assistant', 'Command executed', { toolsCalled: true });
          } else {
            throw new Error(response.error || 'Failed');
          }
        }
      } catch (fallbackError: unknown) {
        addMessage('assistant', `Failed to process command: ${error instanceof Error ? error.message : String(error)}`, { 
          error: true 
        });
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendCommand = async () => {
    if (!commandInput.trim() || isProcessing) return;
    
    const command = commandInput;
    setCommandInput('');
    addMessage('user', command);
    await processCommand(command);
  };

  const handleClearMessages = () => {
    setMessages([]);
  };

  const handleInitializeBrowser = async () => {
    setIsProcessing(true);
    try {
      // Initialize backend browser via API client
      const res = await apiClient.initBrowser();
      if (!res.success) {
        throw new Error(res.error || 'Init failed');
      }
      addMessage('system', `Browser initialized${res.url ? ` at ${res.url}` : ''}. Click "Start Streaming" to see the browser and start interacting with it.`);
    } catch (error) {
      addMessage('assistant', `Failed to initialize browser: ${error instanceof Error ? error.message : String(error)}`, { 
        error: true 
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="h-screen w-screen bg-background overflow-hidden">
      {/* Header */}
      <header className="border-b bg-card px-6 py-4 w-full">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold">Healthcare Intake Assistant</h1>
            <Badge variant={serverStatus === 'online' ? 'default' : serverStatus === 'checking' ? 'secondary' : 'destructive'}>
              Server: {serverStatus}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="default"
              size="sm"
              onClick={handleInitializeBrowser}
              disabled={isProcessing || serverStatus !== 'online'}
            >
              🚀 Start Browser
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setIsTTSEnabled(!isTTSEnabled);
                if (ttsRef.current && ttsRef.current.isSpeaking()) {
                  ttsRef.current.cancel();
                  setIsTTSSpeaking(false);
                }
              }}
              title={isTTSEnabled ? "Disable Text-to-Speech" : "Enable Text-to-Speech"}
            >
              {isTTSEnabled ? (
                <Volume2 className="w-4 h-4" />
              ) : (
                <VolumeX className="w-4 h-4" />
              )}
            </Button>
            <Button
              variant="outline"
              onClick={() => window.open('http://localhost:3003/api/health', '_blank')}
            >
              <Terminal className="w-4 h-4 mr-2" />
              API Health
            </Button>
            <Button
              variant="outline"
              onClick={() => window.open('/instructions', '_blank')}
            >
              Instructions
            </Button>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex h-[calc(100vh-73px)]" style={{ width: '100vw', boxSizing: 'border-box' }}>
        {/* Left Panel - Voice Control */}
        <aside className="shrink-0 border-r border-4 border-red-500 bg-card p-1" style={{ width: '25%', boxSizing: 'border-box' }}>
        <VoiceAnimation
          isListening={isListening}
          onListeningChange={setIsListening}
          onTranscript={handleTranscript}
          isProcessing={isProcessing}
        />
        </aside>

        {/* Center - Browser Display */}
        <div className="shrink-0 border-4 border-blue-500 p-1" style={{ width: '50%', boxSizing: 'border-box' }}>
          <BrowserDisplay
            isStreaming={isStreaming}
            onStreamingChange={setIsStreaming}
            refreshRate={500}  // Reduced to 2 FPS to prevent flickering
          />
        </div>

        {/* Right Panel - Agent Responses */}
        <aside className="shrink-0 border-l border-4 border-green-500 bg-card p-1" style={{ width: '25%', boxSizing: 'border-box' }}>
          <div className="h-full flex flex-col gap-4">
            <div className="flex-1 min-h-0 overflow-hidden">
              <AgentResponses
                messages={messages}
                onClear={handleClearMessages}
                isProcessing={isProcessing}
              />
            </div>
            
            {/* Command Input */}
            <Card className="p-3">
              <div className="flex gap-2">
                <Input
                  placeholder="Type a command..."
                  value={commandInput}
                  onChange={(e) => setCommandInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendCommand()}
                  disabled={isProcessing || serverStatus !== 'online'}
                />
                <Button 
                  onClick={handleSendCommand} 
                  disabled={isProcessing || !commandInput.trim() || serverStatus !== 'online'}
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </Card>
          </div>
        </aside>
      </div>
    </main>
  );
}
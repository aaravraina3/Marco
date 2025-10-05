'use client';

import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Trash2, User, Bot } from 'lucide-react';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  toolsCalled?: boolean;
  error?: boolean;
}

interface AgentResponsesProps {
  messages: Message[];
  onClear?: () => void;
  isProcessing?: boolean;
}

export function AgentResponses({ 
  messages, 
  onClear, 
  isProcessing = false 
}: AgentResponsesProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);

  return (
    <Card className="h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between p-4 border-b flex-shrink-0">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold">Agent Responses</h3>
          {isProcessing && (
            <Badge variant="default" className="animate-pulse">
              Processing...
            </Badge>
          )}
        </div>
        {messages.length > 0 && onClear && (
          <Button
            onClick={onClear}
            variant="ghost"
            size="sm"
            className="text-muted-foreground"
          >
            <Trash2 className="w-4 h-4 mr-1" />
            Clear
          </Button>
        )}
      </div>
      
      <ScrollArea className="flex-1 min-h-0" ref={scrollAreaRef}>
        <div className="p-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-muted-foreground">
              <Bot className="w-12 h-12 mb-3 opacity-50" />
              <p className="text-sm">No messages yet</p>
              <p className="text-xs mt-1">Start by giving a voice command or typing</p>
            </div>
          ) : (
            <div className="space-y-4 pr-2">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-primary" />
                  </div>
                )}
                
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : message.error
                      ? 'bg-destructive/10 border border-destructive/20'
                      : 'bg-muted'
                  }`}
                >
                  {message.toolsCalled && (
                    <Badge variant="outline" className="mb-2">
                      🔧 Tools Used
                    </Badge>
                  )}
                  
                  <div className="text-sm">
                    {message.role === 'assistant' ? (
                      <div className="prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                            ul: ({ children }) => <ul className="list-disc pl-4 mb-2">{children}</ul>,
                            ol: ({ children }) => <ol className="list-decimal pl-4 mb-2">{children}</ol>,
                            li: ({ children }) => <li className="mb-1">{children}</li>,
                            code: ({ inline, children, ...props }) => (
                              inline 
                                ? <code className="px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded text-xs" {...props}>{children}</code>
                                : <pre className="p-2 bg-black/10 dark:bg-white/10 rounded overflow-x-auto"><code {...props}>{children}</code></pre>
                            ),
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p>{message.content}</p>
                    )}
                  </div>
                  
                  <p className="text-xs opacity-60 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
                
                {message.role === 'user' && (
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))}
            
            {isProcessing && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-primary animate-pulse" />
                </div>
                <div className="bg-muted rounded-lg p-3">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}

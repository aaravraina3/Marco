'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { apiClient } from '@/lib/api';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface BrowserDisplayProps {
  isStreaming: boolean;
  onStreamingChange: (streaming: boolean) => void;
  refreshRate?: number; // milliseconds
}

export function BrowserDisplay({ 
  isStreaming, 
  onStreamingChange, 
  refreshRate = 100 // 10 FPS default
}: BrowserDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [currentUrl, setCurrentUrl] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const fpsFramesRef = useRef<number[]>([]);

  const fetchScreenshot = useCallback(async () => {
    if (!canvasRef.current) return;
    
    try {
      const data = await apiClient.getScreenshot();
      
      // Draw the screenshot on canvas
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Clear canvas first
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scaling to maintain aspect ratio
        const imgAspect = img.width / img.height;
        const canvasAspect = canvas.width / canvas.height;
        
        let drawWidth = canvas.width;
        let drawHeight = canvas.height;
        let offsetX = 0;
        let offsetY = 0;
        
        if (imgAspect > canvasAspect) {
          // Image is wider than canvas
          drawHeight = canvas.width / imgAspect;
          offsetY = (canvas.height - drawHeight) / 2;
        } else {
          // Image is taller than canvas
          drawWidth = canvas.height * imgAspect;
          offsetX = (canvas.width - drawWidth) / 2;
        }
        
        // Draw the image scaled and centered
        ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
        
        // Update stats
        setFrameCount(prev => prev + 1);
        setCurrentUrl(data.url || '');
        
        // Calculate FPS
        const now = Date.now();
        fpsFramesRef.current.push(now);
        fpsFramesRef.current = fpsFramesRef.current.filter(t => now - t < 1000);
        setFps(fpsFramesRef.current.length);
      };
      img.src = `data:image/png;base64,${data.screenshot_base64}`;
      setError(null);
    } catch (err) {
      console.error('Failed to fetch screenshot:', err);
      setError('Failed to fetch screenshot');
    }
  }, []);

  useEffect(() => {
    if (isStreaming) {
      // Start polling for screenshots
      fetchScreenshot(); // Fetch immediately
      intervalRef.current = setInterval(fetchScreenshot, refreshRate);
    } else {
      // Stop polling
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isStreaming, refreshRate, fetchScreenshot]);

  const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    
    // Get click position relative to canvas
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    
    // Scale to canvas internal dimensions
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = Math.round(canvasX * scaleX);
    const y = Math.round(canvasY * scaleY);
    
    // Ensure coordinates are within bounds
    const boundedX = Math.max(0, Math.min(x, 1280));
    const boundedY = Math.max(0, Math.min(y, 720));
    
    try {
      await apiClient.clickAt(boundedX, boundedY);
      console.log(`Clicked at (${boundedX}, ${boundedY})`);
    } catch (err) {
      console.error('Click failed:', err);
    }
  };

  return (
    <Card className="relative h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-4">
          <Button
            onClick={() => onStreamingChange(!isStreaming)}
            variant={isStreaming ? "destructive" : "default"}
          >
            {isStreaming ? 'Stop' : 'Start'} Streaming
          </Button>
          <Badge variant={isStreaming ? "default" : "secondary"}>
            {isStreaming ? 'Live' : 'Paused'}
          </Badge>
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>FPS: {fps}</span>
          <span>Frames: {frameCount}</span>
        </div>
      </div>
      
      <div className="flex-1 p-4 flex items-center justify-center bg-black/5 overflow-hidden">
        <div className="relative w-full h-full flex items-center justify-center">
          <canvas
            ref={canvasRef}
            width={1280}
            height={720}
            className="border rounded cursor-crosshair"
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              width: 'auto',
              height: 'auto',
              objectFit: 'contain'
            }}
            onClick={handleCanvasClick}
          />
          {!isStreaming && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded">
              <div className="text-white text-center">
                <p className="text-lg font-medium">Browser Stream Paused</p>
                <p className="text-sm mt-2">Click "Start Streaming" to begin</p>
              </div>
            </div>
          )}
          {error && (
            <div className="absolute top-2 left-2 bg-red-500 text-white px-2 py-1 rounded text-sm">
              {error}
            </div>
          )}
        </div>
      </div>
      
      {currentUrl && (
        <div className="p-2 border-t bg-muted/50">
          <p className="text-xs text-muted-foreground truncate" title={currentUrl}>
            URL: {currentUrl.length > 60 ? currentUrl.substring(0, 60) + '...' : currentUrl}
          </p>
        </div>
      )}
    </Card>
  );
}

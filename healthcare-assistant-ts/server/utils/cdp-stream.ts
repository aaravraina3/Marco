import { WebSocket, WebSocketServer } from 'ws';
import { stagehandService } from '../services/stagehand.service';

interface CDPFrame {
  type: 'frame';
  data: string;
  metadata?: any;
}

interface CDPMessage {
  type: 'click' | 'move' | 'down' | 'up' | 'scroll' | 'keypress' | 'type' | 'goto';
  x?: number;
  y?: number;
  deltaY?: number;
  key?: string;
  text?: string;
  url?: string;
}

export class CDPStreamManager {
  private wss: WebSocketServer | null = null;
  private activeConnections: Set<WebSocket> = new Set();
  private isStreaming = false;
  
  initialize(server: any): void {
    // Create WebSocket server
    this.wss = new WebSocketServer({ server });
    
    this.wss.on('connection', (ws: WebSocket) => {
      console.log('🔌 Client connected for CDP stream');
      this.activeConnections.add(ws);
      
      // Start streaming for this client
      this.handleClientConnection(ws);
      
      ws.on('close', () => {
        console.log('🔌 Client disconnected from CDP stream');
        this.activeConnections.delete(ws);
        
        // Stop streaming if no clients
        if (this.activeConnections.size === 0) {
          this.stopStreaming();
        }
      });
      
      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.activeConnections.delete(ws);
      });
      
      // Handle messages from client
      ws.on('message', async (message: string) => {
        try {
          const msg: CDPMessage = JSON.parse(message.toString());
          await this.handleClientMessage(msg);
        } catch (error) {
          console.error('Error handling client message:', error);
        }
      });
    });
  }
  
  private async handleClientConnection(ws: WebSocket): Promise<void> {
    try {
      // Start streaming if not already
      if (!this.isStreaming) {
        await this.startStreaming();
      }
      
      // Send initial status
      ws.send(JSON.stringify({ type: 'status', connected: true }));
    } catch (error) {
      console.error('Error handling client connection:', error);
      ws.send(JSON.stringify({ 
        type: 'error', 
        message: 'Failed to initialize streaming' 
      }));
    }
  }
  
  private async startStreaming(): Promise<void> {
    if (this.isStreaming) return;
    
    const cdpSession = stagehandService.getCDPSession();
    if (!cdpSession) {
      console.error('CDP session not available');
      return;
    }
    
    try {
      // Start screencast
      await stagehandService.startScreencast('jpeg', 60);
      
      // Listen for frames
      cdpSession.on('Page.screencastFrame', async (params) => {
        // Send frame to all connected clients
        const frame: CDPFrame = {
          type: 'frame',
          data: params.data,
          metadata: params.metadata
        };
        
        this.broadcast(JSON.stringify(frame));
        
        // Acknowledge frame
        try {
          await cdpSession.send('Page.screencastFrameAck', {
            sessionId: params.sessionId
          });
        } catch (error) {
          console.error('Failed to acknowledge frame:', error);
        }
      });
      
      this.isStreaming = true;
      console.log('📹 CDP streaming started');
    } catch (error) {
      console.error('Failed to start streaming:', error);
    }
  }
  
  private async stopStreaming(): Promise<void> {
    if (!this.isStreaming) return;
    
    try {
      await stagehandService.stopScreencast();
      this.isStreaming = false;
      console.log('📹 CDP streaming stopped');
    } catch (error) {
      console.error('Failed to stop streaming:', error);
    }
  }
  
  private async handleClientMessage(msg: CDPMessage): Promise<void> {
    const page = stagehandService.getPage();
    if (!page) {
      console.error('Page not available');
      return;
    }
    
    try {
      switch (msg.type) {
        case 'click':
          if (msg.x !== undefined && msg.y !== undefined) {
            await page.mouse.click(msg.x, msg.y);
          }
          break;
          
        case 'move':
          if (msg.x !== undefined && msg.y !== undefined) {
            await page.mouse.move(msg.x, msg.y);
          }
          break;
          
        case 'down':
          await page.mouse.down();
          break;
          
        case 'up':
          await page.mouse.up();
          break;
          
        case 'scroll':
          if (msg.deltaY !== undefined) {
            await page.mouse.wheel(0, msg.deltaY);
          }
          break;
          
        case 'keypress':
          if (msg.key) {
            await page.keyboard.press(msg.key);
          }
          break;
          
        case 'type':
          if (msg.text) {
            await page.keyboard.type(msg.text);
          }
          break;
          
        case 'goto':
          if (msg.url) {
            await stagehandService.navigateTo(msg.url);
          }
          break;
          
        default:
          console.log('Unknown message type:', msg.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }
  
  private broadcast(message: string): void {
    this.activeConnections.forEach(ws => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }
  
  sendToAll(message: any): void {
    this.broadcast(JSON.stringify(message));
  }
}

// Create singleton instance
export const cdpStreamManager = new CDPStreamManager();

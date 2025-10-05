import express from 'express';
import { WebSocketServer } from 'ws';
import * as playwright from 'playwright';
import cors from 'cors';
import http from 'http';
import path from 'path';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

let browser: playwright.Browser | null = null;
let page: playwright.Page | null = null;
let cdpSession: playwright.CDPSession | null = null;

// Initialize browser with CDP
async function initBrowser() {
  console.log('🚀 Launching Chromium with CDP...');
  
  browser = await playwright.chromium.launch({
    headless: false, // Set to true if you don't want to see the browser
    args: [
      '--remote-debugging-port=9222',
      '--no-sandbox',
      '--disable-setuid-sandbox'
    ]
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 }
  });
  
  page = await context.newPage();
  
  // Create CDP session
  cdpSession = await context.newCDPSession(page);
  
  // Navigate to a default page
  await page.goto('https://www.google.com');
  
  console.log('✅ Browser initialized with CDP');
  return { browser, page, cdpSession };
}

// Handle WebSocket connections
wss.on('connection', async (ws) => {
  console.log('🔌 Client connected');
  
  if (!page || !cdpSession) {
    await initBrowser();
  }
  
  // Start screencast
  try {
    await cdpSession!.send('Page.startScreencast', {
      format: 'jpeg',
      quality: 60,
      maxWidth: 1280,
      maxHeight: 720,
      everyNthFrame: 1
    });
    
    console.log('📹 Screencast started');
  } catch (error) {
    console.error('Failed to start screencast:', error);
  }
  
  // Listen for screencast frames
  cdpSession!.on('Page.screencastFrame', async (params) => {
    // Send frame to client
    ws.send(JSON.stringify({
      type: 'frame',
      data: params.data,
      metadata: params.metadata
    }));
    
    // Acknowledge frame
    try {
      await cdpSession!.send('Page.screencastFrameAck', {
        sessionId: params.sessionId
      });
    } catch (error) {
      console.error('Failed to acknowledge frame:', error);
    }
  });
  
  // Handle messages from client
  ws.on('message', async (message) => {
    try {
      const msg = JSON.parse(message.toString());
      
      switch (msg.type) {
        case 'click':
          await page!.mouse.click(msg.x, msg.y);
          break;
          
        case 'move':
          await page!.mouse.move(msg.x, msg.y);
          break;
          
        case 'down':
          await page!.mouse.down();
          break;
          
        case 'up':
          await page!.mouse.up();
          break;
          
        case 'scroll':
          await page!.mouse.wheel(0, msg.deltaY);
          break;
          
        case 'keypress':
          await page!.keyboard.press(msg.key);
          break;
          
        case 'type':
          await page!.keyboard.type(msg.text);
          break;
          
        case 'goto':
          await page!.goto(msg.url);
          break;
          
        default:
          console.log('Unknown message type:', msg.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  });
  
  ws.on('close', async () => {
    console.log('🔌 Client disconnected');
    
    // Stop screencast
    try {
      await cdpSession!.send('Page.stopScreencast');
    } catch (error) {
      console.error('Failed to stop screencast:', error);
    }
  });
  
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
});

// API endpoint to get current page info
app.get('/api/status', async (req, res) => {
  if (!page) {
    return res.json({ status: 'not_initialized' });
  }
  
  const url = page.url();
  const title = await page.title();
  
  res.json({
    status: 'ready',
    url,
    title
  });
});

// API endpoint to navigate
app.post('/api/navigate', async (req, res) => {
  const { url } = req.body;
  
  if (!page) {
    return res.status(503).json({ error: 'Browser not initialized' });
  }
  
  try {
    await page.goto(url);
    res.json({ success: true, url: page.url() });
  } catch (error) {
    res.status(500).json({ error: String(error) });
  }
});

// Cleanup on exit
process.on('SIGINT', async () => {
  console.log('\n🧹 Cleaning up...');
  
  if (cdpSession) {
    try {
      await cdpSession.send('Page.stopScreencast');
    } catch (error) {
      console.error('Failed to stop screencast:', error);
    }
  }
  
  if (browser) {
    await browser.close();
  }
  
  process.exit(0);
});

const PORT = process.env.PORT || 3002;

server.listen(PORT, () => {
  console.log(`🚀 CDP Server running on http://localhost:${PORT}`);
  console.log(`📡 WebSocket available on ws://localhost:${PORT}`);
  console.log(`\n📝 Instructions:`);
  console.log(`1. Start this server: pnpm run server`);
  console.log(`2. Open the client: http://localhost:${PORT}/`);
  console.log(`3. The browser display will stream via WebSocket`);
});

// Initialize browser on startup
initBrowser().catch(console.error);

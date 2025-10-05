# FastAPI CDP WebSocket Streaming

This enhanced version of the CDP POC works with your existing Python FastAPI server instead of requiring a separate Express/TypeScript server.

## 🚀 Quick Start

### Option 1: Standalone CDP Server (Testing)
```bash
# From the cdp-poc directory
python fastapi_cdp_server.py
```

Then open: http://localhost:3002

### Option 2: Integrate with Your Existing Server

Use the enhanced FastAPI server that includes CDP streaming:

```bash
# From the src directory
python fastapi_stagehand_server_with_cdp.py
```

This gives you:
- All your existing endpoints (`/execute`, `/analyze`, etc.)
- CDP WebSocket streaming at `/ws`
- CDP client interface at `/cdp`

## 📡 WebSocket Protocol

The WebSocket endpoint (`ws://localhost:8000/ws`) accepts these message types:

### From Client → Server
```javascript
// Click at coordinates
{ "type": "click", "x": 100, "y": 200 }

// Move mouse
{ "type": "move", "x": 100, "y": 200 }

// Mouse down/up
{ "type": "down" }
{ "type": "up" }

// Scroll
{ "type": "scroll", "deltaY": 100 }

// Type text
{ "type": "type", "text": "Hello" }

// Press key
{ "type": "keypress", "key": "Enter" }

// Navigate to URL
{ "type": "goto", "url": "https://example.com" }
```

### From Server → Client
```javascript
// Frame data (CDP screencast)
{
  "type": "frame",
  "data": "base64_jpeg_data",
  "metadata": { /* frame metadata */ }
}

// Status messages
{ "type": "status", "connected": true }

// Execution notifications
{ "type": "execution_start", "instruction": "..." }
{ "type": "execution_complete", "instruction": "...", "success": true }
{ "type": "execution_error", "instruction": "...", "error": "..." }
```

## 🏗️ Architecture

```
FastAPI Server (Python)
├── Stagehand Browser Control
├── CDP Session Management
├── WebSocket Handler (/ws)
├── HTTP Endpoints
│   ├── /execute - Computer Use agent
│   ├── /analyze - Gemini analysis
│   └── /status - Browser status
└── Static Files
    └── /cdp - CDP client interface

Browser Client (HTML/JS)
├── Canvas for CDP frames
├── WebSocket connection
├── Mouse/keyboard capture
└── URL navigation bar
```

## 🔧 Integration with Next.js

To use this with your Next.js frontend:

```typescript
// Connect to the CDP WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'frame') {
    // Draw frame to canvas
    const img = new Image();
    img.src = `data:image/jpeg;base64,${msg.data}`;
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
    };
  }
};

// Send interactions
ws.send(JSON.stringify({ 
  type: 'click', 
  x: 100, 
  y: 200 
}));
```

## 🎯 Benefits of This Approach

1. **Single Backend**: Keep your existing Python/FastAPI server
2. **No Node.js Required**: Everything runs in Python
3. **Integrated Features**: CDP streaming works alongside your existing endpoints
4. **Shared Browser Instance**: Same browser for agent execution and streaming
5. **WebSocket Notifications**: Get real-time updates during agent execution

## 📝 Files Overview

- `cdp_websocket_handler.py` - CDP WebSocket handler module
- `fastapi_stagehand_server_with_cdp.py` - Enhanced FastAPI server with CDP
- `fastapi_cdp_server.py` - Standalone CDP server for testing
- `public/index.html` - Browser client interface

## 🔍 Testing

1. Start the enhanced server:
```bash
python src/fastapi_stagehand_server_with_cdp.py
```

2. Open the CDP client:
```
http://localhost:8000/cdp
```

3. Test the existing API endpoints:
```bash
# Execute browser action
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Search for healthcare forms"}'

# The browser actions will be visible in the CDP stream!
```

## 🔄 Next Steps

Now you can:
1. Keep your Python FastAPI backend
2. Build the Next.js frontend that connects to this backend
3. Use the CDP WebSocket for real-time browser display
4. All your existing voice agent code continues to work!

The best part: You don't need to maintain two separate servers - everything runs from your Python backend!


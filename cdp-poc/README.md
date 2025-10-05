# CDP Browser Stream - Proof of Concept

This is a minimal proof of concept showing how to stream a Chromium browser to a web application using Chrome DevTools Protocol (CDP).

## 🎯 What This Demo Shows

- **Real-time browser streaming** using CDP screencast
- **Interactive control** - click, type, scroll in the browser
- **WebSocket communication** between server and client
- **Canvas-based rendering** of browser frames
- **URL navigation** from the web interface

## 🚀 Getting Started

### Prerequisites

- Node.js (v18 or higher)
- pnpm package manager

### Installation

```bash
# Install dependencies
pnpm install

# Install Chromium (if not already installed)
pnpm exec playwright install chromium
```

### Running the Demo

```bash
# Start the server
pnpm run dev

# Or alternatively
pnpm run server
```

The server will start on `http://localhost:3002`

Open your browser and navigate to `http://localhost:3002` to see the CDP stream in action!

## 🏗️ Architecture

### Server (`server.ts`)
- Launches Chromium with CDP enabled
- Creates WebSocket server for streaming
- Handles CDP screencast frames
- Forwards user interactions to browser

### Client (`public/index.html`)
- Connects to WebSocket server
- Renders frames on HTML5 Canvas
- Captures user interactions (mouse, keyboard)
- Sends interactions back to server

## 📡 How It Works

1. **Browser Launch**: Server launches Chromium with `--remote-debugging-port`
2. **CDP Session**: Creates a CDP session with the browser
3. **Screencast**: Starts CDP screencast to capture frames
4. **WebSocket Stream**: Frames are sent to client via WebSocket
5. **Canvas Rendering**: Client renders frames on canvas
6. **User Interaction**: Mouse/keyboard events are captured and sent back
7. **Browser Control**: Server forwards interactions to browser via Playwright

## 🎮 Features

- **Click** - Click anywhere on the page
- **Type** - Type when focused on input fields
- **Scroll** - Use mouse wheel to scroll
- **Navigate** - Use the URL bar to navigate to any website
- **Drag** - Hold and drag to select text or move elements

## 📝 API Endpoints

- `GET /api/status` - Get current browser status
- `POST /api/navigate` - Navigate to a URL
- `WebSocket /` - Real-time browser stream

## 🔧 Configuration

You can modify these settings in `server.ts`:

- **Viewport Size**: Default is 1280x720
- **Frame Quality**: Default is 60% JPEG quality
- **Frame Rate**: Every frame is captured
- **Headless Mode**: Set `headless: true` to run without UI

## 📦 Tech Stack

- **TypeScript** - Type-safe development
- **Express** - HTTP server
- **WebSocket (ws)** - Real-time communication
- **Playwright** - Browser automation
- **CDP** - Chrome DevTools Protocol

## 🚨 Known Limitations

- Single client support (multiple clients would need session management)
- No audio streaming
- Basic keyboard support (special keys may need additional handling)
- No clipboard support
- Context menus are browser-native

## 🔮 Potential Enhancements

- Multiple browser sessions
- Audio streaming
- Better keyboard handling
- Clipboard integration
- Touch gesture support
- Recording capabilities
- Remote browser support

## 📄 License

MIT

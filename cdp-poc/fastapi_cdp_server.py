#!/usr/bin/env python3
"""
CDP WebSocket streaming for FastAPI server
Integrates with existing Stagehand browser to stream frames via WebSocket
"""

import asyncio
import json
import base64
import logging
from typing import Optional, Dict, Set
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Stagehand and browser automation (from existing server)
from stagehand import Stagehand
from playwright.async_api import Page, CDPSession

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for browser and CDP
stagehand_client: Optional[Stagehand] = None
playwright_page: Optional[Page] = None
cdp_session: Optional[CDPSession] = None
active_connections: Set[WebSocket] = set()
is_screencasting = False


# ===== LIFESPAN MANAGER =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize browser and CDP on startup, cleanup on shutdown"""
    global stagehand_client, playwright_page, cdp_session
    
    logger.info("🚀 Starting CDP WebSocket Server...")
    
    try:
        # Initialize Stagehand browser
        logger.info("🤖 Initializing Stagehand browser...")
        
        stagehand_client = Stagehand(
            env="LOCAL",
            headless=False,  # Show browser window
            verbose=True
        )
        
        await stagehand_client.init()
        logger.info("✅ Stagehand initialized")
        
        # Get the Playwright page
        playwright_page = stagehand_client.page
        
        # Set viewport size
        await playwright_page.set_viewport_size({"width": 1280, "height": 720})
        logger.info("✅ Set viewport to 1280x720")
        
        # Create CDP session for screencasting
        context = playwright_page.context()
        cdp_session = await context.newCDPSession(playwright_page)
        logger.info("✅ CDP session created")
        
        # Navigate to default page
        await playwright_page.goto("https://www.google.com")
        logger.info("✅ Navigated to Google")
        
        logger.info("🌐 CDP WebSocket server ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    # Yield control to FastAPI
    yield
    
    # Cleanup on shutdown
    logger.info("🧹 Shutting down server...")
    
    try:
        if cdp_session and is_screencasting:
            await cdp_session.send("Page.stopScreencast")
            
        if stagehand_client:
            await stagehand_client.close()
            logger.info("✅ Stagehand closed")
    except Exception as e:
        logger.error(f"⚠️ Error during cleanup: {e}")


# ===== FASTAPI APP =====
app = FastAPI(
    title="CDP WebSocket Server",
    description="WebSocket server for CDP browser streaming",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the HTML client
app.mount("/static", StaticFiles(directory="public"), name="static")


# ===== CDP WEBSOCKET HANDLER =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for CDP streaming"""
    global cdp_session, playwright_page, is_screencasting
    
    await websocket.accept()
    active_connections.add(websocket)
    logger.info("🔌 Client connected for CDP stream")
    
    try:
        # Start screencast if not already running
        if not is_screencasting and cdp_session:
            await start_screencast()
        
        # Send initial status
        await websocket.send_json({"type": "status", "connected": True})
        
        # Handle messages from client
        while True:
            message = await websocket.receive_text()
            msg_data = json.loads(message)
            
            # Process client commands
            await handle_client_message(msg_data, websocket)
            
    except WebSocketDisconnect:
        logger.info("🔌 Client disconnected from CDP stream")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        
        # Stop screencast if no more connections
        if len(active_connections) == 0 and is_screencasting:
            await stop_screencast()


async def start_screencast():
    """Start CDP screencast"""
    global cdp_session, is_screencasting
    
    if not cdp_session or is_screencasting:
        return
    
    try:
        # Start screencast
        await cdp_session.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 60,
            "maxWidth": 1280,
            "maxHeight": 720,
            "everyNthFrame": 1
        })
        
        # Set up frame handler
        async def on_frame(params):
            # Send frame to all connected clients
            frame_data = {
                "type": "frame",
                "data": params["data"],
                "metadata": params.get("metadata", {})
            }
            
            # Broadcast to all connections
            disconnected = set()
            for connection in active_connections:
                try:
                    await connection.send_json(frame_data)
                except:
                    disconnected.add(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                active_connections.discard(conn)
            
            # Acknowledge frame
            try:
                await cdp_session.send("Page.screencastFrameAck", {
                    "sessionId": params["sessionId"]
                })
            except Exception as e:
                logger.error(f"Failed to acknowledge frame: {e}")
        
        # Register the frame handler
        cdp_session.on("Page.screencastFrame", on_frame)
        
        is_screencasting = True
        logger.info("📹 CDP screencast started")
        
    except Exception as e:
        logger.error(f"Failed to start screencast: {e}")


async def stop_screencast():
    """Stop CDP screencast"""
    global cdp_session, is_screencasting
    
    if not cdp_session or not is_screencasting:
        return
    
    try:
        await cdp_session.send("Page.stopScreencast")
        is_screencasting = False
        logger.info("📹 CDP screencast stopped")
    except Exception as e:
        logger.error(f"Failed to stop screencast: {e}")


async def handle_client_message(msg: Dict, websocket: WebSocket):
    """Handle messages from client"""
    global playwright_page
    
    if not playwright_page:
        await websocket.send_json({"type": "error", "message": "Browser not initialized"})
        return
    
    msg_type = msg.get("type")
    
    try:
        if msg_type == "click":
            x = msg.get("x", 0)
            y = msg.get("y", 0)
            await playwright_page.mouse.click(x, y)
            
        elif msg_type == "move":
            x = msg.get("x", 0)
            y = msg.get("y", 0)
            await playwright_page.mouse.move(x, y)
            
        elif msg_type == "down":
            await playwright_page.mouse.down()
            
        elif msg_type == "up":
            await playwright_page.mouse.up()
            
        elif msg_type == "scroll":
            delta_y = msg.get("deltaY", 0)
            await playwright_page.mouse.wheel(0, delta_y)
            
        elif msg_type == "keypress":
            key = msg.get("key", "")
            if key:
                await playwright_page.keyboard.press(key)
                
        elif msg_type == "type":
            text = msg.get("text", "")
            if text:
                await playwright_page.keyboard.type(text)
                
        elif msg_type == "goto":
            url = msg.get("url", "")
            if url:
                await playwright_page.goto(url)
                
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


# ===== HTTP ENDPOINTS =====
@app.get("/")
async def root():
    """Serve the HTML client"""
    return FileResponse("public/index.html")


@app.get("/api/status")
async def get_status():
    """Get current browser status"""
    global playwright_page
    
    if not playwright_page:
        return {"status": "not_initialized"}
    
    try:
        url = playwright_page.url
        title = await playwright_page.title()
        
        return {
            "status": "ready",
            "url": url,
            "title": title,
            "connections": len(active_connections),
            "screencasting": is_screencasting
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/navigate")
async def navigate(url: str):
    """Navigate browser to URL"""
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        await playwright_page.goto(url)
        return {
            "success": True,
            "url": playwright_page.url,
            "title": await playwright_page.title()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== MAIN =====
if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not found - agent features won't work")
    
    # Run the server
    logger.info("🚀 Starting FastAPI CDP Server on http://localhost:3002")
    logger.info("📡 WebSocket available on ws://localhost:3002/ws")
    logger.info("🌐 Open http://localhost:3002 to see the CDP stream")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3002,
        reload=False,
        log_level="info"
    )

#!/usr/bin/env python3
"""
CDP WebSocket handler module for FastAPI
Add this to your existing FastAPI server to enable CDP browser streaming
"""

import asyncio
import json
import logging
from typing import Set, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class CDPWebSocketHandler:
    """Handles CDP WebSocket connections and streaming"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.is_screencasting = False
        self.cdp_session = None
        self.page = None
        
    def set_browser_context(self, page, cdp_session):
        """Set the browser page and CDP session"""
        self.page = page
        self.cdp_session = cdp_session
        logger.info("✅ CDP WebSocket handler initialized with browser context")
        
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"🔌 Client connected for CDP stream ({len(self.active_connections)} active)")
        
        # Start screencast if not already running
        if not self.is_screencasting and self.cdp_session:
            await self.start_screencast()
        
        # Send initial status
        await websocket.send_json({"type": "status", "connected": True})
        
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.active_connections.discard(websocket)
        logger.info(f"🔌 Client disconnected from CDP stream ({len(self.active_connections)} active)")
        
        # Stop screencast if no more connections
        if len(self.active_connections) == 0 and self.is_screencasting:
            await self.stop_screencast()
    
    async def start_screencast(self):
        """Start CDP screencast"""
        if not self.cdp_session or self.is_screencasting:
            return
        
        try:
            # Start screencast
            await self.cdp_session.send("Page.startScreencast", {
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
                for connection in self.active_connections:
                    try:
                        await connection.send_json(frame_data)
                    except:
                        disconnected.add(connection)
                
                # Remove disconnected clients
                for conn in disconnected:
                    self.active_connections.discard(conn)
                
                # Acknowledge frame
                try:
                    await self.cdp_session.send("Page.screencastFrameAck", {
                        "sessionId": params["sessionId"]
                    })
                except Exception as e:
                    logger.error(f"Failed to acknowledge frame: {e}")
            
            # Register the frame handler
            self.cdp_session.on("Page.screencastFrame", on_frame)
            
            self.is_screencasting = True
            logger.info("📹 CDP screencast started")
            
        except Exception as e:
            logger.error(f"Failed to start screencast: {e}")
    
    async def stop_screencast(self):
        """Stop CDP screencast"""
        if not self.cdp_session or not self.is_screencasting:
            return
        
        try:
            await self.cdp_session.send("Page.stopScreencast")
            self.is_screencasting = False
            logger.info("📹 CDP screencast stopped")
        except Exception as e:
            logger.error(f"Failed to stop screencast: {e}")
    
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle messages from client"""
        if not self.page:
            await websocket.send_json({"type": "error", "message": "Browser not initialized"})
            return
        
        try:
            msg = json.loads(message)
            msg_type = msg.get("type")
            
            if msg_type == "click":
                x = msg.get("x", 0)
                y = msg.get("y", 0)
                await self.page.mouse.click(x, y)
                
            elif msg_type == "move":
                x = msg.get("x", 0)
                y = msg.get("y", 0)
                await self.page.mouse.move(x, y)
                
            elif msg_type == "down":
                await self.page.mouse.down()
                
            elif msg_type == "up":
                await self.page.mouse.up()
                
            elif msg_type == "scroll":
                delta_y = msg.get("deltaY", 0)
                await self.page.mouse.wheel(0, delta_y)
                
            elif msg_type == "keypress":
                key = msg.get("key", "")
                if key:
                    await self.page.keyboard.press(key)
                    
            elif msg_type == "type":
                text = msg.get("text", "")
                if text:
                    await self.page.keyboard.type(text)
                    
            elif msg_type == "goto":
                url = msg.get("url", "")
                if url:
                    await self.page.goto(url)
                    
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)

# Create global instance
cdp_handler = CDPWebSocketHandler()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint handler for FastAPI"""
    await cdp_handler.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive_text()
            await cdp_handler.handle_message(websocket, message)
            
    except WebSocketDisconnect:
        await cdp_handler.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await cdp_handler.disconnect(websocket)

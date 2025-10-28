#!/usr/bin/env python3
"""
Healthcare Intake Assistant - FastAPI Server with Stagehand and CDP WebSocket Streaming
Enhanced version with CDP browser streaming via WebSocket
"""

import os
import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Stagehand and browser automation
from stagehand import Stagehand
from playwright.async_api import async_playwright, Page, Browser

# Google Gemini for image analysis
import google.generativeai as genai

# Import CDP WebSocket handler
from cdp_websocket_handler import cdp_handler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for browser and Stagehand
stagehand_client: Optional[Stagehand] = None
stagehand_agent: Optional[Any] = None
playwright_browser: Optional[Browser] = None
playwright_page: Optional[Page] = None
cdp_session = None  # CDP session for screencasting

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    logger.warning("GOOGLE_API_KEY not found. /analyze route will not work.")
    gemini_model = None

# Request/Response models (same as before)
class AnalyzeRequest(BaseModel):
    query: str
    url: Optional[str] = None

class AnalyzeResponse(BaseModel):
    success: bool
    screenshot_base64: str
    analysis: str
    timestamp: str
    error: Optional[str] = None

class ExecuteRequest(BaseModel):
    instruction: str

class ExecuteResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize browser and Stagehand on startup, cleanup on shutdown"""
    global stagehand_client, stagehand_agent, playwright_browser, playwright_page, cdp_session
    
    logger.info("🚀 Starting Healthcare Intake Assistant Server with CDP Streaming...")
    
    try:
        # Initialize Stagehand
        logger.info("🤖 Initializing Stagehand browser...")
        
        stagehand_client = Stagehand(
            env="LOCAL",
            headless=False,
            verbose=True
        )
        
        await stagehand_client.init()
        logger.info("✅ Stagehand initialized")
        
        # Get the Playwright page
        playwright_page = stagehand_client.page
        logger.info(f"✅ Got Playwright page: {playwright_page}")
        
        # Set viewport size
        await playwright_page.set_viewport_size({"width": 1280, "height": 720})
        logger.info("✅ Set viewport to 1280x720")
        
        # Create CDP session for screencasting
        # Stagehand's page might be a proxy, try to access the underlying Playwright page
        try:
            # Try to get the actual Playwright page if it's wrapped
            if hasattr(playwright_page, '_page'):
                actual_page = playwright_page._page
            else:
                actual_page = playwright_page
                
            # Get context from the page
            if hasattr(actual_page, 'context'):
                context = actual_page.context
            else:
                # Try to get browser context from stagehand
                context = stagehand_client.browser_context
                
            cdp_session = await context.newCDPSession(actual_page)
            logger.info("✅ CDP session created")
        except Exception as e:
            logger.error(f"⚠️ Error creating CDP session: {e}")
            logger.warning("CDP streaming will not be available, but other features will work")
            cdp_session = None
        
        # Initialize the CDP WebSocket handler with browser context
        cdp_handler.set_browser_context(playwright_page, cdp_session)
        logger.info("✅ CDP WebSocket handler initialized")
        
        # Create the Computer Use agent
        logger.info("🤖 Creating Computer Use agent...")
        try:
            stagehand_agent = stagehand_client.agent(**{
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",  # Computer Use model
                "instructions": "You are a helpful assistant that can use a web browser. When you need to search for something you have to navigate to google.com first.",
                "options": {
                    "apiKey": os.getenv("ANTHROPIC_API_KEY"),
                },
            })
            logger.info(f"✅ Agent created successfully: {type(stagehand_agent)}")
        except Exception as agent_error:
            logger.error(f"❌ Failed to create agent: {agent_error}")
            raise
        
        # Navigate to default page
        await playwright_page.goto("https://www.google.com")
        
        logger.info("✅ Browser and Stagehand initialized successfully!")
        logger.info("🌐 Browser is open and ready for commands")
        logger.info("📹 CDP WebSocket streaming available at /ws")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    # Yield control to FastAPI
    yield
    
    # Cleanup on shutdown
    logger.info("🧹 Shutting down server...")
    
    try:
        # Stop CDP screencast if running
        await cdp_handler.stop_screencast()
        
        if stagehand_client:
            await stagehand_client.close()
            logger.info("✅ Stagehand closed")
    except Exception as e:
        logger.error(f"⚠️ Error closing Stagehand: {e}")

# Create FastAPI app
app = FastAPI(
    title="Healthcare Intake Assistant API with CDP Streaming",
    description="Browser automation API with CDP WebSocket streaming",
    version="2.0.0",
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

# Mount static files for serving the CDP client
# app.mount("/cdp-client", StaticFiles(directory="../cdp-poc/public"), name="cdp-client")

# ===== WEBSOCKET ENDPOINT =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for CDP streaming"""
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

# ===== EXISTING ENDPOINTS =====
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Healthcare Intake Assistant with CDP Streaming",
        "browser_connected": playwright_page is not None,
        "stagehand_ready": stagehand_client is not None,
        "agent_ready": stagehand_agent is not None,
        "gemini_configured": gemini_model is not None,
        "cdp_streaming": len(cdp_handler.active_connections) > 0,
        "active_connections": len(cdp_handler.active_connections)
    }

@app.get("/cdp")
async def cdp_client():
    """Serve the CDP client HTML"""
    return FileResponse("../cdp-poc/public/index.html")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_screenshot(request: AnalyzeRequest):
    """Take a screenshot and analyze with Gemini"""
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    try:
        # Navigate if URL provided
        if request.url:
            logger.info(f"📍 Navigating to: {request.url}")
            await playwright_page.goto(request.url)
            await playwright_page.wait_for_load_state("networkidle", timeout=10000)
        
        # Take screenshot
        logger.info("📸 Taking screenshot...")
        screenshot_bytes = await playwright_page.screenshot(
            type="png",
            full_page=False
        )
        
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Analyze with Gemini
        logger.info(f"🔍 Analyzing screenshot with query: {request.query}")
        
        prompt = f"""
        Please analyze this screenshot and answer the following question:
        {request.query}
        
        Provide a detailed and helpful response based on what you can see in the image.
        """
        
        response = gemini_model.generate_content([
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": screenshot_base64
                }
            }
        ])
        analysis = response.text
        
        logger.info("✅ Analysis complete")
        
        return AnalyzeResponse(
            success=True,
            screenshot_base64=screenshot_base64,
            analysis=analysis,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error in analyze: {e}")
        return AnalyzeResponse(
            success=False,
            screenshot_base64="",
            analysis="",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.post("/execute", response_model=ExecuteResponse)
async def execute_agent_instruction(request: ExecuteRequest):
    """Execute instruction using Computer Use agent"""
    global stagehand_agent
    
    if not stagehand_agent:
        raise HTTPException(status_code=503, detail="Computer Use agent not initialized")
    
    try:
        logger.info(f"🤖 Executing instruction: {request.instruction}")
        
        # Notify WebSocket clients that execution is starting
        await cdp_handler.broadcast({
            "type": "execution_start",
            "instruction": request.instruction
        })
        
        # Execute the instruction
        result = await stagehand_agent.execute(request.instruction, max_steps=10)
        
        logger.info(f"✅ Instruction executed successfully")
        
        # Notify WebSocket clients that execution is complete
        await cdp_handler.broadcast({
            "type": "execution_complete",
            "instruction": request.instruction,
            "success": True
        })
        
        return ExecuteResponse(
            success=True,
            result={"action": request.instruction, "completed": True, "details": str(result)},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error executing instruction: {e}")
        
        # Notify WebSocket clients about the error
        await cdp_handler.broadcast({
            "type": "execution_error",
            "instruction": request.instruction,
            "error": str(e)
        })
        
        return ExecuteResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.get("/status")
async def get_status():
    """Get current browser status"""
    global playwright_page
    
    if not playwright_page:
        return {"status": "Browser not initialized"}
    
    try:
        current_url = playwright_page.url
        title = await playwright_page.title()
        viewport = playwright_page.viewport_size
        
        return {
            "status": "ready",
            "current_url": current_url,
            "page_title": title,
            "viewport": viewport,
            "cdp_connections": len(cdp_handler.active_connections),
            "cdp_screencasting": cdp_handler.is_screencasting,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/navigate")
async def navigate_to_url(url: str):
    """Navigate the browser to a URL"""
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        logger.info(f"🌐 Navigating to: {url}")
        await playwright_page.goto(url)
        await playwright_page.wait_for_load_state("networkidle", timeout=10000)
        
        return {
            "success": True,
            "url": playwright_page.url,
            "title": await playwright_page.title()
        }
    except Exception as e:
        logger.error(f"❌ Navigation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("❌ ANTHROPIC_API_KEY not found in environment variables!")
        logger.info("Please set it in your .env file or environment")
        exit(1)
    
    # Run the server
    logger.info("🚀 Starting FastAPI server with CDP WebSocket streaming")
    logger.info("🤖 Using Claude Computer Use model: claude-sonnet-4-20250514")
    logger.info("🔍 Using Gemini model: gemini-2.5-flash")
    logger.info("📹 CDP WebSocket streaming available at ws://localhost:8000/ws")
    logger.info("🌐 CDP client available at http://localhost:8000/cdp")
    logger.info("📚 API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

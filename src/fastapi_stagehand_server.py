#!/usr/bin/env python3
"""
Healthcare Intake Assistant - FastAPI Server with Stagehand
Provides a web API for browser automation and screenshot analysis
"""

import os
import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Stagehand and browser automation
from stagehand import Stagehand
from playwright.async_api import async_playwright, Page, Browser

# Google Gemini for image analysis
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for browser and Stagehand
stagehand_client: Optional[Stagehand] = None
stagehand_agent: Optional[Any] = None  # Global agent instance
playwright_browser: Optional[Browser] = None
playwright_page: Optional[Page] = None

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")  # Using gemini-2.5-flash as requested
else:
    logger.warning("GOOGLE_API_KEY not found. /analyze route will not work.")
    gemini_model = None

# Request/Response models
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
    global stagehand_client, stagehand_agent, playwright_browser, playwright_page
    
    logger.info("🚀 Starting Healthcare Intake Assistant Server...")
    
    try:
        # Initialize Stagehand (browser only, no provider/model configuration)
        logger.info("🤖 Initializing Stagehand browser...")
        
        stagehand_client = Stagehand(
            env="LOCAL",  # Use local browser
            headless=False,  # Show browser window
            verbose=True
        )
        
        # Initialize Stagehand (this starts Playwright and browser)
        await stagehand_client.init()
        logger.info("✅ Stagehand initialized")
        
        # Get the Playwright page from Stagehand
        playwright_page = stagehand_client.page
        logger.info(f"✅ Got Playwright page: {playwright_page}")
        
        # Set viewport size for Computer Use (consistent XY coordinates)
        await playwright_page.set_viewport_size({"width": 1280, "height": 720})
        logger.info("✅ Set viewport to 1280x720")
        
        # Create the Computer Use agent ONCE globally
        logger.info("🤖 Creating Computer Use agent...")
        try:
            # Use ** to unpack dictionary as keyword arguments
            stagehand_agent = stagehand_client.agent(**{
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "instructions": "You are a helpful assistant that can use a web browser. When you need to search for something you have to navigate to google.com first.",
                "options": {
                    "apiKey": os.getenv("ANTHROPIC_API_KEY"),
                },
            })
            logger.info(f"✅ Agent created successfully: {type(stagehand_agent)}")
        except Exception as agent_error:
            logger.error(f"❌ Failed to create agent: {agent_error}")
            logger.error(f"   Agent creation parameters type: {type(stagehand_client.agent)}")
            logger.error(f"   Stagehand client type: {type(stagehand_client)}")
            raise
        
        # Navigate to a default page
        await playwright_page.goto("https://www.google.com")
        
        logger.info("✅ Browser and Stagehand initialized successfully!")
        logger.info("🌐 Browser is open and ready for commands")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    # Yield control to FastAPI
    yield
    
    # Cleanup on shutdown
    logger.info("🧹 Shutting down server...")
    
    try:
        if stagehand_client:
            await stagehand_client.close()
            logger.info("✅ Stagehand closed")
    except Exception as e:
        logger.error(f"⚠️ Error closing Stagehand: {e}")

# Create FastAPI app
app = FastAPI(
    title="Healthcare Intake Assistant API",
    description="Browser automation API using Stagehand Computer Use with Claude and Gemini for screenshot analysis",
    version="1.1.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Healthcare Intake Assistant",
        "browser_connected": playwright_page is not None,
        "stagehand_ready": stagehand_client is not None,
        "agent_ready": stagehand_agent is not None,
        "gemini_configured": gemini_model is not None
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_screenshot(request: AnalyzeRequest):
    """
    Take a screenshot of the current page and analyze it with Gemini
    
    Args:
        request: AnalyzeRequest with query and optional URL
        
    Returns:
        AnalyzeResponse with screenshot and analysis
    """
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API not configured. Please set GOOGLE_API_KEY.")
    
    try:
        # Navigate to URL if provided
        if request.url:
            logger.info(f"📍 Navigating to: {request.url}")
            await playwright_page.goto(request.url)
            await playwright_page.wait_for_load_state("networkidle", timeout=10000)
        
        # Take screenshot
        logger.info("📸 Taking screenshot...")
        screenshot_bytes = await playwright_page.screenshot(
            type="png",
            full_page=False  # Just viewport
        )
        
        # Convert to base64 for response
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Analyze with Gemini using base64 encoded image
        logger.info(f"🔍 Analyzing screenshot with query: {request.query}")
        
        prompt = f"""
        Please analyze this screenshot and answer the following question:
        {request.query}
        
        Provide a detailed and helpful response based on what you can see in the image.
        """
        
        # Create content with inline image data as per Gemini API docs
        response = gemini_model.generate_content([
            {
                "text": prompt
            },
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": screenshot_base64  # Already base64 encoded
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
    """
    Execute an instruction using the Stagehand Computer Use agent
    
    This uses Claude's Computer Use capability to interact with the browser
    using XY coordinates and visual understanding.
    
    Args:
        request: ExecuteRequest with instruction text
        
    Returns:
        ExecuteResponse with result or error
    """
    global stagehand_agent
    
    if not stagehand_agent:
        logger.error("❌ Agent not initialized!")
        logger.error(f"   Global stagehand_agent: {stagehand_agent}")
        logger.error(f"   Global stagehand_client: {stagehand_client}")
        raise HTTPException(status_code=503, detail="Computer Use agent not initialized")
    
    try:
        logger.info(f"🤖 Executing instruction: {request.instruction}")
        logger.info(f"   Agent type: {type(stagehand_agent)}")
        
        # Execute the instruction using the global Computer Use agent
        result = await stagehand_agent.execute(request.instruction,max_steps=10)
        
        logger.info(f"✅ Instruction executed successtion sayfully")
        logger.info(f"   Result: {result}")
        
        return ExecuteResponse(
            success=True,
            result={"action": request.instruction, "completed": True, "details": str(result)},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error executing instruction: {e}")
        logger.error(f"   Error type: {type(e)}")
        logger.error(f"   Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details'}")
        return ExecuteResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.get("/status")
async def get_status():
    """Get current browser status and page information"""
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
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/navigate")
async def navigate_to_url(url: str):
    """Navigate the browser to a specific URL"""
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
    logger.info("🚀 Starting FastAPI server on http://localhost:8000")
    logger.info("🤖 Using Claude Computer Use model: claude-sonnet-4-20250514")
    logger.info("🔍 Using Gemini model: gemini-2.5-flash")
    logger.info("📚 API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )

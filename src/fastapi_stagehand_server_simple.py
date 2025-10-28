#!/usr/bin/env python3
"""
Healthcare Intake Assistant - FastAPI Server with Stagehand (Simplified)
Works with existing Stagehand without CDP streaming complications
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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Stagehand and browser automation
from stagehand import Stagehand
from playwright.async_api import async_playwright, Page, Browser

# Google Gemini for image analysis
import google.generativeai as genai

# LangChain for agent - EXACTLY as in voice_controlled_agent.py
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for browser and Stagehand
stagehand_client: Optional[Stagehand] = None
stagehand_agent: Optional[Any] = None
playwright_page: Optional[Page] = None
langchain_agent: Optional[Any] = None

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
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

class AgentRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    success: bool
    response: str
    tools_used: Optional[list] = None
    error: Optional[str] = None
    timestamp: str

class ScreenshotResponse(BaseModel):
    success: bool
    screenshot_base64: str
    url: str
    title: str
    timestamp: str

# ===== TOOL DEFINITIONS - EXACTLY AS IN voice_controlled_agent.py =====

@tool
async def execute_browser_action(instruction: str) -> str:
    """
    Execute a browser action for general web queries and navigation.
    Use this when you need to search for something on the web or navigate to websites.
    IMPORTANT: Always navigate to google.com first before searching.
    
    Args:
        instruction: The action to perform in the browser
        
    Returns:
        Result of the browser action
    """
    global stagehand_agent
    
    if not stagehand_agent:
        return "❌ Browser agent not initialized"
    
    try:
        result = await stagehand_agent.execute(instruction, max_steps=10)
        return f"✅ Successfully executed: {instruction}"
    except asyncio.TimeoutError:
        return "❌ Execution timed out (60 seconds)"
    except Exception as e:
        return f"❌ Error executing browser action: {str(e)}"

@tool
async def analyze_current_page(query: str) -> str:
    """
    Analyze the current page visible in the browser.
    Use this when the user asks questions about what's currently on the screen.
    
    Args:
        query: The question to answer about the current page
        
    Returns:
        Analysis of the current page based on the query
    """
    global playwright_page, gemini_model
    
    if not gemini_model or not playwright_page:
        return "❌ Analysis not available"
    
    try:
        # Take screenshot
        screenshot_bytes = await playwright_page.screenshot(type="png", full_page=False)
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Analyze with Gemini
        response = gemini_model.generate_content([
            {"text": f"Please analyze this screenshot and answer: {query}"},
            {"inline_data": {"mime_type": "image/png", "data": screenshot_base64}}
        ])
        
        return response.text
    except asyncio.TimeoutError:
        return "❌ Analysis timed out (30 seconds)"
    except Exception as e:
        return f"❌ Error analyzing page: {str(e)}"

# ===== LANGCHAIN AGENT SETUP - EXACTLY AS IN voice_controlled_agent.py =====

def create_voice_agent():
    """Create and configure the LangChain agent with Claude 4 Sonnet"""
    
    # Initialize Claude 4 Sonnet model
    model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # Claude 4 Sonnet model
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
        max_tokens=1000
    )
    
    # Define the tools
    tools = [execute_browser_action, analyze_current_page]
    
    # System prompt with clear instructions
    prompt = """You are a helpful voice-controlled web assistant. You have two main capabilities:

1. **execute_browser_action**: Use this for general web queries, searches, and navigation.
   - Consider navigating to google.com first before searching
   - Use for: "Search for...", "Find...", "Look up...", "Navigate to..."
   
2. **analyze_current_page**: Use this to answer questions about what's currently visible on screen.
   - Use for: "What's on this page?", "What do you see?", "Read the...", "Tell me about what's shown"

IMPORTANT RULES:
- If the user's input seems to be silence, random noise, or not a real command (like "hmm", "uh", breathing sounds), respond with "Waiting for command..." and don't take any action.
- Be concise in your responses since this is voice-controlled.
- Always confirm what action you're taking.
- Return most of the full response of the execute tool especially if the sub agent in the execute tool seems to be asking for something.
- You can always agree/reject cookies or privacy agreements and things like that. You do not need to ask. You should always add this note to the instruction of the execute_browser_action tool.

"""
    
    # Create the agent using the v1-alpha format
    agent = create_agent(
        model=model,
        tools=tools,
        prompt=prompt
    )
    
    return agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize browser and Stagehand on startup, cleanup on shutdown"""
    global stagehand_client, stagehand_agent, playwright_page, langchain_agent
    
    logger.info("🚀 Starting Healthcare Intake Assistant Server...")
    
    try:
        # Initialize Stagehand
        logger.info("🤖 Initializing Stagehand browser...")
        
        stagehand_client = Stagehand(
            env="LOCAL",
            headless=False,
            verbose=True,
            browser_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--remote-debugging-port=9222"
            ]
        )
        
        await stagehand_client.init()
        logger.info("✅ Stagehand initialized")
        
        # Get the Playwright page
        playwright_page = stagehand_client.page
        logger.info(f"✅ Got Playwright page")
        
        # Set viewport size
        await playwright_page.set_viewport_size({"width": 1280, "height": 720})
        logger.info("✅ Set viewport to 1280x720")
        
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
            logger.info(f"✅ Agent created successfully")
        except Exception as agent_error:
            logger.error(f"❌ Failed to create agent: {agent_error}")
            raise
        
        # Navigate to default page with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"📍 Navigating to Google... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(2)  # Give browser more time to fully initialize
                
                # Check if page is still valid
                if playwright_page.is_closed():
                    logger.warning("Page was closed, recreating...")
                    playwright_page = stagehand_client.page
                    await playwright_page.set_viewport_size({"width": 1280, "height": 720})
                
                await playwright_page.goto("https://www.google.com", timeout=30000)
                await playwright_page.wait_for_load_state("networkidle", timeout=10000)
                logger.info("✅ Navigated to Google successfully")
                break
                
            except Exception as nav_error:
                logger.warning(f"⚠️ Navigation attempt {attempt + 1} failed: {nav_error}")
                if attempt == max_retries - 1:
                    logger.error("❌ All navigation attempts failed")
                    logger.info("Continuing without initial navigation...")
                else:
                    await asyncio.sleep(3)  # Wait before retry
        
        # Create LangChain agent - EXACTLY AS IN voice_controlled_agent.py
        logger.info("🤖 Creating LangChain agent with Claude 4 Sonnet...")
        langchain_agent = create_voice_agent()
        logger.info("✅ Agent initialized successfully!")
        
        logger.info("✅ All components initialized successfully!")
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
    description="Browser automation API with screenshot streaming",
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

# Mount static files
# app.mount("/static", StaticFiles(directory="../cdp-poc/public"), name="static")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Healthcare Intake Assistant",
        "browser_connected": playwright_page is not None,
        "stagehand_ready": stagehand_client is not None,
        "agent_ready": stagehand_agent is not None,
        "gemini_configured": gemini_model is not None,
    }

@app.get("/viewer")
async def viewer():
    """Serve the browser viewer HTML"""
    return FileResponse("test_cdp_client.html")

@app.get("/screenshot", response_model=ScreenshotResponse)
async def get_screenshot():
    """Get current browser screenshot"""
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        # Take screenshot
        screenshot_bytes = await playwright_page.screenshot(
            type="png",
            full_page=False
        )
        
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return ScreenshotResponse(
            success=True,
            screenshot_base64=screenshot_base64,
            url=playwright_page.url,
            title=await playwright_page.title(),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error taking screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            url = request.url
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            logger.info(f"📍 Navigating to: {url}")
            await playwright_page.goto(url)
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

@app.post("/agent", response_model=AgentResponse)
async def process_with_langchain_agent(request: AgentRequest):
    """Process user message through LangChain agent - EXACTLY AS IN voice_controlled_agent.py"""
    global langchain_agent
    
    if not langchain_agent:
        raise HTTPException(status_code=503, detail="LangChain agent not initialized")
    
    try:
        logger.info(f"🤖 Agent processing: '{request.message}'")
        
        # Build messages list - EXACTLY AS IN voice_controlled_agent.py
        messages = []
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        # Invoke the agent using v1-alpha format - EXACTLY AS IN voice_controlled_agent.py
        result = await langchain_agent.ainvoke({
            "messages": messages
        })
        
        # Check if tools were called by examining the message history
        tools_used = []
        if result and "messages" in result:
            # Check all messages in the result for tool calls
            for message in result["messages"]:
                # Check for tool call indicators
                if hasattr(message, 'additional_kwargs'):
                    if message.additional_kwargs.get('tool_calls'):
                        tools_used.append("action_performed")
                        break
                # Also check message type
                if message.__class__.__name__ in ['ToolMessage', 'FunctionMessage']:
                    tools_used.append("action_performed")
                    break
                # Check content for tool execution patterns
                if hasattr(message, 'content') and isinstance(message.content, str):
                    content = message.content.lower()
                    if any(indicator in content for indicator in [
                        'executing', 'navigating', 'clicking', 'typing',
                        'screenshot', 'analyzed', 'completed', 'performed', '✅'
                    ]):
                        tools_used.append("action_performed")
                        break
        
        # Extract the response from the last message - EXACTLY AS IN voice_controlled_agent.py
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                output = last_message.content
            else:
                output = str(last_message)
        else:
            output = "No response"
        
        logger.info(f"✅ Agent response: {output}")
        if tools_used:
            logger.info(f"🔧 Tools were called")
        
        return AgentResponse(
            success=True,
            response=output,
            tools_used=tools_used if tools_used else None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing with agent: {e}")
        return AgentResponse(
            success=False,
            response=f"Error: {str(e)}",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.post("/execute", response_model=ExecuteResponse)
async def execute_agent_instruction(request: ExecuteRequest):
    """Execute instruction using Computer Use agent (direct access)"""
    global stagehand_agent
    
    if not stagehand_agent:
        raise HTTPException(status_code=503, detail="Computer Use agent not initialized")
    
    try:
        logger.info(f"🤖 Executing instruction: {request.instruction}")
        
        # Execute the instruction
        result = await stagehand_agent.execute(request.instruction, max_steps=10)
        
        logger.info(f"✅ Instruction executed successfully")
        
        return ExecuteResponse(
            success=True,
            result={"action": request.instruction, "completed": True, "details": str(result)},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error executing instruction: {e}")
        
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
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        
        # Prevent navigation to localhost to avoid recursive streaming
        if 'localhost' in url or '127.0.0.1' in url:
            logger.warning(f"⚠️ Blocked navigation to localhost: {url}")
            return {
                "success": False,
                "error": "Cannot navigate to localhost (would cause recursive streaming)",
                "url": playwright_page.url,
                "title": await playwright_page.title()
            }
            
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

@app.post("/click")
async def click_at_position(x: int, y: int):
    """Click at a specific position"""
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        await playwright_page.mouse.click(x, y)
        return {"success": True, "clicked_at": {"x": x, "y": y}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/type")
async def type_text(text: str):
    """Type text in the current focus"""
    global playwright_page
    
    if not playwright_page:
        raise HTTPException(status_code=503, detail="Browser not initialized")
    
    try:
        await playwright_page.keyboard.type(text)
        return {"success": True, "typed": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("❌ ANTHROPIC_API_KEY not found in environment variables!")
        logger.info("Please set it in your .env file or environment")
        exit(1)
    
    # Use a different port to avoid conflicts
    PORT = 8001
    
    # Run the server
    logger.info(f"🚀 Starting FastAPI server on port {PORT}")
    logger.info("🤖 Using Claude Computer Use model: claude-sonnet-4-20250514")
    logger.info("🔍 Using Gemini model: gemini-2.5-flash")
    logger.info(f"🌐 Browser viewer available at http://localhost:{PORT}/viewer")
    logger.info(f"📷 Screenshot endpoint: http://localhost:{PORT}/screenshot")
    logger.info(f"📚 API documentation available at http://localhost:{PORT}/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )

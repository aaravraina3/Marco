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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import CDP handler for WebSocket streaming
from cdp_websocket_handler import cdp_handler

# Stagehand and browser automation
from stagehand import Stagehand
from playwright.async_api import async_playwright, Page, Browser

# Google Gemini for image analysis
import google.generativeai as genai

# LangChain for agent - EXACTLY as in voice_controlled_agent.py
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

# Memory management for LangChain agent
from langgraph.checkpoint.memory import InMemorySaver
# For production, use: from langgraph.checkpoint.postgres import PostgresSaver

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
current_thread_context: dict = {}  # Store current conversation context

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
    thread_id: Optional[str] = None  # Add thread ID for conversation persistence
    clear_history: Optional[bool] = False  # Option to clear conversation history

class AgentResponse(BaseModel):
    success: bool
    response: str
    tools_used: Optional[list] = None
    error: Optional[str] = None
    timestamp: str
    thread_id: str  # Return thread ID for client to use in subsequent requests
    message_count: int  # Number of messages in the conversation

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
    global stagehand_agent, current_thread_context
    
    if not stagehand_agent:
        return "❌ Browser agent not initialized"
    
    try:
        # Extract conversation history from global context
        context_messages = []
        if current_thread_context and "messages" in current_thread_context:
            # Get the last 5 messages for context (adjust as needed)
            recent_messages = current_thread_context["messages"][-10:]  # Last 10 messages
            
            for msg in recent_messages:
                # Format each message for context
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    msg_type = msg.type
                    content = msg.content
                    
                    # Truncate very long messages
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + "..."
                    
                    # Check if it's a tool message and extract relevant info
                    if msg_type == 'tool':
                        context_messages.append(f"[Previous Action Result]: {content}")
                    elif msg_type == 'human':
                        context_messages.append(f"[User Request]: {content}")
                    elif msg_type == 'ai':
                        # Skip very long AI responses, just note action taken
                        if 'execute_browser_action' in str(content).lower():
                            context_messages.append(f"[Previous Action]: Browser action was executed")
                        elif 'analyze_current_page' in str(content).lower():
                            context_messages.append(f"[Previous Action]: Page was analyzed")
        
        # Build context section
        context_section = ""
        if context_messages:
            logger.info(f"🧠 Passing {len(context_messages)} context messages to Computer Use agent")
            for i, msg in enumerate(context_messages, 1):
                logger.info(f"  Context [{i}]: {msg[:100]}...")
            
            context_section = f"""
## CONVERSATION CONTEXT
You are continuing a multi-step conversation. Here's what has happened so far:

{chr(10).join(context_messages)}

## CURRENT INSTRUCTION
Now, execute the following instruction, keeping the above context in mind:
"""
        else:
            logger.info("🆕 No context to pass - this is the first action")
        
        base_prompt = f"""
        ======================================================================================================
        You are an elite web research and scraping agent powered by Playwright automation. Your mission is to deliver comprehensive, accurate, and deeply researched answers by efficiently navigating and extracting information from the web.
{context_section}

## Core Capabilities

You have access to:
- **Playwright browser automation** for navigation and interaction
- **Google Search** as your primary search engine
- **Screenshot functionality** for visual confirmation and debugging
- **Content extraction** from any accessible web page
- **Multi-page research** capabilities for thorough investigation

## Primary Directives

### 1. Search Strategy
- **Always start fresh**: Before each new user query, check if the Google search bar contains irrelevant content from previous searches. If so, CLEAR it completely before entering the new query.
- **Craft precise queries**: Formulate search terms that maximize relevance and minimize noise
- **Use advanced operators when beneficial**: site:, filetype:, quotes for exact phrases, minus (-) for exclusions
- **Iterative refinement**: If initial results are poor, reformulate and search again with different terms
- **Go beyond page 1**: Don't limit yourself to the first page of results if deeper research is needed

### 2. Content Extraction & Validation
- **Extract substantive content**: Focus on the actual information, not navigation, ads, or boilerplate
- **Verify quality**: Cross-reference multiple sources when claims are significant or controversial
- **Capture metadata**: Note publication dates, authors, and source credibility
- **Handle dynamic content**: Wait for JavaScript-rendered content to load fully before extraction
- **Screenshot critical findings**: Take screenshots of key information for verification and user reference

### 3. Edge Case Handling

**Anti-Bot Measures:**
- Detect CAPTCHAs, rate limiting, or access denials
- Implement random delays between requests (1-3 seconds)
- Rotate user agents if patterns suggest blocking
- If blocked, inform the user and try alternative sources

**Page Load Issues:**
- Set reasonable timeouts (30s for page load, 10s for elements)
- Retry failed requests once before moving to alternative sources
- Handle infinite scroll by scrolling incrementally and checking for new content
- Detect and report when pages fail to load or return errors

**Content Structure Variations:**
- Adapt selectors dynamically; avoid hardcoded CSS/XPath when possible
- Fall back to text-based extraction if structured extraction fails
- Handle single-page applications (SPAs) by waiting for DOM changes
- Process both static HTML and JavaScript-rendered content

**Paywalls & Access Restrictions:**
- Identify paywalled content immediately
- Search for alternative free sources or summaries
- Never attempt to bypass security measures
- Inform user when authoritative sources are behind paywalls

**Cookie Consents & Popups:**
- Auto-dismiss common cookie consent dialogs
- Close popups that block content access
- Handle newsletter signup overlays
- Continue if dismissal fails, but note reduced access

**Stale/Broken Links:**
- Detect 404s and broken pages quickly
- Use Wayback Machine for historical content if relevant
- Find alternative sources for the same information
- Report dead links to the user

**Redirect Chains:**
- Follow redirects up to 5 hops
- Detect redirect loops and abort
- Note when final destination differs significantly from initial URL

**Rate Limiting:**
- Space out requests to the same domain (minimum 2-3s)
- If rate limited, wait and retry with exponential backoff
- Switch to alternative sources if repeatedly limited

### 4. Research Depth & Quality

**Comprehensive Coverage:**
- Don't stop at the first result; compare multiple sources
- Look for primary sources, not just aggregators
- Include recent information (check dates) for time-sensitive topics
- Seek authoritative domains (.edu, .gov, established publications)

**Structured Synthesis:**
- Organize findings logically by subtopic or theme
- Distinguish between facts, opinions, and claims
- Note conflicting information and present multiple perspectives
- Provide context and background when necessary

**Source Attribution:**
- Always cite the specific URL where information was found
- Include page titles and publication dates
- Indicate source type (news, academic, forum, commercial, etc.)
- Note credibility level when relevant

### 5. Response Format

Structure your responses as:

1. **Executive Summary**: Brief overview of findings (2-3 sentences)
2. **Detailed Findings**: Organized by subtopic with citations
3. **Sources**: List of all URLs consulted with titles
4. **Methodology Notes**: Mention if you encountered issues, used alternative strategies, or have confidence caveats

### 6. Error Recovery & Communication

**When things go wrong:**
- Clearly explain what failed (e.g., "Target site is blocking automated access")
- Describe what you tried (e.g., "Attempted 3 alternative sources")
- Provide partial results if available
- Suggest manual verification steps for the user

**Proactive communication:**
- Alert users to outdated information
- Warn about low-confidence answers
- Mention when results are limited by access restrictions
- Indicate when topics require specialized databases you cannot access

### 7. Optimization & Efficiency

- **Parallel processing**: When researching multiple aspects, open multiple tabs
- **Smart caching**: Remember information within a session to avoid redundant scraping
- **Selective depth**: Deep-dive on core query elements, skim peripheral information
- **Know when to stop**: Balance thoroughness with diminishing returns

## Anti-Patterns to Avoid

❌ Using outdated search terms from previous queries  
❌ Scraping only one source for complex questions  
❌ Ignoring publication dates on time-sensitive topics  
❌ Proceeding when blocked instead of finding alternatives  
❌ Extracting navigation/UI text as if it were content  
❌ Giving up after first page load failure  
❌ Returning answers without source URLs  
❌ Claiming certainty when sources conflict  

## Quality Checklist

Before responding, verify:
- [ ] Search bar was cleared/updated for this specific query
- [ ] Multiple credible sources were consulted (minimum 3 for significant claims)
- [ ] All extracted content is substantive and relevant
- [ ] Sources are properly cited with URLs
- [ ] Publication dates were checked for time-sensitive information
- [ ] Edge cases were handled gracefully
- [ ] Response directly addresses the user's question

## Special Instructions

- **Screenshots**: Take screenshots when you encounter critical information, unusual errors, or need to verify visual content
- **Google Search Bar Management**: ALWAYS verify the search bar is relevant to the current query. Clear and re-enter if it contains terms from previous searches
- **Session Context**: You may handle multiple queries in one session. Treat each query independently unless the user explicitly references previous research
- **Dynamic Content**: Many modern sites load content via JavaScript. Always wait for key content to appear before extraction
- **Graceful Degradation**: Partial information is better than no information. If you can only access some of what's needed, provide that with caveats

Remember: Your value lies in going beyond surface-level searching to deliver researched, verified, and properly contextualized information that would take a human significant time to compile manually.
======================================================================================================
YOUR INSTRUCTIONS:


        """
        result = await stagehand_agent.execute(base_prompt + instruction, max_steps=10)
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

def pre_model_hook(state) -> dict[str, list[BaseMessage]]:
    """
    Trim messages to fit within the model's context window.
    This is called before every LLM call to prepare the messages.
    """
    original_count = len(state["messages"])
    
    # Keep the last 10,000 tokens of conversation history
    # Adjust this value based on your model's context window
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=10000,  # Adjust based on your needs
        start_on="human",
        end_on=("human", "tool"),
        # Always keep the system message
        include_system=True
    )
    
    trimmed_count = len(trimmed_messages)
    if trimmed_count < original_count:
        logger.info(f"🔄 Memory trimmed: {original_count} → {trimmed_count} messages")
    
    return {"llm_input_messages": trimmed_messages}

def create_voice_agent():
    """Create and configure the LangChain agent with Claude 4 Sonnet and memory"""
    
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
   
MAXIMUM TOOL CALLS IS A HARD LIMIT OF 2, UNLESS EXPLICITLY SPECIFIED BY USER. ADHERE TO THIS RULE.

WHEN USING TOOLS, PASS IN THE RESULTS OF PREVIOUS TOOL CALLS WHEN INVOKING NEW TOOL CALLS AS CONTEXT. THIS INCLUDES INFORMATION WE HAVE ALREADY GATHERED, WHAT WEBSITES HAVE BEEN VISITED, AND WHAT IS NEEDED,

2. **analyze_current_page**: Use this to answer questions about what's currently visible on screen.
   - Use for: "What's on this page?", "What do you see?", "Read the...", "Tell me about what's shown"

IMPORTANT RULES:
- If the user's input seems to be silence, random noise, or not a real command (like "hmm", "uh", breathing sounds), respond with "Waiting for command..." and don't take any action.
- Be concise in your responses since this is voice-controlled.
- Always confirm what action you're taking.
- Return most of the full response of the execute tool especially if the sub agent in the execute tool seems to be asking for something.
- You can always agree/reject cookies or privacy agreements and things like that. You do not need to ask. You should always add this note to the instruction of the execute_browser_action tool.
- You can always try to solve captchas. You should add that instruction too to execute_browser_action tool.
- YOU MUST PASS IN GOOD CONTEXT TO execute_browser_action tool. Especially if you have called it before. You should pass in all of your memory to it that is relevant. Especially in a multi step workflow.
"""
    
    # Create in-memory checkpointer for short-term memory
    # In production, use PostgresSaver or another persistent storage
    checkpointer = InMemorySaver()
    
    # Create the agent using the v1-alpha format with memory
    agent = create_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
        pre_model_hook=pre_model_hook  # Trim messages before each LLM call
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
            verbose=True
        )
        
        await stagehand_client.init()
        logger.info("✅ Stagehand initialized")
        
        # Get the Playwright page
        playwright_page = stagehand_client.page
        logger.info(f"✅ Got Playwright page")
        
        # Set viewport size
        await playwright_page.set_viewport_size({"width": 1280, "height": 720})
        logger.info("✅ Set viewport to 1280x720")
        
        # Create CDP session for WebSocket streaming
        cdp_session = None
        try:
            # Get the browser context from the page
            context = playwright_page.context
            # Create CDP session
            cdp_session = await context.new_cdp_session(playwright_page)
            logger.info("✅ CDP session created successfully")
            
            # Initialize CDP handler with page and session
            cdp_handler.set_browser_context(playwright_page, cdp_session)
            logger.info("✅ CDP WebSocket handler initialized")
        except Exception as e:
            logger.warning(f"⚠️ CDP session creation failed: {e}")
            logger.warning("CDP streaming will not be available, but polling will still work")
        
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
        
        # Navigate to default page
        await playwright_page.goto("https://www.google.com")
        
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
        # Stop CDP screencast if running
        await cdp_handler.stop_screencast()
        logger.info("✅ CDP screencast stopped")
        
        if stagehand_client:
            await stagehand_client.close()
            logger.info("✅ Stagehand closed")
    except Exception as e:
        logger.error(f"⚠️ Error during cleanup: {e}")

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

# Mount static files - removed since cdp-poc doesn't exist
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
        "cdp_streaming_available": cdp_handler.cdp_session is not None,
        "cdp_active_connections": len(cdp_handler.active_connections)
    }

# ===== CDP WEBSOCKET ENDPOINT =====
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

@app.get("/agent/threads/{thread_id}")
async def get_thread_info(thread_id: str):
    """Get information about a specific conversation thread"""
    global langchain_agent
    
    if not langchain_agent:
        raise HTTPException(status_code=503, detail="LangChain agent not initialized")
    
    try:
        # Get the checkpointer from the agent
        if hasattr(langchain_agent, 'checkpointer'):
            # Try to get thread state
            config = {"configurable": {"thread_id": thread_id}}
            # Note: This is a simplified version - actual implementation may vary
            return {
                "thread_id": thread_id,
                "exists": True,
                "message": f"Thread {thread_id} is available for conversation"
            }
        else:
            return {
                "thread_id": thread_id,
                "exists": False,
                "message": "Memory not configured"
            }
    except Exception as e:
        logger.error(f"Error getting thread info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agent/threads/{thread_id}")
async def clear_thread(thread_id: str):
    """Clear conversation history for a specific thread"""
    logger.info(f"🗑️ Clearing conversation history for thread: {thread_id}")
    # Since we're using InMemorySaver, creating a new thread effectively clears the old one
    # In production with a database, you would delete the thread's records
    return {
        "success": True,
        "message": f"Thread {thread_id} cleared. Next message will start a new conversation.",
        "thread_id": thread_id
    }

@app.post("/agent", response_model=AgentResponse)
async def process_with_langchain_agent(request: AgentRequest):
    """Process user message through LangChain agent with conversation memory"""
    global langchain_agent
    
    if not langchain_agent:
        raise HTTPException(status_code=503, detail="LangChain agent not initialized")
    
    try:
        # Generate thread ID if not provided
        import uuid
        thread_id = request.thread_id or str(uuid.uuid4())
        is_new_thread = request.thread_id is None
        
        logger.info("=" * 80)
        logger.info(f"🎯 NEW AGENT REQUEST")
        logger.info(f"📝 Message: '{request.message}'")
        logger.info(f"🔖 Thread ID: {thread_id}")
        logger.info(f"🆕 New Thread: {is_new_thread}")
        logger.info("=" * 80)
        
        # Configure the thread for conversation persistence
        config = {"configurable": {"thread_id": thread_id}}
        
        # Build the message
        messages = [{"role": "user", "content": request.message}]
        
        # Store the context globally so tools can access it
        # First get the current state if it exists
        global current_thread_context
        current_thread_context = {"messages": messages}
        
        # Invoke the agent with thread configuration for memory
        logger.info("⏳ Processing with LangChain agent...")
        result = await langchain_agent.ainvoke({
            "messages": messages
        }, config)
        
        # Update global context with full conversation after invocation
        if result and "messages" in result:
            current_thread_context = {"messages": result["messages"]}
        
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
        
        # Count total messages in the conversation (from result)
        message_count = len(result.get("messages", [])) if result else 0
        
        # Print the conversation memory
        logger.info("=" * 80)
        logger.info(f"📚 CONVERSATION MEMORY (Thread: {thread_id})")
        logger.info("=" * 80)
        
        if result and "messages" in result:
            for i, msg in enumerate(result["messages"], 1):
                # Get message type and content
                msg_type = msg.__class__.__name__
                role = getattr(msg, 'type', None) or msg_type
                
                # Format the content - show full content for better debugging
                if hasattr(msg, 'content'):
                    content = msg.content
                    # For AI messages with tool calls, format specially
                    if isinstance(content, list):
                        formatted_parts = []
                        for part in content:
                            if isinstance(part, dict):
                                if part.get('type') == 'text':
                                    formatted_parts.append(part.get('text', ''))
                                elif part.get('type') == 'tool_use':
                                    tool_name = part.get('name', 'unknown')
                                    tool_input = part.get('input', {})
                                    # Show the tool call details
                                    formatted_parts.append(f"[CALLING TOOL: {tool_name}]")
                                    if 'instruction' in tool_input:
                                        formatted_parts.append(f"  Instruction: {tool_input['instruction'][:200]}...")
                                    elif 'query' in tool_input:
                                        formatted_parts.append(f"  Query: {tool_input['query'][:200]}...")
                        content = "\n".join(formatted_parts)
                    # Truncate only extremely long messages
                    if isinstance(content, str) and len(content) > 1000:
                        content = content[:1000] + "... [truncated]"
                else:
                    content = str(msg)[:1000] + "... [truncated]" if len(str(msg)) > 1000 else str(msg)
                
                # Check if it's a tool message
                tool_info = ""
                if hasattr(msg, 'additional_kwargs'):
                    if msg.additional_kwargs.get('tool_calls'):
                        tool_info = " [HAS TOOL CALLS]"
                
                # Use different formatting for different message types
                if role == 'human':
                    logger.info(f"\n  👤 USER [{i}]: {content}")
                elif role == 'ai':
                    logger.info(f"\n  🤖 AI [{i}]{tool_info}: {content}")
                elif role == 'tool':
                    # Tool messages often have the result
                    logger.info(f"\n  🔧 TOOL RESULT [{i}]: {content}")
                else:
                    logger.info(f"\n  📝 {role.upper()} [{i}]{tool_info}: {content}")
        
        logger.info("=" * 80)
        logger.info(f"📊 Total messages in memory: {message_count}")
        logger.info(f"✅ Agent response: {output[:200]}..." if len(output) > 200 else f"✅ Agent response: {output}")
        if tools_used:
            logger.info(f"🔧 Tools were called: {tools_used}")
        logger.info("=" * 80)
        
        return AgentResponse(
            success=True,
            response=output,
            tools_used=tools_used if tools_used else None,
            timestamp=datetime.now().isoformat(),
            thread_id=thread_id,
            message_count=message_count
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing with agent: {e}")
        return AgentResponse(
            success=False,
            response=f"Error: {str(e)}",
            error=str(e),
            timestamp=datetime.now().isoformat(),
            thread_id=request.thread_id or "",
            message_count=0
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

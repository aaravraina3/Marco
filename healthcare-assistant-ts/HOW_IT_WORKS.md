# How the Healthcare Intake Assistant Works

## Overview

The Healthcare Intake Assistant is an AI-powered voice-controlled web automation system that helps patients navigate healthcare intake forms and web portals using natural language commands.

## System Architecture

### Three Main Components

1. **Frontend (Next.js)**: React-based UI that provides:
   - Voice input interface with real-time transcription
   - Live browser display showing current webpage
   - AI assistant response panel
   - Text-to-speech audio feedback

2. **Backend (Express.js)**: Server that handles:
   - Browser automation using Stagehand (Playwright)
   - AI agent processing with LangChain + Claude
   - Image analysis with Google Gemini
   - Real-time screenshot streaming

3. **AI Services**:
   - **Anthropic Claude**: Main AI agent that interprets commands and controls browser
   - **Google Gemini**: Analyzes screenshots to understand page content
   - **ElevenLabs**: Converts AI responses to natural-sounding speech

## How It Works

### Step-by-Step Process

#### 1. Initialization
- Click the **"🚀 Start Browser"** button to initialize the browser
- The system automatically navigates to Google.com
- Browser streaming begins, showing real-time updates

#### 2. Voice Input
- Click the microphone button (left panel)
- Speak a command naturally, for example:
  - "Search for health insurance"
  - "Navigate to patient portal"
  - "Fill out the contact form"
  - "What's on this page?"

#### 3. AI Processing
- Your voice command is converted to text
- The text is sent to the Claude AI agent
- Claude analyzes the command and decides what actions to take
- The agent has access to two main tools:
  - **execute_browser_action**: For searching, clicking, typing, navigating
  - **analyze_current_page**: For reading and understanding what's visible

#### 4. Browser Automation
- Stagehand (browser automation framework) receives instructions
- Actions are performed automatically:
  - Navigating to websites
  - Clicking buttons and links
  - Typing in form fields
  - Searching for information

#### 5. Visual Feedback
- Screenshots are captured and streamed in real-time
- You see exactly what the browser is doing
- The AI can analyze what's on screen using Google Gemini

#### 6. Audio Feedback
- The AI responds with what it's doing
- Text-to-speech converts responses to natural voice
- You hear confirmation of actions

## Example Workflows

### Workflow 1: Finding Healthcare Information
1. **User**: "Search for urgent care near me"
2. **System**: Opens Google, types the search, analyzes results
3. **User**: "Click on the first result"
4. **System**: Navigates to the clinic website
5. **User**: "Tell me their hours"
6. **System**: Analyzes the page and reads the hours aloud

### Workflow 2: Filling Out Forms
1. **User**: "Navigate to patient portal"
2. **System**: Opens the portal URL
3. **User**: "Fill out my information"
4. **System**: Asks for details needed for the form
5. **User**: Provides information verbally
6. **System**: Fills out the form automatically

### Workflow 3: Understanding Pages
1. **User**: "What's on this page?"
2. **System**: Takes a screenshot and sends it to Gemini
3. **Gemini**: Analyzes visual content, text, and structure
4. **System**: Describes the page content in natural language
5. **User**: Hears the page description read aloud

## Key Technologies

### Stagehand
- Browser automation framework built on Playwright
- Understands natural language instructions
- Performs actions like clicking, typing, navigating
- Maintains context throughout the session

### Claude AI (Anthropic)
- Large language model that understands commands
- Makes decisions about what actions to take
- Provides conversational responses
- Manages the overall workflow

### Google Gemini
- Vision AI that analyzes screenshots
- Understands visual content and text on pages
- Describes what's visible to the user
- Helps the AI understand context

### Web Speech API
- Browser-native speech recognition
- Converts voice to text in real-time
- No external services needed
- Works entirely client-side

### ElevenLabs TTS
- High-quality text-to-speech
- Natural-sounding voice responses
- Real-time audio feedback
- Configurable voice settings

## Security & Privacy

- **Local Processing**: Voice recognition happens in your browser
- **Secure API Calls**: All external API calls use HTTPS
- **No Data Storage**: No user data is permanently stored
- **Session-Based**: Each session is independent

## Troubleshooting

### Server Not Connecting
- Ensure the backend server is running on port 3003
- Check that environment variables are set
- Verify API keys are valid

### Voice Not Working
- Allow microphone permissions in browser
- Check browser console for errors
- Try typing commands instead

### Browser Not Responding
- Check server status indicator (top right)
- Click "🚀 Start Browser" to reinitialize
- Refresh the page if needed

## Use Cases

### Healthcare Intake
- Help patients navigate complex forms
- Assist users with limited mobility
- Support multiple languages
- Guide visually impaired users

### Accessible Web Browsing
- Voice-first navigation
- Automated form filling
- Page content understanding
- Task automation

### Patient Education
- Explain medical terminology
- Guide through insurance portals
- Navigate telemedicine platforms
- Complete registration forms

## Technical Details

### API Endpoints
- `GET /api/health` - Health check
- `GET /api/browser/screenshot` - Get current screenshot
- `POST /api/browser/navigate` - Navigate to URL
- `POST /api/agent/process` - Process user command with AI

### Real-Time Streaming
- Screenshots captured at 2 FPS (2 frames per second)
- Updates displayed in center panel
- Click-to-interact functionality
- Live URL display

### AI Tools
- **execute_browser_action**: Universal browser control
- **analyze_current_page**: Visual understanding
- Both tools work together seamlessly
- Context maintained across actions

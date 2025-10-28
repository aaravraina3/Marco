# Healthcare Intake Assistant - Vercel Deployment Guide

## Overview
This is a Next.js application with a TypeScript backend that provides voice-controlled web automation for healthcare intake forms. The application uses AI agents to help patients navigate web forms through voice commands.

## Architecture
- **Frontend**: Next.js 15 with React 19, Tailwind CSS
- **Backend**: Express.js server with TypeScript
- **AI Services**: Anthropic Claude, Google Gemini, ElevenLabs TTS
- **Browser Automation**: Stagehand (Playwright-based)

## Deployment Options

### Option 1: Frontend Only (Recommended for Demo)
Deploy just the Next.js frontend to Vercel. The backend services would need to run separately.

### Option 2: Full Stack (Complex)
Deploy both frontend and backend, but requires additional infrastructure for browser automation.

## Quick Deployment (Frontend Only)

### 1. Prepare Environment Variables
Create a `.env.local` file in the project root with:

```bash
# API Keys (Required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ELEVEN_LABS_API_KEY=your_eleven_labs_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Frontend Configuration
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

### 2. Deploy to Vercel

#### Method A: Vercel CLI
```bash
cd /Users/aaravraina/Documents/HackHarvard/marco/healthcare-assistant-ts
npm install -g vercel
vercel --prod
```

#### Method B: Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import from GitHub or upload the `healthcare-assistant-ts` folder
4. Configure environment variables in Vercel dashboard
5. Deploy

### 3. Configure Vercel Settings
- **Framework Preset**: Next.js
- **Root Directory**: `healthcare-assistant-ts`
- **Build Command**: `npm run build`
- **Output Directory**: `.next`

## Backend Deployment (Optional)

The backend requires additional setup for browser automation:

### Requirements
- Node.js 18+
- Playwright browsers installed
- Environment variables configured

### Backend Services
- Express server on port 3003
- WebSocket support for real-time browser streaming
- AI agent integration with LangChain

## Features
- ✅ Voice recognition and speech-to-text
- ✅ Text-to-speech with ElevenLabs
- ✅ Real-time browser automation
- ✅ AI-powered form navigation
- ✅ Screenshot analysis with Gemini
- ✅ Responsive UI with Tailwind CSS

## Demo Instructions
1. Deploy frontend to Vercel
2. Set up backend separately (or use mock data)
3. Configure API keys
4. Test voice commands and browser automation

## Troubleshooting
- Ensure all API keys are valid
- Check browser permissions for microphone access
- Verify backend connectivity if using full stack
- Check console for any build errors

## Production Considerations
- Use environment variables for all API keys
- Implement proper error handling
- Add rate limiting for API calls
- Consider using a CDN for static assets
- Set up monitoring and logging

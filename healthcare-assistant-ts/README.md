# Healthcare Intake Assistant

An AI-powered voice-controlled healthcare intake form assistant that helps patients navigate web forms through natural language commands.

## Features

- 🎤 **Voice Recognition**: Speak commands naturally to control the browser
- 🤖 **AI-Powered Navigation**: Uses Claude AI to understand and execute commands
- 👁️ **Visual Analysis**: Analyzes web pages with Google Gemini
- 🔊 **Text-to-Speech**: Provides audio feedback with ElevenLabs
- 🌐 **Real-time Browser Control**: Live browser automation with Stagehand
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites
- Node.js 18+
- API keys for Anthropic, Google, and ElevenLabs

### Installation
```bash
npm install
```

### Environment Setup
Create a `.env.local` file:
```bash
ANTHROPIC_API_KEY=your_key_here
ELEVEN_LABS_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
NEXT_PUBLIC_API_URL=http://localhost:3003
```

### Development
```bash
# Start frontend
npm run dev

# Start backend (in another terminal)
npm run dev:server

# Start both
npm run dev:all
```

### Production Build
```bash
npm run build
npm start
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

### Quick Vercel Deployment
1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel --prod`
3. Configure environment variables in Vercel dashboard

## Usage

1. **Voice Commands**: Click the microphone and speak commands like:
   - "Search for medical forms"
   - "Navigate to the patient portal"
   - "Fill out the contact information"
   - "What's on this page?"

2. **Text Commands**: Type commands in the input field

3. **Browser Control**: Watch real-time browser automation in the center panel

## Architecture

- **Frontend**: Next.js 15 with React 19, Tailwind CSS
- **Backend**: Express.js with TypeScript
- **AI**: Anthropic Claude, Google Gemini
- **TTS**: ElevenLabs
- **Automation**: Stagehand (Playwright-based)

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/browser/*` - Browser automation
- `POST /api/agent/*` - AI agent processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

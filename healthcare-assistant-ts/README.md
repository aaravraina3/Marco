# Healthcare Intake Assistant (Marco)

Voice‑controlled web assistant that can navigate, search and read pages for you. Streams a live Chromium tab into the UI, accepts speech or typed commands, and replies with TTS.

## What’s inside
- Frontend: Next.js 15 (React 19, Tailwind)
- Backend: Node/Express (TypeScript)
- Browser automation: Stagehand (Playwright)
- AI: Anthropic Claude (agent), Google Gemini (vision), ElevenLabs (TTS)

## One‑time setup
Create `healthcare-assistant-ts/.env` (server) and `healthcare-assistant-ts/.env.local` (frontend):
```
# server
ANTHROPIC_API_KEY=...
ELEVEN_LABS_API_KEY=...
GOOGLE_API_KEY=...
NODE_ENV=development

# frontend
NEXT_PUBLIC_API_URL=http://localhost:3003
```

Install deps:
```
cd healthcare-assistant-ts
npm install
npx playwright install chromium
```

## Run locally
Terminal A (backend):
```
npm run dev:server
```
Terminal B (frontend):
```
npm run dev
```
Open http://localhost:3000

Quick flow:
1) Click “Start Browser” (wait 3–5s) → 2) Click “Start Streaming” → 3) Click “Test Mic” then “Mic”, speak.

Best layout: split‑screen the Chromium window and the Assistant UI so stream refreshes don’t cover the UI.

## Deploy (Vercel + Render)

Backend (Render Web Service):
- Root Directory: `healthcare-assistant-ts`
- Runtime: Node 20
- Build: `npm install && npx playwright install --with-deps chromium && npm run build`
- Start: `npm run start:server`
- Health Check: `/api/health`
- Env: `ANTHROPIC_API_KEY`, `ELEVEN_LABS_API_KEY`, `GOOGLE_API_KEY`, `NODE_ENV=production`, `BROWSER_HEADLESS=true`

Frontend (Vercel):
- Env: `NEXT_PUBLIC_API_URL=https://<your-render-url>`
- Redeploy

Recruiter flow (Vercel):
1) Start Browser → launches headless Chromium on Render
2) Start Streaming → live view appears
3) Test Mic → Allow → pick device (optional) → Mic → speak

## Troubleshooting
- Server badge “offline”: backend not reachable; set `NEXT_PUBLIC_API_URL` correctly and redeploy.
- “Failed to initialize browser”: re‑click Start Browser; keep Chromium open.
- “Screenshot failed: Internal Server Error”: start browser, wait 3–5s, then Start Streaming.
- Mic “blocked/denied”: allow in URL bar; macOS Privacy → Microphone → enable Chrome.
- Mic “No microphone found/NotFoundError”: click Test Mic once; leave device as “Default” or pick a listed input; then Mic.

## Example prompts
- “Go to CNN and give me the latest news on tech.”
- “Navigate to Google and search for urgent care near me.”
- “What’s on this page?”
- “Click the first result.”

For a deeper guide, see [`HOW_IT_WORKS.md`](./HOW_IT_WORKS.md) and the in‑app page at `/instructions`.

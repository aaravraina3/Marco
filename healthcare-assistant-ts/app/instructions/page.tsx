'use client';

export default function InstructionsPage() {
  return (
    <main className="max-w-3xl mx-auto px-6 py-10">
      <h1 className="text-3xl font-bold mb-6">How to Use the Healthcare Intake Assistant</h1>

      <ol className="list-decimal pl-6 space-y-4">
        <li>
          <strong>Start the browser</strong>: Click the "Start Browser" button. This launches a local Chromium window and navigates to Google. Do not interact with the Chromium window manually; use the assistant.
        </li>
        <li>
          <strong>Start streaming</strong>: Click "Start Streaming" to see the Chromium tab mirrored in the center panel.
        </li>
        <li>
          <strong>Give commands</strong>: Use the microphone or the text box. Example prompts:
          <ul className="list-disc pl-6 mt-2 space-y-1">
            <li>"Go to CNN and give me the latest news on tech"</li>
            <li>"Search for urgent care near me"</li>
            <li>"What's on this page?"</li>
            <li>"Click the first result"</li>
          </ul>
        </li>
        <li>
          <strong>Let it work</strong>: The assistant will navigate, click, type, and read results. Keep your hands off the Chromium window.
        </li>
      </ol>

      <h2 className="text-xl font-semibold mt-8 mb-3">Troubleshooting</h2>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li><strong>Mic not working</strong>: Use Chrome, allow mic permission in the URL bar, and enable Microphone in macOS Privacy settings for Chrome.</li>
        <li><strong>Blank stream</strong>: Click "Start Browser" first, wait 3–5s, then "Start Streaming".</li>
        <li><strong>On Vercel</strong>: The UI is hosted, but the browser runs on a backend. Use a tunnel (ngrok) or host the backend to connect.</li>
      </ul>

      <h2 className="text-xl font-semibold mt-8 mb-3">Best Layout (Prevent Window Focus Issues)</h2>
      <p className="text-sm leading-6">
        Split-screen the windows side‑by‑side: place the local Chromium window on one half of the screen and the AI Assistant
        web app on the other half. This prevents Chromium from tabbing over the Assistant UI while streaming refreshes occur.
      </p>

      <h3 className="text-lg font-semibold mt-6 mb-2">Step‑by‑Step (Quick Start)</h3>
      <ol className="list-decimal pl-6 space-y-2 text-sm">
        <li>Open the Assistant site and the Chromium window in a split-screen layout.</li>
        <li>Click <strong>Start Browser</strong> (Chromium launches and navigates to Google).</li>
        <li>Wait 3–5 seconds, then click <strong>Start Streaming</strong> to see the live view.</li>
        <li>Click <strong>Test Mic</strong> → Allow → pick your microphone (optional), then click the big <strong>Mic</strong> to speak.</li>
        <li>Use example prompts like: “Go to CNN and give me the latest news on tech”.</li>
      </ol>

      <h2 className="text-xl font-semibold mt-8 mb-3">Status Indicators & Error Messages</h2>
      <ul className="list-disc pl-6 space-y-2 text-sm">
        <li>
          <strong>Server badge (top‑right):</strong> 
          <em>online</em> means the backend is reachable; 
          <em>checking</em> means the UI is pinging the backend; 
          <em>offline</em> means the backend URL is wrong or down.
          Fix: ensure the backend is running and <code>NEXT_PUBLIC_API_URL</code> points to it.
        </li>
        <li>
          <strong>“Failed to initialize browser”:</strong> the browser agent couldn’t start or the previous page was closed. 
          Fix: click <strong>Start Browser</strong> again (it relaunches if needed). Don’t close the Chromium window.
        </li>
        <li>
          <strong>“Screenshot failed: Internal Server Error”:</strong> streaming failed because the page is still loading or Chromium is closed/minimized. 
          Fix: ensure Chromium is open (Start Browser), wait 3–5s, then click <strong>Start Streaming</strong>.
        </li>
        <li>
          <strong>Mic: “Microphone blocked” / “permission denied”:</strong> allow mic in the URL bar; on macOS enable Chrome under Privacy → Microphone.
        </li>
        <li>
          <strong>Mic: “No microphone found” / NotFoundError:</strong> the requested device id isn’t available right now. 
          Fix: click <strong>Test Mic</strong> once to warm permissions, pick a device from the dropdown (or leave “Default”), then click the big <strong>Mic</strong>.
        </li>
        <li>
          <strong>Mic shows no bars but recognizes speech:</strong> visualization stream is optional; recognition will still work. Bars appear after <strong>Test Mic</strong> when a stream is granted.
        </li>
      </ul>
    </main>
  );
}



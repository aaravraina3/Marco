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
    </main>
  );
}



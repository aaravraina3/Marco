#!/usr/bin/env python3
"""
Simple HTTP server for the local JavaScript gaze tracker
This serves the static HTML/JS files for client-side gaze tracking
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys
from pathlib import Path

class LocalGazeTrackerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Serve index_local.html as the default page
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index_local.html'
        super().do_GET()

def main():
    port = 8080
    server_address = ('', port)
    
    print(f"🚀 Starting Local Gaze Tracker Server")
    print(f"📁 Serving from: {Path(__file__).parent}")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"👁️  Local gaze tracking (no server-side processing)")
    print(f"⚡ Fast client-side MediaPipe processing")
    print(f"\nPress Ctrl+C to stop the server")
    
    try:
        httpd = HTTPServer(server_address, LocalGazeTrackerHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
        httpd.server_close()

if __name__ == "__main__":
    main()

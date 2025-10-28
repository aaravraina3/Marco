/**
 * API client for communicating with the Python FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3003';
const HEALTH_URL = `${API_BASE_URL}/api/health`;
const BROWSER_URL = `${API_BASE_URL}/api/browser`;
const AGENT_URL = `${API_BASE_URL}/api/agent`;

export interface ScreenshotResponse {
  success: boolean;
  screenshot_base64: string;
  url: string;
  title: string;
  timestamp: string;
}

export interface ExecuteResponse {
  success: boolean;
  result?: unknown;
  error?: string;
  timestamp: string;
}

export interface StatusResponse {
  status: string;
  current_url?: string;
  page_title?: string;
  viewport?: { width: number; height: number };
  timestamp?: string;
  error?: string;
}

export interface AnalyzeResponse {
  success: boolean;
  screenshot_base64: string;
  analysis: string;
  timestamp: string;
  error?: string;
}

export interface AgentResponse {
  success: boolean;
  response: string;
  tools_used?: string[];
  timestamp: string;
  error?: string;
}

class APIClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async initBrowser(): Promise<{ success: boolean; url?: string; title?: string; error?: string }> {
    const response = await fetch(`${BROWSER_URL}/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    let data: any = null;
    try { data = await response.json(); } catch {}
    if (!response.ok) {
      return { success: false, error: data?.error || response.statusText };
    }
    return data as { success: boolean; url?: string; title?: string; error?: string };
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(HEALTH_URL);
      const data = await response.json();
      return data.status === 'ok';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async getScreenshot(): Promise<ScreenshotResponse> {
    const response = await fetch(`${BROWSER_URL}/screenshot`);
    if (!response.ok) {
      throw new Error(`Screenshot failed: ${response.statusText}`);
    }
    return response.json();
  }

  async executeCommand(instruction: string): Promise<ExecuteResponse> {
    const response = await fetch(`${BROWSER_URL}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ instruction }),
    });
    
    if (!response.ok) {
      throw new Error(`Execute failed: ${response.statusText}`);
    }
    return response.json();
  }

  async navigateTo(url: string): Promise<unknown> {
    const response = await fetch(`${BROWSER_URL}/navigate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    
    if (!response.ok) {
      throw new Error(`Navigate failed: ${response.statusText}`);
    }
    return response.json();
  }

  async clickAt(x: number, y: number): Promise<unknown> {
    const response = await fetch(`${BROWSER_URL}/click?x=${x}&y=${y}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Click failed: ${response.statusText}`);
    }
    return response.json();
  }

  async typeText(text: string): Promise<unknown> {
    const response = await fetch(`${BROWSER_URL}/type?text=${encodeURIComponent(text)}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Type failed: ${response.statusText}`);
    }
    return response.json();
  }

  async getStatus(): Promise<StatusResponse> {
    const response = await fetch(`${BROWSER_URL}/status`);
    if (!response.ok) {
      throw new Error(`Status failed: ${response.statusText}`);
    }
    return response.json();
  }

  async analyzeScreenshot(query: string, url?: string): Promise<AnalyzeResponse> {
    const response = await fetch(`${BROWSER_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, url }),
    });
    
    if (!response.ok) {
      throw new Error(`Analyze failed: ${response.statusText}`);
    }
    return response.json();
  }

  async processWithAgent(message: string): Promise<AgentResponse> {
    const response = await fetch(`${AGENT_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });
    
    if (!response.ok) {
      throw new Error(`Agent failed: ${response.statusText}`);
    }
    return response.json();
  }
}

export const apiClient = new APIClient();

/**
 * API client for communicating with the Python FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export interface ScreenshotResponse {
  success: boolean;
  screenshot_base64: string;
  url: string;
  title: string;
  timestamp: string;
}

export interface ExecuteResponse {
  success: boolean;
  result?: any;
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
  thread_id: string;
  message_count: number;
  error?: string;
}

class APIClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/`);
      const data = await response.json();
      return data.status === 'online';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async getScreenshot(): Promise<ScreenshotResponse> {
    const response = await fetch(`${this.baseUrl}/screenshot`);
    if (!response.ok) {
      throw new Error(`Screenshot failed: ${response.statusText}`);
    }
    return response.json();
  }

  async executeCommand(instruction: string): Promise<ExecuteResponse> {
    const response = await fetch(`${this.baseUrl}/execute`, {
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

  async navigateTo(url: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/navigate?url=${encodeURIComponent(url)}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Navigate failed: ${response.statusText}`);
    }
    return response.json();
  }

  async clickAt(x: number, y: number): Promise<any> {
    const response = await fetch(`${this.baseUrl}/click?x=${x}&y=${y}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Click failed: ${response.statusText}`);
    }
    return response.json();
  }

  async typeText(text: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/type?text=${encodeURIComponent(text)}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`Type failed: ${response.statusText}`);
    }
    return response.json();
  }

  async getStatus(): Promise<StatusResponse> {
    const response = await fetch(`${this.baseUrl}/status`);
    if (!response.ok) {
      throw new Error(`Status failed: ${response.statusText}`);
    }
    return response.json();
  }

  async analyzeScreenshot(query: string, url?: string): Promise<AnalyzeResponse> {
    const response = await fetch(`${this.baseUrl}/analyze`, {
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

  async processWithAgent(message: string, threadId?: string): Promise<AgentResponse> {
    const response = await fetch(`${this.baseUrl}/agent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        message,
        thread_id: threadId 
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Agent failed: ${response.statusText}`);
    }
    return response.json();
  }
  
  async clearThread(threadId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/agent/threads/${threadId}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error(`Failed to clear thread: ${response.statusText}`);
    }
    return response.json();
  }
  
  async getThreadInfo(threadId: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/agent/threads/${threadId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get thread info: ${response.statusText}`);
    }
    return response.json();
  }
}

export const apiClient = new APIClient();

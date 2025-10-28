import { ChatAnthropic } from '@langchain/anthropic';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { HumanMessage, SystemMessage, AIMessage } from '@langchain/core/messages';
import { config } from '../config';
import { stagehandService } from './stagehand.service';
import { geminiService } from './gemini.service';

// Define the tools for the agent
const executeBrowserAction = tool(
  async (input: unknown) => {
    const { instruction } = input as { instruction: string };
    try {
      const result = await stagehandService.executeAgentInstruction(instruction);
      return JSON.stringify(result);
    } catch (error) {
      return `❌ Error executing browser action: ${error}`;
    }
  },
  {
    name: 'execute_browser_action',
    description: 'Execute browser actions like searching, clicking, typing, navigating. Use for: "Search for...", "Find...", "Navigate to...", "Click on..."',
    schema: z.object({
      instruction: z.string().describe('The instruction for the browser to execute')
    })
  }
);

const analyzeCurrentPage = tool(
  async (input: unknown) => {
    const { query } = input as { query: string };
    try {
      // Take screenshot and analyze with Gemini
      const screenshot = await stagehandService.takeScreenshot();
      const analysis = await geminiService.analyzeScreenshot(screenshot, query);
      return analysis;
    } catch (error) {
      return `❌ Error analyzing page: ${error}`;
    }
  },
  {
    name: 'analyze_current_page',
    description: 'Analyze the current page visible on screen. Use for: "What\'s on this page?", "What do you see?", "Read the...", "Tell me about what\'s shown"',
    schema: z.object({
      query: z.string().describe('What to analyze or look for on the current page')
    })
  }
);

export class LangChainService {
  private model: ChatAnthropic;
  private chatHistory: (HumanMessage | SystemMessage | AIMessage)[] = [];
  
  constructor() {
    // Initialize Claude model (matching voice_controlled_agent.py line 121-126)
    this.model = new ChatAnthropic({
      model: config.claudeLangChainModel, // Claude 4 Sonnet from voice_controlled_agent.py
      apiKey: config.anthropicApiKey,
      temperature: 0.3,
      maxTokens: 1000
    });
    
    // Add system prompt
    const systemPrompt = new SystemMessage(`You are a helpful voice-controlled web assistant. You have two main capabilities:

1. **execute_browser_action**: Use this for general web queries, searches, and navigation.
   - ALWAYS navigate to google.com first before searching
   - Use for: "Search for...", "Find...", "Look up...", "Navigate to..."
   
2. **analyze_current_page**: Use this to answer questions about what's currently visible on screen.
   - Use for: "What's on this page?", "What do you see?", "Read the...", "Tell me about what's shown"

IMPORTANT RULES:
- If the user's input seems to be silence, random noise, or not a real command (like "hmm", "uh", breathing sounds), respond with "Waiting for command..." and don't take any action.
- Be concise in your responses since this is voice-controlled.
- Always confirm what action you're taking.
- Return most of the full response of the execute tool especially if the sub agent in the execute tool seems to be asking for something.`);
    
    this.chatHistory.push(systemPrompt);
  }
  
  async processMessage(message: string): Promise<{ response: string; toolsCalled: boolean }> {
    try {
      // Add user message to history
      const humanMessage = new HumanMessage(message);
      this.chatHistory.push(humanMessage);
      
      // Bind tools to the model
      const modelWithTools = this.model.bindTools([executeBrowserAction, analyzeCurrentPage]);
      
      // Get response from model
      const response = await modelWithTools.invoke(this.chatHistory);
      
      // Check if tools were called
      let toolsCalled = false;
      let finalResponse = '';
      
      if (response.tool_calls && response.tool_calls.length > 0) {
        toolsCalled = true;
        
        // Execute tool calls
        for (const toolCall of response.tool_calls) {
          if (toolCall.name === 'execute_browser_action') {
            const result = await executeBrowserAction.invoke(toolCall.args);
            finalResponse += result + '\n';
          } else if (toolCall.name === 'analyze_current_page') {
            const result = await analyzeCurrentPage.invoke(toolCall.args);
            finalResponse += result + '\n';
          }
        }
      }
      
      // Get the text response
      if (response.content) {
        finalResponse = response.content.toString() + (finalResponse ? '\n\n' + finalResponse : '');
      }
      
      // Add assistant response to history
      const aiMessage = new AIMessage(finalResponse);
      this.chatHistory.push(aiMessage);
      
      // Keep chat history manageable (last 20 messages)
      if (this.chatHistory.length > 21) {
        // Keep system message + last 20 messages
        this.chatHistory = [this.chatHistory[0], ...this.chatHistory.slice(-20)];
      }
      
      return {
        response: finalResponse,
        toolsCalled
      };
    } catch (error) {
      console.error('❌ Error processing message:', error);
      throw error;
    }
  }
  
  clearHistory(): void {
    // Keep only the system message
    this.chatHistory = [this.chatHistory[0]];
  }
  
  getHistory(): Array<{ role: string; content: string }> {
    return this.chatHistory.map(msg => ({
      role: msg._getType(),
      content: msg.content.toString()
    }));
  }
}

// Create singleton instance
export const langchainService = new LangChainService();

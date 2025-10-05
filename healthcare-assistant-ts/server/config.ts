import dotenv from 'dotenv';
import path from 'path';

// Load environment variables from root .env file
dotenv.config({ path: path.join(__dirname, '../../.env') });

export const config = {
  // API Keys
  anthropicApiKey: process.env.ANTHROPIC_API_KEY!,
  elevenLabsApiKey: process.env.ELEVEN_LABS_API_KEY!,
  googleApiKey: process.env.GOOGLE_API_KEY!,
  
  // Server Configuration
  port: process.env.PORT || 3003,
  nodeEnv: process.env.NODE_ENV || 'development',
  
  // Browser Configuration
  browserHeadless: process.env.BROWSER_HEADLESS === 'true',
  
  // Model Configuration (matching Python implementation)
  claudeLangChainModel: 'claude-sonnet-4-5-20250929', // For LangChain agent (from voice_controlled_agent.py)
  claudeComputerUseModel: 'claude-sonnet-4-20250514', // For Stagehand Computer Use (from fastapi_stagehand_server.py)
  geminiModel: 'gemini-2.5-flash',
  
  // Voice Configuration
  elevenLabsVoiceId: 'EXAVITQu4vr4xnSDxMaL', // Default voice ID
  
  // Validate required environment variables
  validate(): void {
    const required = [
      'ANTHROPIC_API_KEY',
      'ELEVEN_LABS_API_KEY', 
      'GOOGLE_API_KEY'
    ];
    
    const missing = required.filter(key => !process.env[key]);
    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
  }
};

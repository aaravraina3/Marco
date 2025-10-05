import { GoogleGenerativeAI } from '@google/generative-ai';
import { config } from '../config';

export class GeminiService {
  private genAI: GoogleGenerativeAI;
  private model: any;

  constructor() {
    this.genAI = new GoogleGenerativeAI(config.googleApiKey);
    this.model = this.genAI.getGenerativeModel({ 
      model: config.geminiModel // gemini-2.5-flash
    });
  }

  async analyzeScreenshot(screenshot: Buffer, query: string): Promise<string> {
    try {
      console.log(`🔍 Analyzing screenshot with query: ${query}`);
      
      const prompt = `
Please analyze this screenshot and answer the following question:
${query}

Provide a detailed and helpful response based on what you can see in the image.
`;

      // Convert screenshot to base64
      const base64Image = screenshot.toString('base64');
      
      // Create content with inline image data
      const result = await this.model.generateContent([
        {
          inlineData: {
            mimeType: 'image/png',
            data: base64Image
          }
        },
        { text: prompt }
      ]);
      
      const response = result.response;
      const analysis = response.text();
      
      console.log('✅ Analysis complete');
      return analysis;
    } catch (error) {
      console.error('❌ Error analyzing screenshot with Gemini:', error);
      throw error;
    }
  }
}

// Create singleton instance
export const geminiService = new GeminiService();

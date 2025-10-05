import { Stagehand } from '@browserbasehq/stagehand';
import * as playwright from 'playwright';
import { config } from '../config';

export class StagehandService {
  private stagehand: Stagehand | null = null;
  private agent: any = null;
  private cdpSession: playwright.CDPSession | null = null;
  private isInitialized = false;

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('⚠️ Stagehand already initialized');
      return;
    }

    try {
      console.log('🤖 Initializing Stagehand browser...');
      
      // Initialize Stagehand with local browser
      this.stagehand = new Stagehand({
        env: 'LOCAL',
        headless: config.browserHeadless,
        verbose: true,
        enableCaching: false
      });

      // Initialize the browser
      await this.stagehand.init();
      console.log('✅ Stagehand browser initialized');

      // Get the page from Stagehand
      const page = this.stagehand.page;
      if (!page) {
        throw new Error('Failed to get page from Stagehand');
      }

      // Set viewport size for consistency with Computer Use
      await page.setViewportSize({ width: 1280, height: 720 });
      console.log('✅ Set viewport to 1280x720');

      // Create CDP session for screencasting
      const context = page.context();
      this.cdpSession = await context.newCDPSession(page);
      console.log('✅ CDP session created');

      // Create the Computer Use agent
      console.log('🤖 Creating Computer Use agent...');
      this.agent = await this.stagehand.agent({
        provider: 'anthropic',
        model: config.claudeComputerUseModel, // Use the Computer Use specific model
        instructions: 'You are a helpful assistant that can use a web browser. When you need to search for something you have to navigate to google.com first.',
        options: {
          apiKey: config.anthropicApiKey,
        },
      });
      console.log('✅ Agent created successfully');

      // Navigate to a default page
      await page.goto('https://www.google.com');
      console.log('✅ Navigated to Google');

      this.isInitialized = true;
      console.log('🌐 Stagehand service ready!');
    } catch (error) {
      console.error('❌ Failed to initialize Stagehand:', error);
      throw error;
    }
  }

  async executeAgentInstruction(instruction: string, maxSteps: number = 10): Promise<any> {
    if (!this.agent) {
      throw new Error('Agent not initialized');
    }

    try {
      console.log(`🤖 Executing instruction: ${instruction}`);
      const result = await this.agent.execute(instruction, { max_steps: maxSteps });
      console.log('✅ Instruction executed successfully');
      return result;
    } catch (error) {
      console.error('❌ Error executing instruction:', error);
      throw error;
    }
  }

  async takeScreenshot(): Promise<Buffer> {
    if (!this.stagehand || !this.stagehand.page) {
      throw new Error('Stagehand not initialized');
    }

    const screenshot = await this.stagehand.page.screenshot({
      type: 'png',
      fullPage: false
    });
    
    return screenshot;
  }

  async navigateTo(url: string): Promise<void> {
    if (!this.stagehand || !this.stagehand.page) {
      throw new Error('Stagehand not initialized');
    }

    await this.stagehand.page.goto(url);
    await this.stagehand.page.waitForLoadState('networkidle');
  }

  async getCurrentPageInfo(): Promise<{ url: string; title: string }> {
    if (!this.stagehand || !this.stagehand.page) {
      throw new Error('Stagehand not initialized');
    }

    const page = this.stagehand.page;
    return {
      url: page.url(),
      title: await page.title()
    };
  }

  async startScreencast(
    format: 'jpeg' | 'png' = 'jpeg',
    quality: number = 60
  ): Promise<void> {
    if (!this.cdpSession) {
      throw new Error('CDP session not initialized');
    }

    await this.cdpSession.send('Page.startScreencast', {
      format,
      quality,
      maxWidth: 1280,
      maxHeight: 720,
      everyNthFrame: 1
    });
  }

  async stopScreencast(): Promise<void> {
    if (!this.cdpSession) {
      throw new Error('CDP session not initialized');
    }

    await this.cdpSession.send('Page.stopScreencast');
  }

  getCDPSession(): playwright.CDPSession | null {
    return this.cdpSession;
  }

  getPage(): playwright.Page | null {
    return this.stagehand?.page || null;
  }

  async cleanup(): Promise<void> {
    try {
      if (this.cdpSession) {
        await this.stopScreencast();
      }
      
      if (this.stagehand) {
        await this.stagehand.close();
        console.log('✅ Stagehand closed');
      }
      
      this.stagehand = null;
      this.agent = null;
      this.cdpSession = null;
      this.isInitialized = false;
    } catch (error) {
      console.error('⚠️ Error during cleanup:', error);
    }
  }
}

// Create singleton instance
export const stagehandService = new StagehandService();

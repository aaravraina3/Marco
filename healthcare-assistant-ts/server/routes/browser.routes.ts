import { Router } from 'express';
import { stagehandService } from '../services/stagehand.service';
import { geminiService } from '../services/gemini.service';

const router = Router();

// Get current browser status and page info
router.get('/status', async (req, res) => {
  try {
    const pageInfo = await stagehandService.getCurrentPageInfo();
    res.json({
      status: 'ready',
      ...pageInfo
    });
  } catch (error) {
    res.status(503).json({ 
      status: 'not_initialized',
      error: 'Browser not ready'
    });
  }
});

// Navigate to a URL
router.post('/navigate', async (req, res) => {
  const { url } = req.body;
  
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }
  
  try {
    await stagehandService.navigateTo(url);
    const pageInfo = await stagehandService.getCurrentPageInfo();
    res.json({ 
      success: true,
      ...pageInfo
    });
  } catch (error: any) {
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

// Take and analyze a screenshot
router.post('/analyze', async (req, res) => {
  const { query, url } = req.body;
  
  if (!query) {
    return res.status(400).json({ error: 'Query is required' });
  }
  
  try {
    // Navigate to URL if provided
    if (url) {
      await stagehandService.navigateTo(url);
    }
    
    // Take screenshot
    const screenshot = await stagehandService.takeScreenshot();
    const screenshotBase64 = screenshot.toString('base64');
    
    // Analyze with Gemini
    const analysis = await geminiService.analyzeScreenshot(screenshot, query);
    
    res.json({
      success: true,
      screenshot: screenshotBase64,
      analysis,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

// Execute browser action using Stagehand agent
router.post('/execute', async (req, res) => {
  const { instruction, maxSteps = 10 } = req.body;
  
  if (!instruction) {
    return res.status(400).json({ error: 'Instruction is required' });
  }
  
  try {
    const result = await stagehandService.executeAgentInstruction(instruction, maxSteps);
    res.json({
      success: true,
      result,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

export default router;

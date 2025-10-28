import { Router } from 'express';
import { stagehandService } from '../services/stagehand.service';
import { geminiService } from '../services/gemini.service';

const router = Router();

// Initialize the Stagehand browser (lazy init)
router.post('/init', async (req, res) => {
  try {
    await stagehandService.initialize();
    const pageInfo = await stagehandService.getCurrentPageInfo();
    res.json({ success: true, ...pageInfo, timestamp: new Date().toISOString() });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

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
      screenshot_base64: screenshotBase64,
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

// Additional endpoints used by the frontend streaming UI
// Get a screenshot of the current page
router.get('/screenshot', async (req, res) => {
  try {
    const screenshot = await stagehandService.takeScreenshot();
    const screenshotBase64 = screenshot.toString('base64');
    const pageInfo = await stagehandService.getCurrentPageInfo();
    res.json({
      success: true,
      screenshot_base64: screenshotBase64,
      url: pageInfo.url,
      title: pageInfo.title,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Click at coordinates on the page
router.post('/click', async (req, res) => {
  try {
    const x = Number(req.query.x ?? req.body?.x);
    const y = Number(req.query.y ?? req.body?.y);
    if (Number.isNaN(x) || Number.isNaN(y)) {
      return res.status(400).json({ success: false, error: 'x and y are required numbers' });
    }
    const page = stagehandService.getPage();
    if (!page) return res.status(503).json({ success: false, error: 'Browser not ready' });
    await page.mouse.click(x, y);
    const pageInfo = await stagehandService.getCurrentPageInfo();
    res.json({ success: true, ...pageInfo, timestamp: new Date().toISOString() });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Type text into the active element
router.post('/type', async (req, res) => {
  try {
    const text = String(req.query.text ?? req.body?.text ?? '');
    if (!text) {
      return res.status(400).json({ success: false, error: 'text is required' });
    }
    const page = stagehandService.getPage();
    if (!page) return res.status(503).json({ success: false, error: 'Browser not ready' });
    await page.keyboard.type(text);
    const pageInfo = await stagehandService.getCurrentPageInfo();
    res.json({ success: true, ...pageInfo, timestamp: new Date().toISOString() });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

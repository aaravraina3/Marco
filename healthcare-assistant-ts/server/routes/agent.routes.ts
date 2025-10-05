import { Router } from 'express';
import { langchainService } from '../services/langchain.service';

const router = Router();

// Process a message with the LangChain agent
router.post('/chat', async (req, res) => {
  const { message } = req.body;
  
  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }
  
  try {
    console.log(`💬 Processing message: ${message}`);
    const { response, toolsCalled } = await langchainService.processMessage(message);
    
    res.json({
      success: true,
      response,
      toolsCalled,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('Error processing message:', error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

// Clear chat history
router.post('/clear', async (req, res) => {
  try {
    langchainService.clearHistory();
    res.json({
      success: true,
      message: 'Chat history cleared'
    });
  } catch (error: any) {
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

// Get chat history
router.get('/history', async (req, res) => {
  try {
    const history = langchainService.getHistory();
    res.json({
      success: true,
      history,
      count: history.length
    });
  } catch (error: any) {
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

export default router;

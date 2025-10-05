import express from 'express';
import cors from 'cors';
import http from 'http';
import path from 'path';
import { config } from './config';
import { stagehandService } from './services/stagehand.service';
import { langchainService } from './services/langchain.service';
import { cdpStreamManager } from './utils/cdp-stream';
import browserRoutes from './routes/browser.routes';
import agentRoutes from './routes/agent.routes';

// Validate environment variables
config.validate();

const app = express();
const server = http.createServer(app);

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../.next')));
}

// API Routes
app.use('/api/browser', browserRoutes);
app.use('/api/agent', agentRoutes);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Healthcare Intake Assistant API',
    timestamp: new Date().toISOString()
  });
});

// Initialize CDP WebSocket streaming
cdpStreamManager.initialize(server);

// Initialize services on startup
async function initializeServices() {
  try {
    console.log('🚀 Initializing Healthcare Intake Assistant Server...');
    
    // Initialize Stagehand and browser
    await stagehandService.initialize();
    
    console.log('✅ All services initialized successfully!');
  } catch (error) {
    console.error('❌ Failed to initialize services:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🧹 Shutting down server...');
  
  try {
    await stagehandService.cleanup();
    server.close(() => {
      console.log('✅ Server closed');
      process.exit(0);
    });
  } catch (error) {
    console.error('⚠️ Error during shutdown:', error);
    process.exit(1);
  }
});

// Start server
const PORT = config.port;

server.listen(PORT, async () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📡 WebSocket available on ws://localhost:${PORT}`);
  console.log(`🌐 API endpoints:`);
  console.log(`   - Health: http://localhost:${PORT}/api/health`);
  console.log(`   - Browser: http://localhost:${PORT}/api/browser/*`);
  console.log(`   - Agent: http://localhost:${PORT}/api/agent/*`);
  console.log(`\n🤖 Model Configuration:`);
  console.log(`   - LangChain Agent: ${config.claudeLangChainModel}`);
  console.log(`   - Computer Use Agent: ${config.claudeComputerUseModel}`);
  console.log(`   - Gemini Analysis: ${config.geminiModel}`);
  
  // Initialize services after server starts
  await initializeServices();
});

export default server;

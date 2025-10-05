// Simple test script to verify agent functionality
// Run with: node test-agent.js

const BASE_URL = 'http://localhost:3003';

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testHealth() {
  console.log('🧪 Testing health endpoint...');
  try {
    const response = await fetch(`${BASE_URL}/api/health`);
    const data = await response.json();
    console.log('✅ Health check:', data);
    return true;
  } catch (error) {
    console.error('❌ Health check failed:', error.message);
    return false;
  }
}

async function testBrowserStatus() {
  console.log('\n🧪 Testing browser status...');
  try {
    const response = await fetch(`${BASE_URL}/api/browser/status`);
    const data = await response.json();
    console.log('✅ Browser status:', data);
    return true;
  } catch (error) {
    console.error('❌ Browser status failed:', error.message);
    return false;
  }
}

async function testAgentChat(message) {
  console.log(`\n🧪 Testing agent chat with message: "${message}"`);
  try {
    const response = await fetch(`${BASE_URL}/api/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    const data = await response.json();
    console.log('✅ Agent response:', data.response);
    console.log('   Tools called:', data.toolsCalled);
    return true;
  } catch (error) {
    console.error('❌ Agent chat failed:', error.message);
    return false;
  }
}

async function testBrowserExecute(instruction) {
  console.log(`\n🧪 Testing browser execute with: "${instruction}"`);
  try {
    const response = await fetch(`${BASE_URL}/api/browser/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instruction, maxSteps: 5 })
    });
    const data = await response.json();
    console.log('✅ Execution result:', data.success ? 'Success' : 'Failed');
    return true;
  } catch (error) {
    console.error('❌ Browser execute failed:', error.message);
    return false;
  }
}

async function runTests() {
  console.log('🚀 Starting Healthcare Assistant Agent Tests');
  console.log('=========================================\n');
  
  // Wait for server to be ready
  console.log('⏳ Waiting for server to initialize...');
  await sleep(3000);
  
  // Test health endpoint
  const healthOk = await testHealth();
  if (!healthOk) {
    console.error('\n❌ Server is not running! Start it with: pnpm run dev:server');
    return;
  }
  
  // Wait for browser to initialize
  await sleep(2000);
  
  // Test browser status
  await testBrowserStatus();
  
  // Test agent with simple query
  await testAgentChat("What's on the screen right now?");
  
  await sleep(2000);
  
  // Test agent with browser action
  await testAgentChat("Search for healthcare intake forms");
  
  await sleep(2000);
  
  // Test direct browser execution
  await testBrowserExecute("Navigate to google.com");
  
  console.log('\n✅ Tests completed!');
}

// Run tests
runTests().catch(console.error);

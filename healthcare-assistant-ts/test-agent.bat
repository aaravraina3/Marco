@echo off
echo ========================================
echo Healthcare Assistant Agent Test
echo ========================================
echo.

echo [1] Testing Health Endpoint...
curl -s http://localhost:3003/api/health
echo.
echo.

timeout /t 2 /nobreak > nul

echo [2] Testing Browser Status...
curl -s http://localhost:3003/api/browser/status
echo.
echo.

timeout /t 2 /nobreak > nul

echo [3] Testing Agent Chat - What's on screen...
curl -s -X POST http://localhost:3003/api/agent/chat -H "Content-Type: application/json" -d "{\"message\": \"What do you see on the screen?\"}"
echo.
echo.

timeout /t 3 /nobreak > nul

echo [4] Testing Agent Search...
curl -s -X POST http://localhost:3003/api/agent/chat -H "Content-Type: application/json" -d "{\"message\": \"Search for healthcare forms\"}"
echo.
echo.

echo ========================================
echo Tests Complete!
echo ========================================
pause

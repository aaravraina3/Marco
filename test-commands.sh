#!/bin/bash
# Quick test commands for Healthcare Assistant API

echo "🧪 Healthcare Assistant API Tests"
echo "=================================="
echo ""

echo "1️⃣ Browser Status:"
curl -s http://localhost:3003/api/browser/status | jq '.'
echo ""
echo ""

echo "2️⃣ Navigate to YouTube:"
curl -s -X POST http://localhost:3003/api/browser/navigate \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com"}' | jq '.'
echo ""
echo ""

echo "3️⃣ Chat with Agent:"
curl -s -X POST http://localhost:3003/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What website am I on?"}' | jq '.'
echo ""
echo ""

echo "4️⃣ Execute Browser Action:"
curl -s -X POST http://localhost:3003/api/browser/execute \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Scroll down the page"}' | jq '.'
echo ""

echo "✅ Tests complete!"


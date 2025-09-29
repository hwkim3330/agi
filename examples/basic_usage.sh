#!/bin/bash

# AGI Trinity - Basic Usage Examples
# Demonstrates common Trinity orchestrator usage patterns

set -euo pipefail

echo "🤖 AGI Trinity - Basic Usage Examples"
echo "====================================="

# Change to the AGI directory
cd "$(dirname "$0")/.."

# Example 1: Simple question with vote consensus
echo ""
echo "📝 Example 1: Simple Technical Question (Vote Consensus)"
echo "-------------------------------------------------------"
python3 trinity.py ask "How to implement a Redis caching layer in Python?" \
    --strategy vote \
    --agents claude,gemini,codex

# Example 2: Complex analysis with synthesis
echo ""
echo "🧠 Example 2: Complex Analysis (Synthesis Consensus)"
echo "----------------------------------------------------"
python3 trinity.py ask "Design a microservices architecture for an e-commerce platform. Include database choices, message queues, and scalability considerations." \
    --strategy synthesis \
    --agents claude,gemini,codex

# Example 3: Creative brainstorming (fanout mode)
echo ""
echo "🎨 Example 3: Creative Brainstorming (Fanout Mode)"
echo "--------------------------------------------------"
python3 trinity.py ask "Generate innovative ideas for a mobile app that helps reduce food waste" \
    --strategy fanout \
    --agents claude,gemini,codex

# Example 4: Technical debugging with specialized agents
echo ""
echo "🔧 Example 4: Technical Debugging (Specialized)"
echo "-----------------------------------------------"
python3 trinity.py ask "Debug this Python error: 'AttributeError: module has no attribute'. The code uses asyncio and imports seem correct." \
    --strategy vote \
    --agents claude,gemini

# Example 5: Market analysis (data-focused)
echo ""
echo "📊 Example 5: Market Analysis"
echo "-----------------------------"
python3 trinity.py ask "Analyze the current state of the AI/ML job market in 2024. What skills are most in demand?" \
    --strategy synthesis \
    --agents gemini,codex

echo ""
echo "✅ Basic examples completed!"
echo "💡 Pro tip: Use 'python3 trinity.py status' to check system health"
#!/bin/bash

# AGI Trinity - Revenue Generation Examples
# Demonstrates using Trinity for business and revenue generation

set -euo pipefail

echo "💰 AGI Trinity - Revenue Generation Examples"
echo "============================================"

# Change to the AGI directory
cd "$(dirname "$0")/.."

# Example 1: Market opportunity analysis
echo ""
echo "📈 Example 1: Market Opportunity Analysis"
echo "-----------------------------------------"
python3 trinity.py ask "Analyze emerging opportunities in the AI/automation consulting market for 2024-2025. What services have highest profit margins and lowest competition?" \
    --strategy synthesis \
    --agents claude,gemini,codex

# Example 2: Business model validation
echo ""
echo "🎯 Example 2: Business Model Validation"
echo "---------------------------------------"
python3 trinity.py ask "Validate this business idea: AI-powered code review service for enterprise clients. Analyze market size, competition, pricing strategy, and revenue potential." \
    --strategy synthesis \
    --agents claude,gemini,codex

# Example 3: Product development strategy
echo ""
echo "🚀 Example 3: Product Development Strategy"
echo "------------------------------------------"
python3 trinity.py ask "Design a roadmap for commercializing the AGI Trinity orchestrator. Include pricing models, target customers, feature prioritization, and go-to-market strategy." \
    --strategy synthesis \
    --agents claude,gemini,codex

# Example 4: Client proposal generation
echo ""
echo "📋 Example 4: Client Proposal Generation"
echo "----------------------------------------"
python3 trinity.py ask "Generate a consulting proposal for a fintech startup needing AI/ML implementation. Include scope, timeline, deliverables, team structure, and pricing." \
    --strategy synthesis \
    --agents claude,codex

# Example 5: Competitive analysis
echo ""
echo "🔍 Example 5: Competitive Analysis"
echo "----------------------------------"
python3 trinity.py ask "Analyze competitors in the multi-agent AI orchestration space. Identify gaps in their offerings and differentiation opportunities." \
    --strategy synthesis \
    --agents gemini,codex

# Example 6: Revenue optimization analysis
echo ""
echo "💡 Example 6: Revenue Optimization"
echo "----------------------------------"
python3 trinity.py ask "Analyze current SaaS pricing models in AI tools market. Recommend optimal pricing strategy for subscription tiers: Basic, Pro, Enterprise." \
    --strategy vote \
    --agents claude,gemini,codex

# Example 7: Client acquisition strategy
echo ""
echo "🎪 Example 7: Client Acquisition Strategy"
echo "-----------------------------------------"
python3 trinity.py ask "Develop a client acquisition strategy for AI consulting services. Include digital marketing, content strategy, partnerships, and sales funnel design." \
    --strategy synthesis \
    --agents gemini,codex

# Example 8: Service packaging
echo ""
echo "📦 Example 8: Service Packaging"
echo "-------------------------------"
python3 trinity.py ask "Design service packages for AI implementation consulting: 1) Quick Assessment (1-2 weeks), 2) Full Implementation (2-3 months), 3) Ongoing Support. Include pricing and deliverables." \
    --strategy synthesis \
    --agents claude,gemini,codex

# Example 9: Partnership opportunities
echo ""
echo "🤝 Example 9: Partnership Analysis"
echo "----------------------------------"
python3 trinity.py ask "Identify strategic partnership opportunities for an AI orchestration platform. Consider cloud providers, consulting firms, and technology vendors." \
    --strategy synthesis \
    --agents claude,gemini

# Example 10: ROI demonstration
echo ""
echo "📊 Example 10: ROI Demonstration"
echo "--------------------------------"
python3 trinity.py ask "Create a framework to demonstrate ROI for clients using AI orchestration services. Include metrics, case studies structure, and value proposition messaging." \
    --strategy synthesis \
    --agents claude,gemini,codex

echo ""
echo "✅ Revenue generation examples completed!"
echo ""
echo "🎯 Key Takeaways for Commercialization:"
echo "======================================="
echo "1. Use Trinity for client consulting and proposal generation"
echo "2. Leverage multiple AI perspectives for better market analysis"
echo "3. Create service packages around Trinity's capabilities"
echo "4. Demonstrate value through multi-agent collaboration"
echo "5. Target enterprise clients needing AI implementation"
echo ""
echo "💡 Next Steps:"
echo "- Run these examples to generate actual business insights"
echo "- Use outputs to create real client proposals"
echo "- Develop case studies from successful implementations"
echo "- Build a portfolio of Trinity-powered consulting successes"
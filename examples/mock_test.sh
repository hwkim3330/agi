#!/bin/bash

# AGI Trinity - Mock Agent Test
# Tests the Trinity system with mock CLI agents

echo "🧪 AGI Trinity - Mock Agent Test"
echo "================================"

cd "$(dirname "$0")/.."

# Create mock CLI agents for testing
mkdir -p mock_agents

# Mock Claude agent
cat > mock_agents/mock-claude << 'EOF'
#!/bin/bash
echo "Mock Claude Agent Response:"
echo "As a technical expert, I can analyze that 2 + 2 = 4."
echo "This is a basic arithmetic operation in base-10 number system."
echo "From a software perspective, this would be implemented as:"
echo "  int result = 2 + 2; // returns 4"
echo ""
echo "Technical considerations:"
echo "- Integer overflow: Not applicable for small numbers"
echo "- Precision: Exact result for integer arithmetic"
echo "- Performance: O(1) constant time operation"
EOF

# Mock Gemini agent
cat > mock_agents/mock-gemini << 'EOF'
#!/bin/bash
echo "Mock Gemini Analysis Response:"
echo "Analyzing the mathematical query '2 + 2':"
echo ""
echo "Historical Context:"
echo "- Addition has been fundamental since ancient civilizations"
echo "- The numeral system used is decimal (base-10)"
echo "- Mathematical proof: 2 + 2 = 4 (Peano axioms)"
echo ""
echo "Data Analysis:"
echo "- Input: Two integer operands (2, 2)"
echo "- Operation: Binary addition"
echo "- Output: Single integer result (4)"
echo "- Confidence: 100% (mathematical certainty)"
EOF

# Mock Codex agent
cat > mock_agents/mock-codex << 'EOF'
#!/bin/bash
echo "Mock Codex Creative Response:"
echo "Creative approaches to 2 + 2 = 4:"
echo ""
echo "1. Visual representation:"
echo "   🔵🔵 + 🔵🔵 = 🔵🔵🔵🔵"
echo ""
echo "2. Alternative implementations:"
echo "   - Recursive: add(2, 2) = succ(succ(2))"
echo "   - Functional: [2,2].reduce((a,b) => a + b)"
echo "   - Creative: 'two' + 'two' = 'four'"
echo ""
echo "3. Real-world application:"
echo "   - If you have 2 apples and get 2 more, you have 4 apples!"
echo "   - 2 cups + 2 cups = 4 cups (perfect for coffee break!)"
EOF

# Make mock agents executable
chmod +x mock_agents/*

# Create temporary config with mock agents
cat > config/mock-agents.yaml << 'EOF'
agents:
  - name: "claude"
    role: "Technical Expert"
    specialty: "Code analysis, debugging, system design"
    mode: "batch"
    cmd: ["./mock_agents/mock-claude"]
    timeout_s: 30
    personality: "Analytical, precise, technical"

  - name: "gemini"
    role: "Data Analyst"
    specialty: "Research, analysis, fact-checking"
    mode: "batch"
    cmd: ["./mock_agents/mock-gemini"]
    timeout_s: 30
    personality: "Methodical, thorough, evidence-based"

  - name: "codex"
    role: "Creative Problem Solver"
    specialty: "Innovation, creative solutions, brainstorming"
    mode: "batch"
    cmd: ["./mock_agents/mock-codex"]
    timeout_s: 30
    personality: "Innovative, creative, solution-oriented"
EOF

echo ""
echo "🎯 Testing Trinity with Mock Agents"
echo "===================================="

# Test 1: Vote consensus
echo ""
echo "Test 1: Vote Consensus"
echo "---------------------"
python3 trinity.py ask "What is 2 + 2?" \
    --strategy vote \
    --agents claude,gemini,codex \
    --config config/mock-agents.yaml

# Test 2: Synthesis consensus
echo ""
echo "Test 2: Synthesis Consensus"
echo "----------------------------"
python3 trinity.py ask "Explain the concept of recursion in programming" \
    --strategy synthesis \
    --agents claude,gemini,codex \
    --config config/mock-agents.yaml

# Test 3: Fanout mode
echo ""
echo "Test 3: Fanout Mode"
echo "-------------------"
python3 trinity.py ask "Generate creative ideas for a productivity app" \
    --strategy fanout \
    --agents claude,gemini,codex \
    --config config/mock-agents.yaml

echo ""
echo "✅ Mock agent testing completed!"
echo ""
echo "🎯 Results Summary:"
echo "=================="
python3 trinity.py status

# Cleanup
echo ""
echo "🧹 Cleaning up mock agents..."
rm -rf mock_agents/
rm -f config/mock-agents.yaml

echo "✨ Trinity system test completed successfully!"
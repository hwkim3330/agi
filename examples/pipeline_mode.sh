#!/bin/bash

# AGI Trinity - Pipeline Mode Examples
# Demonstrates observation bus and continuous monitoring

set -euo pipefail

echo "🔄 AGI Trinity - Pipeline Mode Examples"
echo "======================================="

# Change to the AGI directory
cd "$(dirname "$0")/.."

# Example 1: Monitor system logs and get AI insights
echo ""
echo "📊 Example 1: System Log Analysis Pipeline"
echo "------------------------------------------"

# Start background monitoring (simulate system logs)
echo "Starting system log simulation..."
(
    for i in {1..10}; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Application startup sequence $i/10"
        sleep 2
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: High memory usage detected: 85%"
        sleep 1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Database connection timeout"
        sleep 3
    done
) | tee >(python3 trinity.py observe --agent claude) \
        >(python3 trinity.py observe --agent gemini) \
  | python3 trinity.py observe --agent codex

# Wait for observations to be collected
sleep 5

# Synthesize the observations
echo ""
echo "🧠 Synthesizing system log observations..."
python3 trinity.py synthesize --strategy synthesis --agents claude,gemini,codex

# Example 2: Code review pipeline
echo ""
echo "🔍 Example 2: Code Review Pipeline"
echo "----------------------------------"

# Simulate a git diff output
cat << 'EOF' | python3 trinity.py observe --agent claude
--- a/src/app.py
+++ b/src/app.py
@@ -1,10 +1,15 @@
 def process_user_data(user_input):
-    # No validation
-    return user_input.upper()
+    # Add input validation
+    if not isinstance(user_input, str):
+        raise ValueError("Input must be a string")
+
+    if len(user_input) > 1000:
+        raise ValueError("Input too long")
+
+    return user_input.strip().upper()

 def main():
-    data = process_user_data(input("Enter data: "))
+    try:
+        data = process_user_data(input("Enter data: "))
+        print(f"Processed: {data}")
+    except ValueError as e:
+        print(f"Error: {e}")
EOF

# Get code review from agents
echo ""
echo "👥 Getting code review from Trinity..."
python3 trinity.py ask "Review the code changes in the observation buffer. Focus on security, performance, and best practices." \
    --strategy synthesis \
    --agents claude,gemini

# Example 3: Performance monitoring pipeline
echo ""
echo "⚡ Example 3: Performance Monitoring Pipeline"
echo "--------------------------------------------"

# Simulate performance metrics
(
    echo "=== System Performance Metrics ==="
    echo "CPU Usage: 78%"
    echo "Memory Usage: 65%"
    echo "Disk I/O: 145 MB/s read, 89 MB/s write"
    echo "Network: 234 Mbps in, 156 Mbps out"
    echo "Active Connections: 1,247"
    echo "Database Queries/sec: 89"
    echo "Response Time P50: 245ms"
    echo "Response Time P95: 1.2s"
    echo "Error Rate: 0.3%"
) | tee >(python3 trinity.py observe --agent claude --max-lines 1000) \
        >(python3 trinity.py observe --agent gemini --max-lines 1000)

# Wait and synthesize
sleep 3
echo ""
echo "📈 Analyzing performance metrics..."
python3 trinity.py synthesize --strategy vote --agents claude,gemini --context-lines 50

# Example 4: Continuous integration pipeline
echo ""
echo "🚀 Example 4: CI/CD Pipeline Analysis"
echo "-------------------------------------"

# Simulate CI/CD pipeline logs
cat << 'EOF' > /tmp/ci_logs.txt
=== Build Pipeline Results ===
✅ Lint: PASSED (0 issues)
✅ Unit Tests: PASSED (127/127)
⚠️  Integration Tests: PASSED (45/47) - 2 flaky tests
❌ Security Scan: FAILED (3 high-severity vulnerabilities)
✅ Build: PASSED (2m 34s)
❌ Deploy: FAILED (dependency conflict)

Vulnerabilities Found:
- SQL Injection risk in user input validation
- Exposed API keys in environment variables
- Outdated dependencies with known CVEs
EOF

# Feed to all agents
cat /tmp/ci_logs.txt | tee >(python3 trinity.py observe --agent claude) \
                           >(python3 trinity.py observe --agent gemini) \
                     | python3 trinity.py observe --agent codex

# Get recommendations
sleep 2
echo ""
echo "🛠️  Getting remediation recommendations..."
python3 trinity.py ask "Based on the CI/CD pipeline results in observations, provide specific action items to fix the failures and improve the pipeline." \
    --strategy synthesis \
    --agents claude,gemini,codex

# Cleanup
rm -f /tmp/ci_logs.txt

echo ""
echo "✅ Pipeline mode examples completed!"
echo "💡 Pro tip: Use multiple terminals to run real-time monitoring"
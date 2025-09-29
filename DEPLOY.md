# 🚀 AGI Trinity - Deployment Guide

## Quick Start Deployment

### Prerequisites
```bash
# Python 3.8+ required
python3 --version

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install python3-pip python3-venv expect

# Or install using pip
pip3 install --break-system-packages typer rich pyyaml
```

### Basic Setup

1. **Clone and Setup**:
```bash
git clone https://github.com/your-username/agi.git
cd agi
pip3 install -r requirements.txt
```

2. **Configure Your AI CLI Tools**:
```bash
# Edit agent configuration
cp config/agents.yaml.example config/agents.yaml
nano config/agents.yaml
```

3. **Test Installation**:
```bash
python3 trinity.py status
examples/mock_test.sh
```

## Production Deployment

### 1. Environment Setup

**Option A: Virtual Environment (Recommended)**
```bash
python3 -m venv trinity-env
source trinity-env/bin/activate
pip install -r requirements.txt
```

**Option B: Docker Container**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python3", "trinity.py", "ask", "--help"]
```

**Option C: System-wide Installation**
```bash
sudo pip3 install -r requirements.txt
sudo chmod +x trinity.py scripts/*
sudo ln -s $(pwd)/trinity.py /usr/local/bin/trinity
```

### 2. CLI Tool Configuration

**Claude Code Setup**:
```bash
# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | sh
echo 'export PATH="$HOME/.claude/bin:$PATH"' >> ~/.bashrc

# Test access
claude-code --version
```

**Gemini CLI Setup**:
```bash
# Option 1: Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Option 2: Direct Gemini CLI (if available)
pip3 install google-generativeai
export GOOGLE_API_KEY="your-api-key"
```

**OpenAI CLI Setup**:
```bash
# Install OpenAI CLI
pip3 install openai
export OPENAI_API_KEY="your-api-key"

# Test access
openai --version
```

### 3. Agent Configuration

Update `config/agents.yaml` with your specific CLI commands:

```yaml
agents:
  - name: "claude"
    cmd: ["claude-code", "--dangerously-skip-permissions"]
    # Add your Claude Code configuration

  - name: "gemini"
    cmd: ["gcloud", "ai", "models", "generate-text", "--model=gemini-pro", "--prompt={PROMPT}"]
    # Add your Gemini configuration

  - name: "codex"
    cmd: ["openai", "chat", "completions", "create", "-m", "gpt-4", "-u", "user", "{PROMPT}"]
    # Add your OpenAI configuration
```

## Service Deployment

### systemd Service (Linux)

Create `/etc/systemd/system/trinity.service`:
```ini
[Unit]
Description=AGI Trinity Multi-Agent Orchestrator
After=network.target

[Service]
Type=simple
User=trinity
Group=trinity
WorkingDirectory=/opt/trinity
Environment=PATH=/opt/trinity/venv/bin:/usr/bin:/bin
ExecStart=/opt/trinity/venv/bin/python trinity.py ask --help
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trinity
sudo systemctl start trinity
sudo systemctl status trinity
```

### Web API Deployment

For web API access, create `api_server.py`:

```python
#!/usr/bin/env python3
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json

app = FastAPI(title="AGI Trinity API")

class QueryRequest(BaseModel):
    prompt: str
    strategy: str = "synthesis"
    agents: str = "claude,gemini,codex"

@app.post("/ask")
async def ask_trinity(request: QueryRequest):
    try:
        result = subprocess.run([
            "python3", "trinity.py", "ask", request.prompt,
            "--strategy", request.strategy,
            "--agents", request.agents,
            "--no-save-session"
        ], capture_output=True, text=True, timeout=300)

        return {"success": True, "response": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run API server:
```bash
python3 api_server.py
# Or with gunicorn for production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api_server:app
```

## Monitoring & Logging

### Log Configuration
```bash
# Create log directory
mkdir -p /var/log/trinity

# Configure logrotate
cat > /etc/logrotate.d/trinity << 'EOF'
/var/log/trinity/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 0644 trinity trinity
}
EOF
```

### Health Checks
```bash
#!/bin/bash
# health_check.sh
python3 trinity.py status > /var/log/trinity/health.log 2>&1
if [ $? -eq 0 ]; then
    echo "$(date): Trinity healthy" >> /var/log/trinity/health.log
else
    echo "$(date): Trinity unhealthy" >> /var/log/trinity/health.log
    # Send alert notification
fi
```

### Performance Monitoring
```bash
# Add to crontab for regular monitoring
*/5 * * * * /opt/trinity/scripts/health_check.sh
0 */6 * * * /opt/trinity/scripts/performance_report.sh
```

## Security Considerations

### 1. API Keys and Credentials
```bash
# Store secrets in environment files
cat > /etc/trinity/env << 'EOF'
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
EOF

# Secure permissions
chmod 600 /etc/trinity/env
chown trinity:trinity /etc/trinity/env
```

### 2. Network Security
```bash
# Firewall rules (if running web API)
sudo ufw allow 8000/tcp
sudo ufw enable

# Or restrict to specific IPs
sudo ufw allow from 192.168.1.0/24 to any port 8000
```

### 3. Process Isolation
```bash
# Create dedicated user
sudo useradd -r -s /bin/false trinity
sudo mkdir -p /home/trinity/.trinity
sudo chown -R trinity:trinity /home/trinity
```

## Scaling and Performance

### Horizontal Scaling
```yaml
# Load balancer configuration (nginx)
upstream trinity_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://trinity_backend;
    }
}
```

### Performance Tuning
```bash
# Increase file descriptor limits
echo "trinity soft nofile 65536" >> /etc/security/limits.conf
echo "trinity hard nofile 65536" >> /etc/security/limits.conf

# Optimize Python performance
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

## Troubleshooting

### Common Issues

**Issue 1: Missing Dependencies**
```bash
# Install missing system packages
sudo apt install python3-dev build-essential

# Reinstall Python packages
pip3 install --force-reinstall -r requirements.txt
```

**Issue 2: Permission Denied**
```bash
# Fix script permissions
chmod +x scripts/* trinity.py
chown -R $USER:$USER ~/.trinity/
```

**Issue 3: Agent Timeouts**
```bash
# Increase timeout in config
# config/agents.yaml
timeout_s: 300  # Increase from default 180s
```

**Issue 4: Memory Issues**
```bash
# Monitor memory usage
ps aux | grep trinity
free -h

# Add swap if needed
sudo fallocate -l 2G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Debug Mode
```bash
# Run in debug mode
python3 trinity.py --verbose ask "test prompt"

# Check logs
tail -f ~/.trinity/sessions/trinity_*.json
cat /var/log/trinity/*.log
```

## Backup and Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/trinity/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuration and sessions
cp -r config/ $BACKUP_DIR/
cp -r ~/.trinity/ $BACKUP_DIR/
cp trinity.py $BACKUP_DIR/

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR/
rm -rf $BACKUP_DIR
```

### Recovery Process
```bash
# Restore from backup
tar -xzf /backup/trinity/20240929.tar.gz
cp -r 20240929/config/ ./
cp -r 20240929/.trinity/ ~/
```

## Success Metrics

Track these KPIs for your Trinity deployment:
- **Response Time**: Average time per query
- **Success Rate**: Percentage of successful agent responses
- **Agent Utilization**: Usage distribution across agents
- **Error Rate**: Failed requests per hour
- **Consensus Quality**: User satisfaction with synthesized results

---

For additional support, see the [README.md](README.md) and example scripts in the `examples/` directory.
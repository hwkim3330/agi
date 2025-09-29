# AGI Trinity - Agent Adapters

This directory contains adapter implementations for integrating different AI CLI tools into the Trinity orchestrator.

## Agent Adapter Structure

Each agent adapter should implement the following interface:
- Handle agent-specific command formatting
- Manage authentication and connection
- Parse responses into standardized format
- Implement retry logic and error handling

## Available Adapters

### Claude Code Adapter
- **File**: `claude_adapter.py`
- **Purpose**: Interface with Claude Code CLI
- **Features**:
  - Permission bypass automation
  - REPL session management
  - Response parsing and formatting

### Gemini Pro Adapter
- **File**: `gemini_adapter.py`
- **Purpose**: Interface with Gemini CLI tools
- **Features**:
  - Multiple CLI tool support (gcloud, gemini)
  - API key management
  - Response validation

### Codex/OpenAI Adapter
- **File**: `codex_adapter.py`
- **Purpose**: Interface with OpenAI CLI tools
- **Features**:
  - Multi-model support (GPT-4, Codex)
  - Token management
  - Rate limiting handling

## Usage

Agent adapters are automatically loaded by the Trinity orchestrator based on the configuration in `config/agents.yaml`. Each adapter extends the base `AgentAdapter` class and implements specific methods for their respective AI service.

## Adding New Agents

To add a new agent adapter:

1. Create a new Python file in this directory
2. Implement the `AgentAdapter` base class
3. Add configuration entry in `agents.yaml`
4. Test with the Trinity orchestrator

Example adapter structure:
```python
class CustomAgentAdapter(AgentAdapter):
    def __init__(self, config):
        super().__init__(config)

    async def execute(self, prompt):
        # Implement agent-specific execution
        pass

    def parse_response(self, raw_response):
        # Parse and format response
        pass
```
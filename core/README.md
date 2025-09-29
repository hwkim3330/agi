# AGI Trinity - Core Orchestration Logic

This directory contains the core orchestration components that handle multi-agent coordination, consensus mechanisms, and system monitoring.

## Core Components

### Consensus Engine (`consensus.py`)
- **Purpose**: Implements voting and synthesis algorithms
- **Features**:
  - Vote-based consensus with weighted scoring
  - Synthesis-based response combination
  - Confidence scoring and reasoning
  - Fallback mechanisms for failed agents

### Request Router (`router.py`)
- **Purpose**: Handles request distribution and load balancing
- **Features**:
  - Agent selection based on specialization
  - Load balancing across available agents
  - Request queuing and prioritization
  - Health-based routing decisions

### Real-time Monitor (`monitor.py`)
- **Purpose**: System health and performance monitoring
- **Features**:
  - Agent performance tracking
  - Resource usage monitoring
  - Real-time status updates
  - Alert and notification system

## Architecture

```
┌─────────────────┐
│   Trinity CLI   │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │  Router   │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Consensus │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Monitor   │
    └───────────┘
```

## Usage

Core components are automatically instantiated by the Trinity orchestrator. They work together to provide:

1. **Request Processing**: Router determines which agents to use
2. **Parallel Execution**: Monitor tracks agent performance
3. **Response Synthesis**: Consensus engine combines results
4. **Quality Assurance**: Monitor validates output quality

## Configuration

Core components are configured through the main `config/agents.yaml` file:

```yaml
global:
  max_concurrent: 3
  default_strategy: "synthesis"

consensus:
  vote:
    content_length_weight: 0.3
    success_rate_weight: 0.4

monitoring:
  health_check_interval: 30
  performance_tracking: true
```

## Extension Points

- **Custom Consensus Algorithms**: Implement new voting/synthesis strategies
- **Advanced Routing**: Add ML-based agent selection
- **Enhanced Monitoring**: Integrate with external monitoring systems
- **Plugin Architecture**: Add custom orchestration logic
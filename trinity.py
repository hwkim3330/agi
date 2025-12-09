#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Trinity - Multi-Agent AI Orchestrator
Next-generation AI collaboration framework

"Three minds, one consciousness"
"""
import os
import sys
import time
import json
import yaml
import asyncio
import pathlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

app = typer.Typer(
    help="🤖 AGI Trinity - Multi-Agent AI Orchestrator",
    rich_markup_mode="rich"
)
console = Console()

# Global state
HOME = pathlib.Path(os.path.expanduser("~"))
TRINITY_STATE = HOME / ".trinity"
OBSERVATION_BUFFERS = TRINITY_STATE / "observations"
SESSION_LOGS = TRINITY_STATE / "sessions"

# Ensure directories exist
TRINITY_STATE.mkdir(parents=True, exist_ok=True)
OBSERVATION_BUFFERS.mkdir(parents=True, exist_ok=True)
SESSION_LOGS.mkdir(parents=True, exist_ok=True)

@dataclass
class AgentResponse:
    """Response from an individual agent"""
    agent: str
    success: bool
    content: str
    latency: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ConsensusResult:
    """Result of consensus mechanism"""
    strategy: str
    individual_responses: List[AgentResponse]
    consensus_content: str
    confidence: float
    reasoning: str

class AgentManager:
    """Manages individual AI agents"""

    def __init__(self, config_path: str = "config/agents.yaml"):
        self.config_path = config_path
        self.agents = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load agent configurations"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[red]Config file not found: {self.config_path}[/red]")
            console.print("Creating default configuration...")
            self._create_default_config()
            return self._load_config()

    def _create_default_config(self):
        """Create default agent configuration"""
        default_config = {
            "agents": [
                {
                    "name": "claude",
                    "role": "Technical Expert",
                    "specialty": "Code analysis, debugging, system design",
                    "mode": "expect",
                    "cmd": ["claude-code", "--dangerously-skip-permissions"],
                    "end_pattern": "\\n> $",
                    "send_suffix": "\\n",
                    "startup_wait_ms": 500,
                    "timeout_s": 180,
                    "personality": "Analytical, precise, technical"
                },
                {
                    "name": "gemini",
                    "role": "Data Analyst",
                    "specialty": "Research, analysis, fact-checking",
                    "mode": "batch",
                    "cmd": ["gemini", "generate", "{PROMPT}"],
                    "timeout_s": 180,
                    "personality": "Methodical, thorough, evidence-based"
                },
                {
                    "name": "codex",
                    "role": "Creative Problem Solver",
                    "specialty": "Innovation, creative solutions, brainstorming",
                    "mode": "batch",
                    "cmd": ["openai", "chat", "completions", "create", "-m", "gpt-4", "-u", "user", "{PROMPT}"],
                    "timeout_s": 180,
                    "personality": "Innovative, creative, solution-oriented"
                }
            ]
        }

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

    async def execute_agent(self, agent_name: str, prompt: str) -> AgentResponse:
        """Execute a single agent with the given prompt"""
        agent_config = next((a for a in self.agents["agents"] if a["name"] == agent_name), None)
        if not agent_config:
            raise ValueError(f"Agent {agent_name} not found")

        start_time = time.time()

        try:
            if agent_config["mode"] == "batch":
                success, content, error = await self._run_batch_agent(agent_config, prompt)
            else:
                success, content, error = await self._run_expect_agent(agent_config, prompt)

            latency = time.time() - start_time

            return AgentResponse(
                agent=agent_name,
                success=success,
                content=content if success else error,
                latency=latency,
                metadata={"mode": agent_config["mode"], "role": agent_config["role"]},
                timestamp=datetime.now()
            )

        except Exception as e:
            latency = time.time() - start_time
            return AgentResponse(
                agent=agent_name,
                success=False,
                content=f"Error: {str(e)}",
                latency=latency,
                metadata={"error": True},
                timestamp=datetime.now()
            )

    async def _run_batch_agent(self, agent_config: Dict, prompt: str) -> tuple:
        """Run batch-mode agent"""
        cmd = agent_config["cmd"].copy()

        # Replace {PROMPT} tokens
        cmd = [part.replace("{PROMPT}", prompt) for part in cmd]

        # Add non-interactive wrapper
        wrapped_cmd = ["./scripts/nonint.sh"] + cmd

        try:
            process = await asyncio.create_subprocess_exec(
                *wrapped_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode() if "{PROMPT}" not in " ".join(cmd) else None),
                timeout=agent_config.get("timeout_s", 180)
            )

            success = process.returncode == 0
            content = stdout.decode(errors="ignore").strip()
            error = stderr.decode(errors="ignore").strip()

            return success, content, error

        except asyncio.TimeoutError:
            return False, "", "Timeout exceeded"
        except Exception as e:
            return False, "", str(e)

    async def _run_expect_agent(self, agent_config: Dict, prompt: str) -> tuple:
        """Run expect-mode agent (REPL/TUI)"""
        # This would use the agent_wrap.expect script
        # For now, simplified implementation
        return await self._run_batch_agent(agent_config, prompt)

class ConsensusEngine:
    """Handles consensus and synthesis of multiple agent responses"""

    @staticmethod
    def vote_consensus(responses: List[AgentResponse]) -> ConsensusResult:
        """Simple voting-based consensus"""
        successful_responses = [r for r in responses if r.success]

        if not successful_responses:
            return ConsensusResult(
                strategy="vote",
                individual_responses=responses,
                consensus_content="All agents failed to respond",
                confidence=0.0,
                reasoning="No successful responses to synthesize"
            )

        # Simple implementation: longest response wins
        # In practice, this would use more sophisticated NLP techniques
        best_response = max(successful_responses, key=lambda r: len(r.content))

        confidence = len(successful_responses) / len(responses)

        return ConsensusResult(
            strategy="vote",
            individual_responses=responses,
            consensus_content=best_response.content,
            confidence=confidence,
            reasoning=f"Selected response from {best_response.agent} (highest content richness)"
        )

    @staticmethod
    def synthesis_consensus(responses: List[AgentResponse]) -> ConsensusResult:
        """Synthesis-based consensus (combines insights from all agents)"""
        successful_responses = [r for r in responses if r.success]

        if not successful_responses:
            return ConsensusResult(
                strategy="synthesis",
                individual_responses=responses,
                consensus_content="All agents failed to respond",
                confidence=0.0,
                reasoning="No successful responses to synthesize"
            )

        # Create a synthesis prompt
        synthesis_sections = []
        for response in successful_responses:
            role = response.metadata.get("role", "Agent")
            synthesis_sections.append(f"**{role} ({response.agent}):**\\n{response.content}")

        synthesis_content = "\\n\\n".join(synthesis_sections)
        confidence = len(successful_responses) / len(responses)

        return ConsensusResult(
            strategy="synthesis",
            individual_responses=responses,
            consensus_content=synthesis_content,
            confidence=confidence,
            reasoning=f"Synthesized insights from {len(successful_responses)} agents"
        )

# CLI Commands

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The question or task to ask all agents"),
    strategy: str = typer.Option("vote", help="Consensus strategy: vote, synthesis, fanout"),
    agents: str = typer.Option("claude,gemini,codex", help="Comma-separated list of agents to use"),
    config: str = typer.Option("config/agents.yaml", help="Path to agent configuration file"),
    save_session: bool = typer.Option(True, help="Save session to logs"),
):
    """🤖 Ask a question to the AGI Trinity collective"""

    console.print(Panel.fit(
        f"[bold blue]AGI Trinity Orchestrator[/bold blue]\\n"
        f"[dim]Strategy: {strategy}[/dim]\\n"
        f"[dim]Agents: {agents}[/dim]",
        title="🤖 Initializing"
    ))

    agent_names = [name.strip() for name in agents.split(",")]
    manager = AgentManager(config)
    consensus_engine = ConsensusEngine()

    async def execute_trinity():
        # Execute all agents in parallel
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:

            # Create tasks for each agent
            tasks = []
            progress_tasks = []

            for agent_name in agent_names:
                task = asyncio.create_task(manager.execute_agent(agent_name, prompt))
                progress_task = progress.add_task(f"[cyan]{agent_name}[/cyan]", total=100)
                tasks.append(task)
                progress_tasks.append(progress_task)

            # Wait for all agents to complete
            responses = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                response = await task
                responses.append(response)
                progress.update(progress_tasks[i], completed=100)

        return responses

    # Run the trinity
    responses = asyncio.run(execute_trinity())

    # Display individual responses
    console.print("\\n[bold]Individual Agent Responses:[/bold]")
    for response in responses:
        status = "✅" if response.success else "❌"
        role = response.metadata.get("role", "Agent")

        panel_content = response.content if response.success else f"[red]{response.content}[/red]"

        console.print(Panel.fit(
            panel_content,
            title=f"{status} {role} ({response.agent}) - {response.latency:.2f}s",
            border_style="green" if response.success else "red"
        ))

    # Generate consensus
    if strategy == "vote":
        consensus = consensus_engine.vote_consensus(responses)
    elif strategy == "synthesis":
        consensus = consensus_engine.synthesis_consensus(responses)
    elif strategy == "fanout":
        # No consensus needed for fanout
        console.print("\\n[bold green]✨ Trinity Complete - Fanout Mode[/bold green]")
        return
    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        return

    # Display consensus
    console.print("\\n[bold]🎯 Trinity Consensus:[/bold]")
    console.print(Panel.fit(
        Markdown(consensus.consensus_content),
        title=f"Consensus ({consensus.strategy}) - Confidence: {consensus.confidence:.2%}",
        border_style="blue"
    ))

    console.print(f"\\n[dim]Reasoning: {consensus.reasoning}[/dim]")

    # Save session if requested
    if save_session:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = SESSION_LOGS / f"trinity_{session_id}.json"

        session_data = {
            "prompt": prompt,
            "strategy": strategy,
            "agents": agent_names,
            "responses": [
                {
                    "agent": r.agent,
                    "success": r.success,
                    "content": r.content,
                    "latency": r.latency,
                    "metadata": r.metadata,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in responses
            ],
            "consensus": {
                "strategy": consensus.strategy,
                "content": consensus.consensus_content,
                "confidence": consensus.confidence,
                "reasoning": consensus.reasoning
            }
        }

        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        console.print(f"\\n[dim]Session saved: {session_file}[/dim]")

@app.command()
def observe(
    agent: str = typer.Option(..., help="Agent name to store observations for"),
    max_lines: int = typer.Option(10000, help="Maximum lines to store in buffer"),
):
    """📊 Store stdin observations for an agent (for pipeline mode)"""

    # Read from stdin and store in agent's observation buffer
    buffer_file = OBSERVATION_BUFFERS / f"{agent}.log"

    try:
        data = sys.stdin.read()

        # Append to buffer
        with open(buffer_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {data}\\n")

        # Trim buffer if too large
        if buffer_file.exists():
            with open(buffer_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) > max_lines:
                with open(buffer_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-max_lines:])

        console.print(f"[dim]Observation stored for {agent}[/dim]")

    except Exception as e:
        console.print(f"[red]Error storing observation: {e}[/red]")

@app.command()
def synthesize(
    strategy: str = typer.Option("vote", help="Synthesis strategy: vote, synthesis"),
    agents: str = typer.Option("claude,gemini,codex", help="Agents to include in synthesis"),
    context_lines: int = typer.Option(100, help="Lines of context to include from observations"),
    config: str = typer.Option("config/agents.yaml", help="Path to agent configuration file"),
):
    """🧠 Synthesize observations from multiple agents"""

    agent_names = [name.strip() for name in agents.split(",")]

    # Gather context from observation buffers
    context_data = {}
    for agent_name in agent_names:
        buffer_file = OBSERVATION_BUFFERS / f"{agent_name}.log"
        if buffer_file.exists():
            with open(buffer_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                context_data[agent_name] = "".join(lines[-context_lines:])

    if not context_data:
        console.print("[yellow]No observation data found for synthesis[/yellow]")
        return

    # Create synthesis prompt
    synthesis_prompt = "Based on the following observations, provide a synthesis and next action recommendation:\\n\\n"

    for agent_name, context in context_data.items():
        synthesis_prompt += f"=== {agent_name.upper()} OBSERVATIONS ===\\n{context}\\n\\n"

    synthesis_prompt += "Provide: 1) Summary of key insights, 2) Recommended next action"

    # Run synthesis through trinity
    console.print("[dim]Running synthesis through Trinity...[/dim]")

    # This would call the ask command internally
    # For now, simplified output
    console.print(Panel.fit(
        f"Synthesis request prepared with {len(context_data)} agent observations",
        title="🧠 Synthesis Ready"
    ))

@app.command()
def status():
    """📊 Show Trinity system status"""

    table = Table(title="🤖 AGI Trinity Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Check configuration
    config_exists = os.path.exists("config/agents.yaml")
    table.add_row(
        "Configuration",
        "✅ Ready" if config_exists else "❌ Missing",
        "config/agents.yaml" if config_exists else "Run trinity.py ask to create"
    )

    # Check observation buffers
    obs_count = len(list(OBSERVATION_BUFFERS.glob("*.log")))
    table.add_row(
        "Observation Buffers",
        f"📊 {obs_count} active",
        str(OBSERVATION_BUFFERS)
    )

    # Check session logs
    session_count = len(list(SESSION_LOGS.glob("trinity_*.json")))
    table.add_row(
        "Session History",
        f"📚 {session_count} sessions",
        str(SESSION_LOGS)
    )

    console.print(table)

if __name__ == "__main__":
    app()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Trinity - Continual Learning AGI with LFM2-VL
지속학습 기반 AGI 시스템

LFM2-VL-1.6B 비전-언어 모델 기반의 지속학습 AGI
"""
import os
import sys
import asyncio
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

app = typer.Typer(
    help="🧠 AGI Trinity - Continual Learning AGI with LFM2-VL",
    rich_markup_mode="rich"
)
console = Console()

# Global paths
HOME = Path(os.path.expanduser("~"))
AGI_HOME = HOME / ".trinity"
CONFIG_PATH = Path(__file__).parent / "config" / "lfm2_config.yaml"

# Ensure directories exist
AGI_HOME.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """설정 로드"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


# Global agent instance
_agent = None
_learning_engine = None


async def get_agent():
    """에이전트 인스턴스 가져오기 (싱글톤)"""
    global _agent

    if _agent is None:
        from agents.lfm2_adapter import LFM2VLAdapter, LFM2Config

        config = load_config()
        model_config = config.get("model", {})
        gen_config = config.get("generation", {})
        memory_config = config.get("memory", {})
        cl_config = config.get("continual_learning", {})

        lfm2_config = LFM2Config(
            model_id=model_config.get("id", "LiquidAI/LFM2-VL-1.6B"),
            device=model_config.get("device", "auto"),
            dtype=model_config.get("dtype", "bfloat16"),
            max_new_tokens=gen_config.get("max_new_tokens", 512),
            temperature=gen_config.get("temperature", 0.1),
            min_p=gen_config.get("min_p", 0.15),
            repetition_penalty=gen_config.get("repetition_penalty", 1.05),
            memory_path=memory_config.get("storage_path", "~/.trinity/lfm2_memory"),
            enable_continual_learning=cl_config.get("enabled", True),
            learning_rate=cl_config.get("learning_rate", 1e-5)
        )

        _agent = LFM2VLAdapter(lfm2_config=lfm2_config)

    return _agent


async def get_learning_engine():
    """학습 엔진 인스턴스 가져오기"""
    global _learning_engine

    if _learning_engine is None:
        from core.continual_learning import ContinualLearningEngine

        agent = await get_agent()
        _learning_engine = ContinualLearningEngine(model_adapter=agent)

    return _learning_engine


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="질문 또는 작업 요청"),
    image: Optional[str] = typer.Option(None, "--image", "-i", help="이미지 경로 또는 URL"),
    save_history: bool = typer.Option(True, "--save/--no-save", help="대화 기록 저장"),
    show_stats: bool = typer.Option(False, "--stats", help="메모리 통계 표시")
):
    """
    🧠 AGI에게 질문하기

    텍스트 또는 이미지와 함께 질문할 수 있습니다.
    모든 상호작용은 지속학습에 활용됩니다.
    """
    console.print(Panel.fit(
        "[bold blue]AGI Trinity[/bold blue] - Continual Learning AGI\n"
        "[dim]Powered by LFM2-VL-1.6B[/dim]",
        title="🧠"
    ))

    async def run():
        agent = await get_agent()
        learning_engine = await get_learning_engine()

        # 이미지 처리
        images = None
        if image:
            console.print(f"[dim]Loading image: {image}[/dim]")
            images = [image]

        # 모델 로드
        with console.status("[bold green]Loading AGI model...[/bold green]"):
            if not agent._is_loaded:
                await agent.load_model()

        # 추론
        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            response = await agent.execute(prompt, images)

        # 결과 표시
        if response.success:
            console.print("\n[bold green]AGI Response:[/bold green]")
            console.print(Panel(
                Markdown(response.content),
                border_style="green"
            ))

            # 메타데이터
            console.print(f"\n[dim]Latency: {response.latency:.2f}s | "
                         f"Tokens: {response.metadata.get('tokens_generated', 'N/A')}[/dim]")

            # 학습 엔진에 기록
            if save_history:
                exp_id = await learning_engine.record_interaction(
                    prompt=prompt,
                    response=response.content,
                    has_image=images is not None
                )
                console.print(f"[dim]Experience ID: {exp_id}[/dim]")

        else:
            console.print(f"\n[bold red]Error:[/bold red] {response.error}")

        # 통계 표시
        if show_stats:
            stats = agent.get_memory_stats()
            learning_stats = learning_engine.get_learning_stats()

            table = Table(title="Memory & Learning Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Experience Buffer", str(stats["experience_buffer_size"]))
            table.add_row("Learned Concepts", str(stats["learned_concepts"]))
            table.add_row("Interactions", str(stats["interaction_count"]))
            table.add_row("Training Count", str(learning_stats["training_count"]))
            table.add_row("Current Difficulty", f"{learning_stats['current_difficulty']:.2f}")

            console.print("\n")
            console.print(table)

    asyncio.run(run())


@app.command()
def feedback(
    experience_id: str = typer.Argument(..., help="경험 ID"),
    quality: float = typer.Option(..., "--quality", "-q", help="품질 점수 (0.0-1.0)"),
    correction: Optional[str] = typer.Option(None, "--correction", "-c", help="수정된 응답"),
    comment: Optional[str] = typer.Option(None, "--comment", help="피드백 코멘트")
):
    """
    📝 피드백 제공하기

    이전 응답에 대한 피드백을 제공하여 AGI의 학습을 돕습니다.
    """
    if not 0.0 <= quality <= 1.0:
        console.print("[red]Error: Quality must be between 0.0 and 1.0[/red]")
        raise typer.Exit(1)

    async def run():
        learning_engine = await get_learning_engine()

        await learning_engine.provide_feedback(
            experience_id=experience_id,
            quality_score=quality,
            user_feedback=comment,
            correction=correction
        )

        console.print(f"[green]✓ Feedback recorded for {experience_id}[/green]")
        console.print(f"  Quality: {quality:.1%}")
        if correction:
            console.print(f"  Correction provided: {len(correction)} chars")

    asyncio.run(run())


@app.command()
def train(
    force: bool = typer.Option(False, "--force", "-f", help="강제 훈련 트리거"),
    min_quality: float = typer.Option(0.7, "--min-quality", help="최소 품질 임계값")
):
    """
    🎓 지속학습 훈련 실행

    수집된 고품질 경험으로 모델을 훈련합니다.
    """
    console.print(Panel.fit(
        "[bold yellow]Starting Continual Learning Training[/bold yellow]",
        title="🎓 Training"
    ))

    async def run():
        learning_engine = await get_learning_engine()

        stats = learning_engine.get_learning_stats()
        console.print(f"Buffer size: {stats['buffer_size']}")
        console.print(f"High quality ratio: {stats['high_quality_ratio']:.1%}")

        if force or stats['buffer_size'] >= 10:
            with console.status("[bold green]Training...[/bold green]"):
                await learning_engine.trigger_training()

            console.print("[green]✓ Training completed[/green]")
        else:
            console.print("[yellow]Not enough data for training. Continue interacting with the AGI.[/yellow]")

    asyncio.run(run())


@app.command()
def status():
    """
    📊 시스템 상태 확인
    """
    table = Table(title="🧠 AGI Trinity Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Config check
    config_exists = CONFIG_PATH.exists()
    table.add_row(
        "Configuration",
        "✅ Ready" if config_exists else "⚠️ Using defaults",
        str(CONFIG_PATH) if config_exists else "Default config"
    )

    # Memory check
    memory_path = AGI_HOME / "lfm2_memory"
    memory_exists = memory_path.exists()
    if memory_exists:
        exp_files = list(memory_path.glob("experiences_*.json"))
        table.add_row(
            "Memory Storage",
            f"📊 {len(exp_files)} experience files",
            str(memory_path)
        )
    else:
        table.add_row(
            "Memory Storage",
            "📭 Empty",
            "No experiences yet"
        )

    # Model checkpoint
    checkpoint_path = memory_path / "model_checkpoint"
    if checkpoint_path.exists():
        table.add_row(
            "Model Checkpoint",
            "✅ Available",
            str(checkpoint_path)
        )
    else:
        table.add_row(
            "Model Checkpoint",
            "📦 Using base model",
            "LiquidAI/LFM2-VL-1.6B"
        )

    # Knowledge
    knowledge_path = AGI_HOME / "learning" / "knowledge" / "knowledge_graph.json"
    if knowledge_path.exists():
        with open(knowledge_path, 'r') as f:
            kg = json.load(f)
        table.add_row(
            "Knowledge Graph",
            f"🧠 {len(kg)} concepts",
            str(knowledge_path)
        )
    else:
        table.add_row(
            "Knowledge Graph",
            "📭 Empty",
            "No concepts learned yet"
        )

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            table.add_row(
                "GPU",
                f"✅ {gpu_name}",
                f"{gpu_mem:.1f} GB"
            )
        else:
            table.add_row("GPU", "❌ Not available", "CPU mode")
    except ImportError:
        table.add_row("PyTorch", "❌ Not installed", "pip install torch")

    console.print(table)


@app.command()
def chat():
    """
    💬 대화형 채팅 모드

    연속적인 대화를 통해 AGI와 상호작용합니다.
    """
    console.print(Panel.fit(
        "[bold blue]AGI Trinity Chat Mode[/bold blue]\n"
        "[dim]Type 'exit' to quit, 'stats' for statistics[/dim]",
        title="💬 Chat"
    ))

    async def run():
        agent = await get_agent()
        learning_engine = await get_learning_engine()

        # 모델 로드
        with console.status("[bold green]Loading AGI...[/bold green]"):
            if not agent._is_loaded:
                await agent.load_model()

        console.print("[green]AGI ready! Start chatting.[/green]\n")

        conversation_history = []

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

                if user_input.lower() == 'exit':
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if user_input.lower() == 'stats':
                    stats = learning_engine.get_learning_stats()
                    console.print(f"\n[dim]Buffer: {stats['buffer_size']} | "
                                 f"Concepts: {stats['knowledge_concepts']} | "
                                 f"Difficulty: {stats['current_difficulty']:.2f}[/dim]\n")
                    continue

                if user_input.lower() == 'train':
                    await learning_engine.trigger_training()
                    console.print("[green]Training triggered[/green]\n")
                    continue

                if not user_input.strip():
                    continue

                # 컨텍스트 추가
                context = ""
                if conversation_history:
                    context = "\n".join([
                        f"User: {h['user']}\nAssistant: {h['assistant']}"
                        for h in conversation_history[-3:]  # 최근 3개
                    ])
                    context += "\n\n"

                full_prompt = context + user_input

                # 응답 생성
                with console.status("[bold cyan]Thinking...[/bold cyan]"):
                    response = await agent.execute(full_prompt)

                if response.success:
                    console.print(f"\n[bold green]AGI[/bold green]: {response.content}\n")

                    # 기록
                    conversation_history.append({
                        "user": user_input,
                        "assistant": response.content
                    })

                    # 학습 엔진에 기록
                    await learning_engine.record_interaction(
                        prompt=user_input,
                        response=response.content
                    )
                else:
                    console.print(f"\n[red]Error: {response.error}[/red]\n")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue

    asyncio.run(run())


@app.command()
def learn(
    topic: str = typer.Argument(..., help="학습할 주제"),
    depth: int = typer.Option(3, "--depth", "-d", help="학습 깊이 (1-5)")
):
    """
    📚 특정 주제 학습하기

    AGI가 특정 주제에 대해 자기 주도적으로 학습합니다.
    """
    console.print(Panel.fit(
        f"[bold blue]Learning Topic: {topic}[/bold blue]\n"
        f"[dim]Depth: {depth}[/dim]",
        title="📚 Self-Learning"
    ))

    async def run():
        agent = await get_agent()
        learning_engine = await get_learning_engine()

        if not agent._is_loaded:
            with console.status("[bold green]Loading AGI...[/bold green]"):
                await agent.load_model()

        # 자기 주도 학습 루프
        for level in range(1, depth + 1):
            console.print(f"\n[bold]Level {level}/{depth}[/bold]")

            # 학습 프롬프트 생성
            prompts = [
                f"What is {topic}? Explain the fundamentals.",
                f"What are the key concepts and principles of {topic}?",
                f"How does {topic} relate to other fields?",
                f"What are practical applications of {topic}?",
                f"What are advanced topics in {topic}?"
            ]

            prompt = prompts[min(level - 1, len(prompts) - 1)]

            with console.status(f"[cyan]Learning: {prompt[:50]}...[/cyan]"):
                response = await agent.execute(prompt)

            if response.success:
                console.print(Panel(
                    Markdown(response.content[:500] + "..." if len(response.content) > 500 else response.content),
                    title=f"Level {level} Understanding"
                ))

                # 학습 기록
                await learning_engine.record_interaction(
                    prompt=prompt,
                    response=response.content,
                    domain=topic
                )

        console.print("\n[green]✓ Self-learning session completed[/green]")

    asyncio.run(run())


@app.command()
def knowledge(
    query: Optional[str] = typer.Argument(None, help="검색할 개념"),
    list_all: bool = typer.Option(False, "--list", "-l", help="모든 개념 나열")
):
    """
    🧠 지식 그래프 조회

    AGI가 학습한 개념들을 조회합니다.
    """
    async def run():
        learning_engine = await get_learning_engine()

        if list_all:
            # 모든 개념 나열
            concepts = learning_engine.knowledge.knowledge_graph

            if not concepts:
                console.print("[yellow]No concepts learned yet.[/yellow]")
                return

            table = Table(title="🧠 Learned Concepts")
            table.add_column("Concept", style="cyan")
            table.add_column("Domain", style="green")
            table.add_column("Access Count", style="yellow")

            for concept_id, data in concepts.items():
                table.add_row(
                    data.get("name", concept_id)[:30],
                    data.get("domain", "general"),
                    str(data.get("access_count", 0))
                )

            console.print(table)

        elif query:
            # 개념 검색
            results = await learning_engine.get_relevant_knowledge(query)

            if not results:
                console.print(f"[yellow]No concepts found for '{query}'[/yellow]")
                return

            console.print(f"\n[bold]Found {len(results)} related concepts:[/bold]\n")

            for i, concept in enumerate(results, 1):
                console.print(Panel(
                    f"[bold]{concept.get('name', 'Unknown')}[/bold]\n\n"
                    f"{concept.get('definition', '')[:300]}",
                    title=f"#{i}"
                ))

        else:
            console.print("[yellow]Please provide a query or use --list[/yellow]")

    asyncio.run(run())


@app.command()
def export(
    output_path: str = typer.Argument("agi_export.json", help="출력 파일 경로"),
    include_model: bool = typer.Option(False, "--include-model", help="모델 체크포인트 포함")
):
    """
    📤 AGI 상태 내보내기

    학습된 지식과 경험을 내보냅니다.
    """
    async def run():
        learning_engine = await get_learning_engine()
        agent = await get_agent()

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "learning_stats": learning_engine.get_learning_stats(),
            "memory_stats": agent.get_memory_stats(),
            "knowledge_graph": learning_engine.knowledge.knowledge_graph,
            "long_term_memory": agent.long_term_memory
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        console.print(f"[green]✓ Exported to {output_path}[/green]")

    asyncio.run(run())


# ============================================
# 웹 크롤링 기반 지속학습 명령어
# ============================================

@app.command()
def crawl(
    topic: str = typer.Argument(..., help="크롤링할 주제"),
    pages: int = typer.Option(20, "--pages", "-p", help="크롤링할 페이지 수"),
    urls: Optional[List[str]] = typer.Option(None, "--url", "-u", help="시드 URL (여러 개 가능)")
):
    """
    🌐 웹 크롤링으로 주제 학습

    웹에서 정보를 수집하여 특정 주제를 학습합니다.
    """
    console.print(Panel.fit(
        f"[bold blue]Web Crawling: {topic}[/bold blue]\n"
        f"[dim]Pages: {pages}[/dim]",
        title="🌐 Web Learning"
    ))

    async def run():
        from core.web_crawler import ContinuousWebLearner

        learning_engine = await get_learning_engine()

        web_learner = ContinuousWebLearner(
            learning_engine=learning_engine,
            storage_path=str(AGI_HOME / "web_learning")
        )

        with console.status(f"[bold green]Crawling web for '{topic}'...[/bold green]"):
            result = await web_learner.learn_topic(
                topic=topic,
                max_pages=pages,
                sources=list(urls) if urls else None
            )

        console.print("\n[bold green]Crawling Complete![/bold green]")

        table = Table(title="Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Pages Crawled", str(result["pages_crawled"]))
        table.add_row("Items Learned", str(result["items_learned"]))
        table.add_row("Time Elapsed", f"{result['elapsed_seconds']:.1f}s")
        table.add_row("Avg Quality", f"{result['avg_quality']:.2f}")

        console.print(table)

        # 전체 통계
        stats = web_learner.get_stats()
        console.print(f"\n[dim]Total learned: {stats['total_learned']} items[/dim]")

    asyncio.run(run())


@app.command()
def web_learn(
    source: str = typer.Option("tech", "--source", "-s",
        help="소스 유형: wikipedia, tech, science, programming, korean"),
    pages: int = typer.Option(30, "--pages", "-p", help="크롤링할 페이지 수")
):
    """
    📚 사전정의 소스에서 학습

    신뢰할 수 있는 지식 소스에서 학습합니다.
    - wikipedia: 위키피디아
    - tech: AI/기술 뉴스
    - science: 과학 뉴스
    - programming: 프로그래밍
    - korean: 한국어 콘텐츠
    """
    console.print(Panel.fit(
        f"[bold blue]Learning from: {source}[/bold blue]\n"
        f"[dim]Max pages: {pages}[/dim]",
        title="📚 Source Learning"
    ))

    async def run():
        from core.web_crawler import ContinuousWebLearner

        learning_engine = await get_learning_engine()

        web_learner = ContinuousWebLearner(
            learning_engine=learning_engine,
            storage_path=str(AGI_HOME / "web_learning")
        )

        with console.status(f"[bold green]Learning from {source} sources...[/bold green]"):
            result = await web_learner.learn_from_sources(
                source_type=source,
                max_pages=pages
            )

        console.print("\n[bold green]Learning Complete![/bold green]")
        console.print(f"Pages crawled: {result['pages_crawled']}")
        console.print(f"Total learned: {result['total_learned']}")

    asyncio.run(run())


@app.command()
def auto_learn(
    topics: Optional[List[str]] = typer.Option(None, "--topic", "-t", help="학습 주제 (여러 개 가능)"),
    interval: int = typer.Option(60, "--interval", "-i", help="학습 간격 (분)"),
    pages: int = typer.Option(10, "--pages", "-p", help="주제당 페이지 수"),
    cycles: int = typer.Option(0, "--cycles", "-c", help="최대 주기 (0=무한)"),
    background: bool = typer.Option(False, "--background", "-b", help="백그라운드 실행")
):
    """
    🔄 자동 지속학습 실행

    백그라운드에서 웹을 크롤링하며 지속적으로 학습합니다.

    예시:
        python agi.py auto-learn --topic "machine learning" --topic "python" --interval 30
    """
    if background:
        console.print("[yellow]백그라운드 모드는 scripts/auto_learner.py를 직접 실행하세요:[/yellow]")
        console.print("  python scripts/auto_learner.py --interval 60 &")
        return

    console.print(Panel.fit(
        f"[bold blue]Auto Learning Mode[/bold blue]\n"
        f"[dim]Interval: {interval} min | Pages/topic: {pages}[/dim]\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        title="🔄 Continuous Learning"
    ))

    async def run():
        from core.web_crawler import ContinuousWebLearner

        learning_engine = await get_learning_engine()

        web_learner = ContinuousWebLearner(
            learning_engine=learning_engine,
            storage_path=str(AGI_HOME / "web_learning")
        )

        # 기본 주제
        default_topics = [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "python programming",
            "computer science",
            "인공지능",
            "머신러닝"
        ]

        learn_topics = list(topics) if topics else default_topics

        console.print(f"Topics to learn: {', '.join(learn_topics)}")
        console.print("\n[green]Starting continuous learning...[/green]\n")

        try:
            await web_learner.continuous_learn(
                topics=learn_topics,
                interval_minutes=interval,
                pages_per_topic=pages
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
            web_learner.stop()

        # 최종 통계
        stats = web_learner.get_stats()
        console.print(f"\n[bold]Final Stats:[/bold]")
        console.print(f"  Total crawled: {stats['total_crawled']}")
        console.print(f"  Total learned: {stats['total_learned']}")

        if stats['top_topics']:
            console.print(f"\n[bold]Top Topics:[/bold]")
            for topic, count in stats['top_topics'][:5]:
                console.print(f"  • {topic}: {count}")

    asyncio.run(run())


@app.command()
def web_stats():
    """
    📊 웹 학습 통계 조회

    웹 크롤링을 통한 학습 통계를 표시합니다.
    """
    async def run():
        from core.web_crawler import ContinuousWebLearner

        web_learner = ContinuousWebLearner(
            storage_path=str(AGI_HOME / "web_learning")
        )

        stats = web_learner.get_stats()

        table = Table(title="🌐 Web Learning Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Pages Crawled", str(stats['total_crawled']))
        table.add_row("Total Items Learned", str(stats['total_learned']))
        table.add_row("Last Run", stats.get('last_run', 'Never'))
        table.add_row("Storage Path", stats['storage_path'])

        console.print(table)

        if stats['top_topics']:
            console.print("\n[bold]Top Learned Topics:[/bold]")
            for topic, count in stats['top_topics']:
                bar = "█" * min(count, 20)
                console.print(f"  {topic[:30]:30} {bar} ({count})")

    asyncio.run(run())


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
AGI Trinity - Claude Code Adapter
Claude Code CLI 어댑터
"""
import asyncio
import shutil
from datetime import datetime
from typing import Optional

from .base import BaseAgentAdapter, AgentConfig, AgentResponse, AgentStatus


class ClaudeAdapter(BaseAgentAdapter):
    """
    Claude Code CLI 어댑터

    Claude Code CLI를 통해 Claude AI와 통신합니다.
    코드 분석, 디버깅, 시스템 설계에 특화되어 있습니다.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="claude",
                role="Technical Expert",
                specialty="Code analysis, debugging, system design, security review",
                mode="batch",
                cmd=["claude", "--print", "--dangerously-skip-permissions"],
                timeout_s=180,
                personality="Analytical, precise, technical",
                strengths=[
                    "Deep code analysis",
                    "System architecture design",
                    "Debugging and optimization",
                    "Security vulnerability detection",
                    "Best practices enforcement"
                ]
            )
        super().__init__(config)
        self._claude_path = self._find_claude_cli()

    def _find_claude_cli(self) -> Optional[str]:
        """Claude CLI 경로 찾기"""
        # 일반적인 설치 경로들
        possible_paths = [
            "claude",
            "claude-code",
            "/usr/local/bin/claude",
            "/usr/bin/claude",
        ]

        for path in possible_paths:
            if shutil.which(path):
                return path
        return None

    async def execute(self, prompt: str) -> AgentResponse:
        """
        Claude에게 프롬프트를 실행합니다.
        """
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()

        try:
            if not self._claude_path:
                raise RuntimeError("Claude CLI not found. Please install claude-code.")

            # 명령어 구성
            cmd = self.config.cmd + [prompt]

            # 프로세스 실행
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    self._process.communicate(),
                    timeout=self.config.timeout_s
                )
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
                raise TimeoutError(f"Claude timeout after {self.config.timeout_s}s")

            latency = asyncio.get_event_loop().time() - start_time
            success = self._process.returncode == 0
            content = stdout.decode('utf-8', errors='ignore').strip()
            error_msg = stderr.decode('utf-8', errors='ignore').strip() if not success else None

            response = AgentResponse(
                agent_name=self.name,
                success=success,
                content=content,
                latency=latency,
                metadata={
                    "role": self.role,
                    "specialty": self.config.specialty,
                    "return_code": self._process.returncode
                },
                error=error_msg,
                confidence=0.95 if success else 0.0
            )

            self._update_metrics(response)
            self.status = AgentStatus.SUCCESS if success else AgentStatus.ERROR
            return response

        except TimeoutError as e:
            latency = asyncio.get_event_loop().time() - start_time
            self.status = AgentStatus.TIMEOUT
            response = AgentResponse(
                agent_name=self.name,
                success=False,
                content="",
                latency=latency,
                error=str(e)
            )
            self._update_metrics(response)
            return response

        except Exception as e:
            latency = asyncio.get_event_loop().time() - start_time
            self.status = AgentStatus.ERROR
            response = AgentResponse(
                agent_name=self.name,
                success=False,
                content="",
                latency=latency,
                error=str(e)
            )
            self._update_metrics(response)
            return response

    async def health_check(self) -> bool:
        """Claude CLI 상태 확인"""
        try:
            if not self._claude_path:
                return False

            process = await asyncio.create_subprocess_exec(
                self._claude_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except Exception:
            return False

    async def analyze_code(self, code: str, language: str = "python") -> AgentResponse:
        """
        코드 분석 특화 메서드
        """
        prompt = f"""Analyze the following {language} code:

```{language}
{code}
```

Provide:
1. Code quality assessment
2. Potential bugs or issues
3. Security vulnerabilities
4. Performance optimization suggestions
5. Best practice recommendations
"""
        return await self.execute(prompt)

    async def review_architecture(self, description: str) -> AgentResponse:
        """
        아키텍처 리뷰 특화 메서드
        """
        prompt = f"""Review the following system architecture:

{description}

Provide:
1. Architecture assessment
2. Scalability analysis
3. Potential bottlenecks
4. Security considerations
5. Improvement recommendations
"""
        return await self.execute(prompt)

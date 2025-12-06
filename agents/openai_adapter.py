#!/usr/bin/env python3
"""
AGI Trinity - OpenAI GPT-4 Adapter
OpenAI API 어댑터
"""
import asyncio
import os
import json
from datetime import datetime
from typing import Optional

from .base import BaseAgentAdapter, AgentConfig, AgentResponse, AgentStatus


class OpenAIAdapter(BaseAgentAdapter):
    """
    OpenAI GPT-4 어댑터

    OpenAI API를 통해 GPT-4와 통신합니다.
    창의적 문제 해결, 브레인스토밍, 전략 수립에 특화되어 있습니다.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="codex",
                role="Creative Problem Solver",
                specialty="Innovation, creative solutions, brainstorming, strategic thinking",
                mode="batch",
                cmd=["openai"],
                timeout_s=180,
                personality="Innovative, creative, solution-oriented",
                strengths=[
                    "Creative problem solving",
                    "Strategic innovation",
                    "Brainstorming and ideation",
                    "Out-of-the-box thinking",
                    "Narrative and storytelling"
                ]
            )
        super().__init__(config)
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self._model = "gpt-4"

    async def execute(self, prompt: str) -> AgentResponse:
        """
        GPT-4에게 프롬프트를 실행합니다.
        """
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()

        try:
            if not self._api_key:
                raise RuntimeError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )

            # API 호출 구성
            api_url = "https://api.openai.com/v1/chat/completions"

            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a creative problem solver with exceptional ability "
                            "to think outside the box. Your responses should be innovative, "
                            "insightful, and provide unique perspectives on problems."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.8,
                "max_tokens": 4096,
                "top_p": 0.95
            }

            cmd = [
                "curl", "-s", "-X", "POST",
                api_url,
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {self._api_key}",
                "-d", json.dumps(payload)
            ]

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    self._process.communicate(),
                    timeout=self.config.timeout_s
                )
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
                raise TimeoutError(f"OpenAI timeout after {self.config.timeout_s}s")

            latency = asyncio.get_event_loop().time() - start_time

            # 응답 파싱
            try:
                response_data = json.loads(stdout.decode('utf-8'))

                if "error" in response_data:
                    raise RuntimeError(response_data["error"].get("message", "Unknown API error"))

                # 콘텐츠 추출
                choices = response_data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    success = True
                else:
                    content = ""
                    success = False

                # 토큰 사용량
                usage = response_data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

            except json.JSONDecodeError:
                content = stdout.decode('utf-8', errors='ignore')
                success = bool(content.strip())
                tokens_used = 0

            response = AgentResponse(
                agent_name=self.name,
                success=success,
                content=content,
                latency=latency,
                metadata={
                    "role": self.role,
                    "specialty": self.config.specialty,
                    "model": self._model
                },
                confidence=0.85 if success else 0.0,
                tokens_used=tokens_used
            )

            self._update_metrics(response)
            self.status = AgentStatus.SUCCESS if success else AgentStatus.ERROR
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
        """OpenAI API 상태 확인"""
        try:
            if not self._api_key:
                return False

            response = await self.execute("Hello, respond with 'OK'")
            return response.success
        except Exception:
            return False

    async def brainstorm(self, topic: str, count: int = 10) -> AgentResponse:
        """
        브레인스토밍 특화 메서드
        """
        prompt = f"""Brainstorm {count} innovative ideas for:

Topic: {topic}

For each idea, provide:
1. A catchy name
2. Brief description (2-3 sentences)
3. Key differentiator
4. Potential challenges
5. Implementation approach

Be creative and think beyond conventional solutions.
"""
        return await self.execute(prompt)

    async def solve_creatively(self, problem: str) -> AgentResponse:
        """
        창의적 문제 해결 특화 메서드
        """
        prompt = f"""Solve the following problem creatively:

Problem: {problem}

Provide:
1. Reframe the problem from different angles
2. At least 3 unconventional solutions
3. For each solution:
   - Pros and cons
   - Resources needed
   - Risk assessment
4. Recommended approach with reasoning
5. Alternative perspectives to consider
"""
        return await self.execute(prompt)

    async def create_strategy(self, goal: str, context: str = "") -> AgentResponse:
        """
        전략 수립 특화 메서드
        """
        context_section = f"\nContext: {context}" if context else ""
        prompt = f"""Create a strategic plan for achieving this goal:

Goal: {goal}{context_section}

Provide:
1. Vision and mission alignment
2. Strategic objectives (SMART goals)
3. Key initiatives and milestones
4. Resource requirements
5. Risk analysis and mitigation strategies
6. Success metrics and KPIs
7. Short-term and long-term action items
"""
        return await self.execute(prompt)

    async def generate_narrative(self, topic: str, style: str = "professional") -> AgentResponse:
        """
        내러티브 생성 특화 메서드
        """
        prompt = f"""Create a compelling {style} narrative about:

Topic: {topic}

The narrative should:
1. Have a clear beginning, middle, and end
2. Include engaging storytelling elements
3. Convey key messages effectively
4. Resonate with the target audience
5. Include a call to action or memorable conclusion
"""
        return await self.execute(prompt)

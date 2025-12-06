#!/usr/bin/env python3
"""
AGI Trinity - Gemini Pro Adapter
Google Gemini API 어댑터
"""
import asyncio
import os
import json
from datetime import datetime
from typing import Optional

from .base import BaseAgentAdapter, AgentConfig, AgentResponse, AgentStatus


class GeminiAdapter(BaseAgentAdapter):
    """
    Google Gemini Pro 어댑터

    Gemini API를 통해 Google AI와 통신합니다.
    연구, 데이터 분석, 팩트 체킹에 특화되어 있습니다.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="gemini",
                role="Data Analyst",
                specialty="Research, analysis, fact-checking, data interpretation",
                mode="batch",
                cmd=["gemini"],
                timeout_s=180,
                personality="Methodical, thorough, evidence-based",
                strengths=[
                    "Comprehensive research",
                    "Data analysis and visualization",
                    "Fact verification",
                    "Market analysis",
                    "Multi-modal understanding"
                ]
            )
        super().__init__(config)
        self._api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    async def execute(self, prompt: str) -> AgentResponse:
        """
        Gemini에게 프롬프트를 실행합니다.
        """
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()

        try:
            # API 키 확인
            if not self._api_key:
                raise RuntimeError(
                    "Gemini API key not found. "
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                )

            # curl을 사용한 API 호출
            api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 8192
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            cmd = [
                "curl", "-s", "-X", "POST",
                f"{api_url}?key={self._api_key}",
                "-H", "Content-Type: application/json",
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
                raise TimeoutError(f"Gemini timeout after {self.config.timeout_s}s")

            latency = asyncio.get_event_loop().time() - start_time

            # 응답 파싱
            try:
                response_data = json.loads(stdout.decode('utf-8'))

                if "error" in response_data:
                    raise RuntimeError(response_data["error"].get("message", "Unknown API error"))

                # 콘텐츠 추출
                candidates = response_data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    success = True
                else:
                    content = ""
                    success = False

                # 토큰 사용량
                usage = response_data.get("usageMetadata", {})
                tokens_used = usage.get("totalTokenCount", 0)

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
                    "model": "gemini-pro"
                },
                confidence=0.9 if success else 0.0,
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
        """Gemini API 상태 확인"""
        try:
            if not self._api_key:
                return False

            # 간단한 테스트 요청
            response = await self.execute("Hello, respond with 'OK'")
            return response.success
        except Exception:
            return False

    async def analyze_data(self, data: str, analysis_type: str = "general") -> AgentResponse:
        """
        데이터 분석 특화 메서드
        """
        prompt = f"""Analyze the following data with focus on {analysis_type}:

{data}

Provide:
1. Data summary and key statistics
2. Patterns and trends identified
3. Anomalies or outliers
4. Insights and conclusions
5. Recommendations based on the analysis
"""
        return await self.execute(prompt)

    async def fact_check(self, claim: str) -> AgentResponse:
        """
        팩트 체킹 특화 메서드
        """
        prompt = f"""Fact-check the following claim:

"{claim}"

Provide:
1. Verdict (True/False/Partially True/Unverifiable)
2. Evidence supporting or refuting the claim
3. Context and nuances
4. Sources that could verify this information
5. Confidence level in your assessment
"""
        return await self.execute(prompt)

    async def research_topic(self, topic: str, depth: str = "comprehensive") -> AgentResponse:
        """
        주제 연구 특화 메서드
        """
        prompt = f"""Research the following topic in {depth} detail:

Topic: {topic}

Provide:
1. Overview and background
2. Key concepts and terminology
3. Current state of knowledge
4. Major developments and trends
5. Open questions and future directions
6. Practical applications
"""
        return await self.execute(prompt)

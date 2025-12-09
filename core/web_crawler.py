#!/usr/bin/env python3
"""
AGI Trinity - Web Crawler for Continuous Learning
웹 크롤링 기반 지속학습 모듈

웹에서 정보를 수집하여 AGI의 지속적인 학습을 지원합니다.
"""
import asyncio
import aiohttp
import hashlib
import json
import os
import re
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from collections import deque
import logging

# HTML 파싱
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Robots.txt 파싱
try:
    from urllib.robotparser import RobotFileParser
    ROBOTPARSER_AVAILABLE = True
except ImportError:
    ROBOTPARSER_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawledContent:
    """크롤링된 콘텐츠"""
    url: str
    title: str
    content: str
    summary: Optional[str] = None
    domain: str = ""
    language: str = "unknown"
    word_count: int = 0
    crawled_at: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.5
    topics: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        if not self.domain:
            self.domain = urlparse(self.url).netloc
        self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "domain": self.domain,
            "language": self.language,
            "word_count": self.word_count,
            "crawled_at": self.crawled_at.isoformat(),
            "quality_score": self.quality_score,
            "topics": self.topics,
            "hash": self.hash
        }


@dataclass
class CrawlConfig:
    """크롤링 설정"""
    max_pages: int = 100
    max_depth: int = 3
    delay_seconds: float = 1.0
    timeout_seconds: int = 30
    max_content_length: int = 100000  # 100KB
    min_content_length: int = 500
    respect_robots_txt: bool = True
    user_agent: str = "AGI-Trinity-Learner/1.0 (Educational AI; +https://github.com/hwkim3330/agi)"

    # 콘텐츠 필터
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=lambda: [
        "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
        "linkedin.com", "pinterest.com", "reddit.com"  # 소셜 미디어 제외
    ])
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "ko"])

    # 품질 필터
    min_quality_score: float = 0.3
    required_topics: List[str] = field(default_factory=list)


class ContentExtractor:
    """웹 페이지에서 콘텐츠 추출"""

    # 제거할 태그들
    REMOVE_TAGS = [
        'script', 'style', 'nav', 'header', 'footer', 'aside',
        'advertisement', 'ads', 'sidebar', 'menu', 'popup', 'modal',
        'cookie', 'banner', 'social', 'share', 'comment', 'form'
    ]

    # 메인 콘텐츠 태그들
    CONTENT_TAGS = ['article', 'main', 'content', 'post', 'entry', 'body']

    @classmethod
    def extract(cls, html: str, url: str) -> CrawledContent:
        """HTML에서 콘텐츠 추출"""
        if not BS4_AVAILABLE:
            # BeautifulSoup 없으면 간단한 정규식 사용
            return cls._extract_simple(html, url)

        soup = BeautifulSoup(html, 'html.parser')

        # 불필요한 태그 제거
        for tag in cls.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()
            for element in soup.find_all(class_=re.compile(tag, re.I)):
                element.decompose()
            for element in soup.find_all(id=re.compile(tag, re.I)):
                element.decompose()

        # 제목 추출
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()

        # 메인 콘텐츠 찾기
        main_content = None
        for tag in cls.CONTENT_TAGS:
            main_content = soup.find(tag)
            if main_content:
                break
            main_content = soup.find(class_=re.compile(tag, re.I))
            if main_content:
                break
            main_content = soup.find(id=re.compile(tag, re.I))
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body') or soup

        # 텍스트 추출
        text = main_content.get_text(separator='\n', strip=True)

        # 텍스트 정리
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # 중복 라인 제거
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen and len(line) > 10:
                seen.add(line)
                unique_lines.append(line)

        content = '\n'.join(unique_lines)

        # 링크 추출
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http'):
                links.append(href)
            elif href.startswith('/'):
                links.append(urljoin(url, href))

        # 언어 감지
        language = cls._detect_language(content)

        # 토픽 추출
        topics = cls._extract_topics(title + " " + content)

        return CrawledContent(
            url=url,
            title=title,
            content=content,
            language=language,
            topics=topics,
            links=links[:50]  # 최대 50개 링크
        )

    @classmethod
    def _extract_simple(cls, html: str, url: str) -> CrawledContent:
        """간단한 정규식 기반 추출"""
        # 태그 제거
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 제목 추출
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.I | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        return CrawledContent(
            url=url,
            title=title,
            content=text,
            language=cls._detect_language(text)
        )

    @staticmethod
    def _detect_language(text: str) -> str:
        """간단한 언어 감지"""
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "unknown"

        korean_ratio = korean_chars / total_chars
        if korean_ratio > 0.3:
            return "ko"
        return "en"

    @staticmethod
    def _extract_topics(text: str) -> List[str]:
        """토픽/키워드 추출"""
        # 불용어
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'and', 'but', 'or', 'for', 'with', 'not', 'you', 'your', 'they',
            'their', 'its', 'from', 'about', 'to', 'of', 'in', 'on', 'at', 'by'
        }

        words = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', text.lower())
        word_freq = {}

        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq >= 2]


class QualityAssessor:
    """콘텐츠 품질 평가"""

    @classmethod
    def assess(cls, content: CrawledContent) -> float:
        """콘텐츠 품질 점수 계산 (0-1)"""
        scores = []

        # 1. 길이 점수 (적절한 길이)
        word_count = content.word_count
        if word_count < 100:
            length_score = 0.2
        elif word_count < 500:
            length_score = 0.5
        elif word_count < 2000:
            length_score = 1.0
        elif word_count < 5000:
            length_score = 0.8
        else:
            length_score = 0.6
        scores.append(length_score)

        # 2. 구조 점수 (단락 수)
        paragraphs = [p for p in content.content.split('\n') if len(p) > 50]
        if len(paragraphs) >= 5:
            structure_score = 1.0
        elif len(paragraphs) >= 3:
            structure_score = 0.7
        else:
            structure_score = 0.4
        scores.append(structure_score)

        # 3. 토픽 점수 (키워드 다양성)
        topic_score = min(1.0, len(content.topics) / 5)
        scores.append(topic_score)

        # 4. 제목 존재
        title_score = 1.0 if content.title else 0.3
        scores.append(title_score)

        # 5. 언어 일관성 (혼합된 언어는 낮은 점수)
        lang_score = 1.0 if content.language != "unknown" else 0.5
        scores.append(lang_score)

        # 6. 스팸 체크
        spam_patterns = [
            r'buy now', r'click here', r'limited time', r'act now',
            r'free money', r'congratulations', r'you won', r'subscribe',
            r'광고', r'클릭', r'무료', r'당첨'
        ]
        spam_count = sum(1 for p in spam_patterns if re.search(p, content.content.lower()))
        spam_score = max(0, 1.0 - spam_count * 0.2)
        scores.append(spam_score)

        # 가중 평균
        weights = [0.2, 0.15, 0.15, 0.1, 0.15, 0.25]
        quality = sum(s * w for s, w in zip(scores, weights))

        return round(quality, 3)


class WebCrawler:
    """웹 크롤러"""

    def __init__(self, config: Optional[CrawlConfig] = None):
        self.config = config or CrawlConfig()
        self.visited_urls: Set[str] = set()
        self.url_queue: deque = deque()
        self.crawled_contents: List[CrawledContent] = []
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP 세션 가져오기"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = {"User-Agent": self.config.user_agent}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self):
        """세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _normalize_url(self, url: str) -> str:
        """URL 정규화"""
        parsed = urlparse(url)
        # fragment 제거
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized.rstrip('/')

    def _is_allowed_domain(self, url: str) -> bool:
        """도메인 허용 여부 확인"""
        domain = urlparse(url).netloc.lower()

        # 차단된 도메인 체크
        for blocked in self.config.blocked_domains:
            if blocked in domain:
                return False

        # 허용된 도메인만 있으면 체크
        if self.config.allowed_domains:
            return any(allowed in domain for allowed in self.config.allowed_domains)

        return True

    async def _can_fetch(self, url: str) -> bool:
        """robots.txt 준수 확인"""
        if not self.config.respect_robots_txt:
            return True

        if not ROBOTPARSER_AVAILABLE:
            return True

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if robots_url not in self.robots_cache:
            try:
                session = await self._get_session()
                async with session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        rp = RobotFileParser()
                        rp.parse(content.split('\n'))
                        self.robots_cache[robots_url] = rp
                    else:
                        self.robots_cache[robots_url] = None
            except Exception:
                self.robots_cache[robots_url] = None

        rp = self.robots_cache.get(robots_url)
        if rp is None:
            return True

        return rp.can_fetch(self.config.user_agent, url)

    async def _fetch_page(self, url: str) -> Optional[str]:
        """페이지 가져오기"""
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type:
                    return None

                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > self.config.max_content_length:
                    return None

                html = await response.text()
                return html

        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None

    async def crawl_url(self, url: str) -> Optional[CrawledContent]:
        """단일 URL 크롤링"""
        url = self._normalize_url(url)

        if url in self.visited_urls:
            return None

        if not self._is_allowed_domain(url):
            return None

        if not await self._can_fetch(url):
            logger.debug(f"Blocked by robots.txt: {url}")
            return None

        self.visited_urls.add(url)

        html = await self._fetch_page(url)
        if not html:
            return None

        # 콘텐츠 추출
        content = ContentExtractor.extract(html, url)

        # 최소 길이 체크
        if content.word_count < self.config.min_content_length // 5:
            return None

        # 언어 체크
        if content.language not in self.config.allowed_languages:
            return None

        # 품질 평가
        content.quality_score = QualityAssessor.assess(content)

        if content.quality_score < self.config.min_quality_score:
            return None

        # 요약 생성 (첫 3문장)
        sentences = re.split(r'[.!?。]', content.content)
        content.summary = '. '.join(s.strip() for s in sentences[:3] if s.strip())[:500]

        return content

    async def crawl(
        self,
        seed_urls: List[str],
        topics: Optional[List[str]] = None,
        callback: Optional[callable] = None
    ) -> List[CrawledContent]:
        """
        시드 URL에서 시작하여 크롤링

        Args:
            seed_urls: 시작 URL 목록
            topics: 관심 토픽 (관련 페이지 우선 크롤링)
            callback: 각 페이지 크롤링 후 콜백

        Returns:
            크롤링된 콘텐츠 목록
        """
        self._running = True
        self.crawled_contents = []
        self.visited_urls = set()

        # 시드 URL 추가
        for url in seed_urls:
            self.url_queue.append((url, 0))

        if topics:
            self.config.required_topics = topics

        logger.info(f"Starting crawl with {len(seed_urls)} seed URLs")

        try:
            while self.url_queue and len(self.crawled_contents) < self.config.max_pages:
                if not self._running:
                    break

                url, depth = self.url_queue.popleft()

                if depth > self.config.max_depth:
                    continue

                # 딜레이
                await asyncio.sleep(self.config.delay_seconds + random.uniform(0, 0.5))

                # 크롤링
                content = await self.crawl_url(url)

                if content:
                    self.crawled_contents.append(content)
                    logger.info(f"[{len(self.crawled_contents)}/{self.config.max_pages}] "
                               f"Crawled: {url[:60]}... (quality: {content.quality_score:.2f})")

                    # 콜백 호출
                    if callback:
                        await callback(content)

                    # 새 링크 추가
                    if depth < self.config.max_depth:
                        for link in content.links:
                            if link not in self.visited_urls:
                                # 토픽 관련 링크 우선
                                if topics and any(t.lower() in link.lower() for t in topics):
                                    self.url_queue.appendleft((link, depth + 1))
                                else:
                                    self.url_queue.append((link, depth + 1))

        finally:
            await self.close()

        logger.info(f"Crawl completed: {len(self.crawled_contents)} pages")
        return self.crawled_contents

    def stop(self):
        """크롤링 중지"""
        self._running = False


class ContinuousWebLearner:
    """
    지속적 웹 학습 엔진

    웹에서 정보를 수집하고 AGI를 지속적으로 학습시킵니다.
    """

    # 학습 소스 (신뢰할 수 있는 지식 소스)
    KNOWLEDGE_SOURCES = {
        "wikipedia": [
            "https://en.wikipedia.org/wiki/Main_Page",
            "https://ko.wikipedia.org/wiki/위키백과:대문"
        ],
        "tech": [
            "https://arxiv.org/list/cs.AI/recent",
            "https://news.ycombinator.com/",
            "https://dev.to/",
            "https://medium.com/tag/artificial-intelligence"
        ],
        "science": [
            "https://www.nature.com/",
            "https://www.sciencedaily.com/",
            "https://phys.org/"
        ],
        "programming": [
            "https://stackoverflow.com/questions",
            "https://github.com/trending",
            "https://www.freecodecamp.org/news/"
        ],
        "korean": [
            "https://ko.wikipedia.org/wiki/위키백과:대문",
            "https://brunch.co.kr/",
            "https://velog.io/"
        ]
    }

    def __init__(
        self,
        learning_engine=None,
        agi_agent=None,
        storage_path: str = "~/.trinity/web_learning"
    ):
        self.learning_engine = learning_engine
        self.agi_agent = agi_agent
        self.storage_path = Path(os.path.expanduser(storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.crawler = WebCrawler()
        self._running = False
        self._stats = {
            "total_crawled": 0,
            "total_learned": 0,
            "last_run": None,
            "topics_learned": {}
        }

        self._load_stats()

    def _load_stats(self):
        """통계 로드"""
        stats_file = self.storage_path / "learning_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self._stats = json.load(f)

    def _save_stats(self):
        """통계 저장"""
        stats_file = self.storage_path / "learning_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self._stats, f, indent=2, default=str)

    def _save_content(self, content: CrawledContent):
        """크롤링된 콘텐츠 저장"""
        date_str = datetime.now().strftime("%Y%m%d")
        content_file = self.storage_path / f"crawled_{date_str}.jsonl"

        with open(content_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(content.to_dict(), ensure_ascii=False) + '\n')

    async def _process_content(self, content: CrawledContent):
        """크롤링된 콘텐츠 처리 및 학습"""
        self._stats["total_crawled"] += 1

        # 콘텐츠 저장
        self._save_content(content)

        # 학습 엔진에 기록
        if self.learning_engine:
            # Q&A 쌍 생성
            qa_pairs = self._generate_qa_pairs(content)

            for question, answer in qa_pairs:
                await self.learning_engine.record_interaction(
                    prompt=question,
                    response=answer,
                    domain=content.topics[0] if content.topics else "web"
                )
                self._stats["total_learned"] += 1

        # 토픽 통계 업데이트
        for topic in content.topics[:3]:
            self._stats["topics_learned"][topic] = \
                self._stats["topics_learned"].get(topic, 0) + 1

    def _generate_qa_pairs(
        self,
        content: CrawledContent,
        max_pairs: int = 5
    ) -> List[Tuple[str, str]]:
        """콘텐츠에서 Q&A 쌍 생성"""
        pairs = []

        # 1. 제목 기반 질문
        if content.title:
            pairs.append((
                f"What is {content.title}?",
                content.summary or content.content[:500]
            ))

        # 2. 토픽 기반 질문
        for topic in content.topics[:2]:
            question = f"Tell me about {topic}."
            # 해당 토픽이 언급된 문장 찾기
            sentences = re.split(r'[.!?。]', content.content)
            relevant = [s.strip() for s in sentences if topic.lower() in s.lower()][:3]
            if relevant:
                pairs.append((question, '. '.join(relevant)))

        # 3. 요약 질문
        if content.summary:
            pairs.append((
                f"Summarize the content about {content.title or 'this topic'}",
                content.summary
            ))

        return pairs[:max_pairs]

    async def learn_topic(
        self,
        topic: str,
        max_pages: int = 20,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        특정 주제 학습

        Args:
            topic: 학습할 주제
            max_pages: 최대 크롤링 페이지 수
            sources: 시드 URL (없으면 검색 엔진 사용)

        Returns:
            학습 결과 통계
        """
        logger.info(f"Starting to learn about: {topic}")

        # 시드 URL 준비
        if sources:
            seed_urls = sources
        else:
            # 위키피디아 + 검색 기반
            seed_urls = [
                f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                f"https://ko.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            ]

        # 크롤러 설정
        self.crawler.config.max_pages = max_pages
        self.crawler.config.required_topics = [topic]

        start_time = time.time()
        learned_count = 0

        async def on_content(content: CrawledContent):
            nonlocal learned_count
            await self._process_content(content)
            learned_count += 1

        # 크롤링 및 학습
        contents = await self.crawler.crawl(
            seed_urls=seed_urls,
            topics=[topic],
            callback=on_content
        )

        elapsed = time.time() - start_time

        result = {
            "topic": topic,
            "pages_crawled": len(contents),
            "items_learned": learned_count,
            "elapsed_seconds": round(elapsed, 2),
            "avg_quality": sum(c.quality_score for c in contents) / len(contents) if contents else 0
        }

        self._stats["last_run"] = datetime.now().isoformat()
        self._save_stats()

        logger.info(f"Learning completed: {result}")
        return result

    async def continuous_learn(
        self,
        topics: List[str],
        interval_minutes: int = 60,
        pages_per_topic: int = 10
    ):
        """
        지속적 학습 실행

        Args:
            topics: 학습할 주제 목록
            interval_minutes: 학습 간격 (분)
            pages_per_topic: 주제당 크롤링 페이지 수
        """
        self._running = True
        cycle = 0

        logger.info(f"Starting continuous learning: {len(topics)} topics, "
                   f"interval={interval_minutes}min")

        while self._running:
            cycle += 1
            logger.info(f"Learning cycle {cycle}")

            for topic in topics:
                if not self._running:
                    break

                try:
                    await self.learn_topic(topic, max_pages=pages_per_topic)
                except Exception as e:
                    logger.error(f"Error learning {topic}: {e}")

                # 토픽 간 딜레이
                await asyncio.sleep(5)

            if self._running:
                logger.info(f"Cycle {cycle} completed. Waiting {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)

        logger.info("Continuous learning stopped")

    async def learn_from_sources(
        self,
        source_type: str = "tech",
        max_pages: int = 30
    ) -> Dict[str, Any]:
        """
        사전 정의된 소스에서 학습

        Args:
            source_type: 소스 유형 (wikipedia, tech, science, programming, korean)
            max_pages: 최대 페이지 수
        """
        if source_type not in self.KNOWLEDGE_SOURCES:
            raise ValueError(f"Unknown source type: {source_type}")

        seed_urls = self.KNOWLEDGE_SOURCES[source_type]
        logger.info(f"Learning from {source_type} sources...")

        self.crawler.config.max_pages = max_pages

        contents = await self.crawler.crawl(
            seed_urls=seed_urls,
            callback=self._process_content
        )

        return {
            "source_type": source_type,
            "pages_crawled": len(contents),
            "total_learned": self._stats["total_learned"]
        }

    def stop(self):
        """학습 중지"""
        self._running = False
        self.crawler.stop()

    def get_stats(self) -> Dict[str, Any]:
        """학습 통계"""
        return {
            **self._stats,
            "storage_path": str(self.storage_path),
            "top_topics": sorted(
                self._stats["topics_learned"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

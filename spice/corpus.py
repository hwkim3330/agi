"""
Browser Corpus - Web-based document collection for SPICE

Provides corpus grounding by:
1. Browsing real web pages via VLA agent
2. Extracting document content
3. Building a dynamic corpus for self-play
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

try:
    from playwright.async_api import async_playwright, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


@dataclass
class Document:
    """A document from the web corpus"""
    id: str
    url: str
    title: str
    content: str
    domain: str
    collected_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "domain": self.domain,
            "collected_at": self.collected_at,
            "metadata": self.metadata
        }


class BrowserCorpus:
    """
    Dynamic web corpus for SPICE training.

    Uses browser automation to:
    - Navigate to diverse web pages
    - Extract text content
    - Build a grounded corpus for task generation

    This provides the "corpus environment" that grounds
    the self-play loop and prevents hallucinations.
    """

    # Diverse seed URLs for exploration
    SEED_URLS = {
        "tech": [
            "https://news.ycombinator.com/newest",
            "https://lobste.rs",
            "https://www.reddit.com/r/programming/",
        ],
        "science": [
            "https://arxiv.org/list/cs.AI/recent",
            "https://arxiv.org/list/cs.LG/recent",
            "https://www.nature.com/subjects/computer-science",
        ],
        "news": [
            "https://www.bbc.com/news/technology",
            "https://techcrunch.com/",
        ],
        "wiki": [
            "https://en.wikipedia.org/wiki/Special:Random",
        ]
    }

    def __init__(self, data_dir: str = "data/corpus", max_documents: int = 1000):
        self.data_dir = data_dir
        self.max_documents = max_documents
        os.makedirs(data_dir, exist_ok=True)

        self.documents: Dict[str, Document] = {}
        self.visited_urls: Set[str] = set()
        self._doc_counter = 0

        self._browser = None
        self._page = None

        self._load()

    async def initialize(self, headless: bool = True):
        """Initialize browser for corpus collection"""
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright required: pip install playwright && playwright install chromium")

        playwright = await async_playwright().start()
        self._browser = await playwright.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await self._browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self._page = await context.new_page()
        print("[Corpus] Browser initialized")

    async def collect(self, category: str = "tech", count: int = 10) -> List[Document]:
        """
        Collect documents from web by browsing.

        Args:
            category: Category of URLs to start from
            count: Number of documents to collect

        Returns:
            List of collected Document objects
        """
        if not self._page:
            await self.initialize()

        seed_urls = self.SEED_URLS.get(category, self.SEED_URLS["tech"])
        collected = []

        for _ in range(count):
            # Pick random seed URL or follow link
            if random.random() < 0.3 or not collected:
                url = random.choice(seed_urls)
            else:
                # Try to find links on current page
                links = await self._extract_links()
                url = random.choice(links) if links else random.choice(seed_urls)

            # Skip if already visited
            if url in self.visited_urls:
                continue

            try:
                doc = await self._fetch_document(url)
                if doc and len(doc.content) > 200:  # Minimum content length
                    collected.append(doc)
                    self.documents[doc.id] = doc
                    self.visited_urls.add(url)
                    print(f"[Corpus] Collected: {doc.title[:50]}...")

            except Exception as e:
                print(f"[Corpus] Error fetching {url}: {e}")
                continue

            await asyncio.sleep(1)  # Rate limiting

        self._save()
        return collected

    async def _fetch_document(self, url: str) -> Optional[Document]:
        """Fetch and parse a web document"""
        try:
            await self._page.goto(url, timeout=15000)
            await asyncio.sleep(1)

            # Extract content
            title = await self._page.title()

            # Get main text content
            content = await self._page.evaluate("""
                () => {
                    // Remove scripts, styles, nav, footer
                    const remove = document.querySelectorAll('script, style, nav, footer, header, aside');
                    remove.forEach(el => el.remove());

                    // Get main content
                    const main = document.querySelector('main, article, .content, #content, .post');
                    if (main) return main.innerText;

                    // Fallback to body
                    return document.body.innerText;
                }
            """)

            # Clean content
            content = self._clean_content(content)

            if not content or len(content) < 100:
                return None

            self._doc_counter += 1
            domain = urlparse(url).netloc

            return Document(
                id=f"doc_{self._doc_counter}",
                url=url,
                title=title or "Untitled",
                content=content[:10000],  # Limit content size
                domain=domain,
                collected_at=datetime.now().isoformat(),
                metadata={"category": self._categorize_domain(domain)}
            )

        except Exception as e:
            print(f"[Corpus] Parse error: {e}")
            return None

    async def _extract_links(self) -> List[str]:
        """Extract links from current page"""
        try:
            links = await self._page.evaluate("""
                () => {
                    const anchors = document.querySelectorAll('a[href]');
                    return Array.from(anchors)
                        .map(a => a.href)
                        .filter(href => href.startsWith('http') && !href.includes('#'))
                        .slice(0, 20);
                }
            """)
            return links
        except:
            return []

    def _clean_content(self, text: str) -> str:
        """Clean extracted text content"""
        if not text:
            return ""

        # Remove excessive whitespace
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Remove very short lines (likely navigation)
        lines = [line for line in lines if len(line) > 20]

        return '\n'.join(lines)

    def _categorize_domain(self, domain: str) -> str:
        """Categorize domain"""
        domain = domain.lower()
        if 'arxiv' in domain:
            return 'academic'
        elif 'wiki' in domain:
            return 'encyclopedia'
        elif 'news' in domain or 'bbc' in domain:
            return 'news'
        elif 'github' in domain:
            return 'code'
        else:
            return 'general'

    def sample(self, n: int = 1, category: Optional[str] = None) -> List[Document]:
        """Sample random documents from corpus"""
        docs = list(self.documents.values())

        if category:
            docs = [d for d in docs if d.metadata.get("category") == category]

        if not docs:
            return []

        return random.sample(docs, min(n, len(docs)))

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def size(self) -> int:
        """Get corpus size"""
        return len(self.documents)

    def _save(self):
        """Save corpus to disk"""
        corpus_file = os.path.join(self.data_dir, "corpus.json")
        data = {
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
            "visited_urls": list(self.visited_urls),
            "doc_counter": self._doc_counter
        }
        with open(corpus_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load corpus from disk"""
        corpus_file = os.path.join(self.data_dir, "corpus.json")
        if os.path.exists(corpus_file):
            with open(corpus_file, "r") as f:
                data = json.load(f)

            for k, v in data.get("documents", {}).items():
                self.documents[k] = Document(**v)

            self.visited_urls = set(data.get("visited_urls", []))
            self._doc_counter = data.get("doc_counter", 0)

            print(f"[Corpus] Loaded {len(self.documents)} documents")

    async def close(self):
        """Close browser"""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BrowserCorpus({len(self.documents)} documents)"

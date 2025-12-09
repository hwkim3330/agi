#!/usr/bin/env python3
"""
AGI Trinity - Multimodal Processing Engine
멀티모달 처리 엔진

텍스트, 이미지, 오디오 등 다양한 모달리티 처리
"""
import asyncio
import base64
import io
import os
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from PIL import Image
import numpy as np


class ModalityType(Enum):
    """모달리티 유형"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    CODE = "code"


@dataclass
class ModalityInput:
    """모달리티 입력 데이터"""
    type: ModalityType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None

    def __post_init__(self):
        if self.hash is None:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """데이터 해시 계산"""
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()[:12]
        elif isinstance(self.data, bytes):
            return hashlib.md5(self.data).hexdigest()[:12]
        elif isinstance(self.data, Image.Image):
            return hashlib.md5(self.data.tobytes()).hexdigest()[:12]
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()[:12]


@dataclass
class ProcessedModality:
    """처리된 모달리티"""
    original: ModalityInput
    processed_data: Any
    features: Optional[np.ndarray] = None
    description: Optional[str] = None
    confidence: float = 1.0


class ImageProcessor:
    """
    이미지 처리기

    이미지 전처리, 변환, 특징 추출
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize

    def load_image(
        self,
        source: Union[str, bytes, Image.Image]
    ) -> Image.Image:
        """이미지 로드"""
        if isinstance(source, Image.Image):
            return source
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source))
        elif isinstance(source, str):
            if source.startswith(("http://", "https://")):
                return self._load_from_url(source)
            elif source.startswith("data:image"):
                return self._load_from_base64(source)
            else:
                return Image.open(source)
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")

    def _load_from_url(self, url: str) -> Image.Image:
        """URL에서 이미지 로드"""
        import urllib.request
        with urllib.request.urlopen(url) as response:
            data = response.read()
        return Image.open(io.BytesIO(data))

    def _load_from_base64(self, data_url: str) -> Image.Image:
        """Base64에서 이미지 로드"""
        header, data = data_url.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data))

    def preprocess(
        self,
        image: Image.Image,
        preserve_aspect: bool = True
    ) -> Image.Image:
        """이미지 전처리"""
        # RGB 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 리사이즈
        if preserve_aspect:
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        return image

    def to_tensor(self, image: Image.Image) -> np.ndarray:
        """이미지를 텐서로 변환"""
        arr = np.array(image, dtype=np.float32)

        if self.normalize:
            arr = arr / 255.0
            # ImageNet 정규화
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = (arr - mean) / std

        # CHW 포맷으로 변환
        arr = arr.transpose((2, 0, 1))

        return arr

    def to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """이미지를 Base64로 변환"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{data}"

    def extract_patches(
        self,
        image: Image.Image,
        patch_size: int = 64,
        stride: int = 32
    ) -> List[Image.Image]:
        """이미지에서 패치 추출"""
        width, height = image.size
        patches = []

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)

        return patches


class TextProcessor:
    """
    텍스트 처리기

    텍스트 정규화, 토큰화, 임베딩
    """

    def __init__(self, max_length: int = 2048):
        self.max_length = max_length

    def preprocess(self, text: str) -> str:
        """텍스트 전처리"""
        # 기본 정규화
        text = text.strip()
        text = " ".join(text.split())  # 공백 정규화
        return text

    def truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """텍스트 잘라내기"""
        max_len = max_length or self.max_length
        if len(text) > max_len:
            return text[:max_len - 3] + "..."
        return text

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """키워드 추출 (간단한 구현)"""
        # 불용어
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'and', 'but', 'or', 'for', 'with',
            'not', 'you', 'your', 'they', 'their', 'its', 'from', 'about',
            'to', 'of', 'in', 'on', 'at', 'by', 'as', 'it', 'we', 'i'
        }

        words = text.lower().split()
        word_freq = {}

        for word in words:
            # 영숫자만 유지
            word = ''.join(c for c in word if c.isalnum())
            if word and word not in stopwords and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]

    def detect_language(self, text: str) -> str:
        """언어 감지 (간단한 휴리스틱)"""
        # 한글 비율 체크
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        total_chars = len([c for c in text if c.isalpha()])

        if total_chars == 0:
            return "unknown"

        korean_ratio = korean_chars / total_chars

        if korean_ratio > 0.5:
            return "ko"
        else:
            return "en"


class CodeProcessor:
    """
    코드 처리기

    프로그래밍 코드 분석 및 처리
    """

    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin"
    }

    def detect_language(self, code: str, filename: Optional[str] = None) -> str:
        """프로그래밍 언어 감지"""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in self.LANGUAGE_EXTENSIONS:
                return self.LANGUAGE_EXTENSIONS[ext]

        # 휴리스틱 기반 감지
        if "def " in code and ":" in code:
            return "python"
        elif "function " in code or "const " in code or "let " in code:
            return "javascript"
        elif "public class" in code or "public static void main" in code:
            return "java"
        elif "#include" in code:
            return "cpp"
        elif "fn " in code and "->" in code:
            return "rust"
        elif "func " in code and "package " in code:
            return "go"

        return "unknown"

    def extract_functions(self, code: str, language: str = "python") -> List[Dict[str, str]]:
        """함수 추출"""
        functions = []

        if language == "python":
            import re
            pattern = r'def\s+(\w+)\s*\((.*?)\):'
            matches = re.finditer(pattern, code)
            for match in matches:
                functions.append({
                    "name": match.group(1),
                    "params": match.group(2),
                    "position": match.start()
                })

        return functions


class MultimodalEngine:
    """
    멀티모달 통합 엔진

    여러 모달리티를 통합 처리
    """

    def __init__(self, cache_dir: str = "~/.trinity/multimodal_cache"):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        self.code_processor = CodeProcessor()

        self._cache: Dict[str, ProcessedModality] = {}

    async def process(
        self,
        inputs: List[Union[str, bytes, Image.Image, ModalityInput]]
    ) -> List[ProcessedModality]:
        """
        입력 처리

        Args:
            inputs: 다양한 형식의 입력 리스트

        Returns:
            처리된 모달리티 리스트
        """
        results = []

        for inp in inputs:
            # ModalityInput으로 변환
            if isinstance(inp, ModalityInput):
                modality_input = inp
            else:
                modality_input = self._detect_modality(inp)

            # 캐시 확인
            if modality_input.hash in self._cache:
                results.append(self._cache[modality_input.hash])
                continue

            # 모달리티별 처리
            processed = await self._process_modality(modality_input)
            self._cache[modality_input.hash] = processed
            results.append(processed)

        return results

    def _detect_modality(self, data: Any) -> ModalityInput:
        """모달리티 자동 감지"""
        if isinstance(data, Image.Image):
            return ModalityInput(type=ModalityType.IMAGE, data=data)

        elif isinstance(data, bytes):
            # 이미지인지 확인
            try:
                img = Image.open(io.BytesIO(data))
                return ModalityInput(type=ModalityType.IMAGE, data=img)
            except Exception:
                pass
            return ModalityInput(type=ModalityType.TEXT, data=data.decode(errors='ignore'))

        elif isinstance(data, str):
            # URL 또는 파일 경로
            if data.startswith(("http://", "https://", "data:image")):
                return ModalityInput(
                    type=ModalityType.IMAGE,
                    data=data,
                    metadata={"source": "url"}
                )

            # 파일 경로
            if os.path.isfile(data):
                ext = Path(data).suffix.lower()
                if ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]:
                    return ModalityInput(type=ModalityType.IMAGE, data=data)
                elif ext in self.code_processor.LANGUAGE_EXTENSIONS:
                    with open(data, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return ModalityInput(
                        type=ModalityType.CODE,
                        data=content,
                        metadata={"filename": data, "language": self.code_processor.LANGUAGE_EXTENSIONS.get(ext)}
                    )

            # 코드 감지
            if any(keyword in data for keyword in ["def ", "function ", "class ", "#include", "import "]):
                return ModalityInput(type=ModalityType.CODE, data=data)

            # 기본 텍스트
            return ModalityInput(type=ModalityType.TEXT, data=data)

        raise ValueError(f"Cannot detect modality for type: {type(data)}")

    async def _process_modality(self, modality: ModalityInput) -> ProcessedModality:
        """모달리티별 처리"""
        if modality.type == ModalityType.IMAGE:
            return await self._process_image(modality)
        elif modality.type == ModalityType.TEXT:
            return await self._process_text(modality)
        elif modality.type == ModalityType.CODE:
            return await self._process_code(modality)
        else:
            # 기본 처리
            return ProcessedModality(
                original=modality,
                processed_data=modality.data
            )

    async def _process_image(self, modality: ModalityInput) -> ProcessedModality:
        """이미지 처리"""
        # 이미지 로드 및 전처리
        image = self.image_processor.load_image(modality.data)
        processed = self.image_processor.preprocess(image)

        # 특징 추출 (텐서 변환)
        features = self.image_processor.to_tensor(processed)

        return ProcessedModality(
            original=modality,
            processed_data=processed,
            features=features,
            description=f"Image {processed.size[0]}x{processed.size[1]}",
            confidence=1.0
        )

    async def _process_text(self, modality: ModalityInput) -> ProcessedModality:
        """텍스트 처리"""
        text = modality.data if isinstance(modality.data, str) else str(modality.data)
        processed = self.text_processor.preprocess(text)
        keywords = self.text_processor.extract_keywords(processed)
        language = self.text_processor.detect_language(processed)

        return ProcessedModality(
            original=modality,
            processed_data=processed,
            description=f"Text ({language}, {len(processed)} chars)",
            confidence=1.0
        )

    async def _process_code(self, modality: ModalityInput) -> ProcessedModality:
        """코드 처리"""
        code = modality.data
        language = modality.metadata.get("language") or self.code_processor.detect_language(code)
        functions = self.code_processor.extract_functions(code, language)

        return ProcessedModality(
            original=modality,
            processed_data=code,
            description=f"Code ({language}, {len(functions)} functions)",
            confidence=1.0
        )

    async def create_unified_prompt(
        self,
        text_prompt: str,
        processed_modalities: List[ProcessedModality]
    ) -> Dict[str, Any]:
        """
        통합 프롬프트 생성

        LFM2-VL 모델용 대화 형식으로 변환
        """
        content = []

        # 이미지 먼저 추가
        for pm in processed_modalities:
            if pm.original.type == ModalityType.IMAGE:
                if isinstance(pm.processed_data, Image.Image):
                    content.append({
                        "type": "image",
                        "image": pm.processed_data
                    })

        # 텍스트/코드 컨텍스트 추가
        context_parts = []
        for pm in processed_modalities:
            if pm.original.type == ModalityType.CODE:
                lang = pm.original.metadata.get("language", "code")
                context_parts.append(f"```{lang}\n{pm.processed_data[:2000]}\n```")
            elif pm.original.type == ModalityType.TEXT and pm.processed_data != text_prompt:
                context_parts.append(pm.processed_data[:1000])

        # 최종 텍스트 프롬프트
        if context_parts:
            full_prompt = "\n\n".join(context_parts) + "\n\n" + text_prompt
        else:
            full_prompt = text_prompt

        content.append({
            "type": "text",
            "text": full_prompt
        })

        return {
            "role": "user",
            "content": content
        }

    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()

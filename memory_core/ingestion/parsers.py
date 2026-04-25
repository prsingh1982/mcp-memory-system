"""Parsers for supported local source formats."""

from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser, Parser
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from memory_core.domain.enums import SourceType
from memory_core.domain.models import SourceReference
from memory_core.interfaces.parsing import DocumentParser

try:
    from docx import Document
except ImportError:  # pragma: no cover - handled at runtime
    Document = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime
    PdfReader = None


class ParserRegistry:
    """Routes sources to the first parser that supports their type."""

    def __init__(self, parsers: Iterable[DocumentParser]) -> None:
        self._parsers = list(parsers)

    def parse(self, source: SourceReference) -> str:
        for parser in self._parsers:
            if parser.supports(source.source_type):
                return parser.parse(source)
        raise ValueError(f"No parser registered for source type: {source.source_type}")


class _BaseFileParser(DocumentParser):
    """Base helper for parsers that read from source metadata or file paths."""

    source_types: tuple[SourceType, ...] = ()

    def supports(self, source_type: SourceType) -> bool:
        return source_type in self.source_types

    @staticmethod
    def _read_text(source: SourceReference) -> str:
        raw_text = source.metadata.get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            return raw_text

        if not source.file_path:
            raise ValueError(f"Source {source.source_id} has no file_path or raw_text")

        path = Path(source.file_path)
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode text source: {path}")

    @staticmethod
    def _read_bytes(source: SourceReference) -> bytes:
        raw_bytes = source.metadata.get("raw_bytes")
        if isinstance(raw_bytes, bytes) and raw_bytes:
            return raw_bytes

        if not source.file_path:
            raise ValueError(f"Source {source.source_id} has no file_path or raw_bytes")

        return Path(source.file_path).read_bytes()


class PlainTextParser(_BaseFileParser):
    source_types = (SourceType.TEXT,)

    def parse(self, source: SourceReference) -> str:
        return self._read_text(source).strip()


class MarkdownParser(_BaseFileParser):
    source_types = (SourceType.MARKDOWN,)

    def parse(self, source: SourceReference) -> str:
        text = self._read_text(source)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*_`>#-]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


class _HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_tag_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignored_tag_depth += 1
        if tag in {"p", "div", "br", "li", "section", "article", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_tag_depth > 0:
            self._ignored_tag_depth -= 1
        if tag in {"p", "div", "li", "section", "article"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_tag_depth == 0 and data.strip():
            self._parts.append(data.strip())
            self._parts.append(" ")

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(html)
    return parser.get_text()


class WebPageParser(_BaseFileParser):
    source_types = (SourceType.WEB_PAGE,)

    def parse(self, source: SourceReference) -> str:
        html = self._read_text(source)
        return _html_to_text(html)


class EmailParser(_BaseFileParser):
    source_types = (SourceType.EMAIL,)

    def parse(self, source: SourceReference) -> str:
        message = self._load_message(source)
        headers = []
        for header_name in ("subject", "from", "to", "date"):
            header_value = message.get(header_name)
            if header_value:
                headers.append(f"{header_name.title()}: {header_value}")

        body = self._extract_body(message).strip()
        combined = "\n".join(headers + ["", body]).strip()
        return combined

    def _load_message(self, source: SourceReference):
        if source.file_path:
            raw_bytes = self._read_bytes(source)
            return BytesParser(policy=policy.default).parsebytes(raw_bytes)
        raw_text = self._read_text(source)
        return Parser(policy=policy.default).parsestr(raw_text)

    def _extract_body(self, message) -> str:
        if message.is_multipart():
            text_parts: list[str] = []
            html_parts: list[str] = []
            for part in message.walk():
                content_type = part.get_content_type()
                if part.get_content_disposition() == "attachment":
                    continue
                payload = part.get_content()
                if not isinstance(payload, str):
                    continue
                if content_type == "text/plain":
                    text_parts.append(payload)
                elif content_type == "text/html":
                    html_parts.append(_html_to_text(payload))
            if text_parts:
                return "\n\n".join(text_parts)
            if html_parts:
                return "\n\n".join(html_parts)
            return ""

        payload = message.get_content()
        if not isinstance(payload, str):
            return ""
        if message.get_content_type() == "text/html":
            return _html_to_text(payload)
        return payload


class PdfParser(_BaseFileParser):
    source_types = (SourceType.PDF,)

    def parse(self, source: SourceReference) -> str:
        if PdfReader is None:
            raise ImportError("pypdf is required to parse PDF sources.")
        if not source.file_path:
            raise ValueError(f"Source {source.source_id} has no file_path for PDF parsing")

        reader = PdfReader(source.file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page.strip() for page in pages if page.strip())


class DocxParser(_BaseFileParser):
    source_types = (SourceType.DOCX,)

    def parse(self, source: SourceReference) -> str:
        if Document is None:
            raise ImportError("python-docx is required to parse DOCX sources.")
        if not source.file_path:
            raise ValueError(f"Source {source.source_id} has no file_path for DOCX parsing")

        document = Document(source.file_path)
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n\n".join(paragraphs)

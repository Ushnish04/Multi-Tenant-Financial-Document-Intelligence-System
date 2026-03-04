"""
app/services/rag/chunker.py
Token-based semantic chunking with overlap.
Uses tiktoken for accurate token counting compatible with OpenAI models.
"""
import re
from dataclasses import dataclass

import tiktoken

from app.core.config import get_settings

settings = get_settings()

# Use cl100k_base encoder (compatible with GPT-4 and text-embedding-3-*)
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    page_hint: int | None = None


class SemanticChunker:
    """
    Token-aware chunker that splits document text into overlapping chunks.
    Respects sentence boundaries when possible to preserve semantic context.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE_TOKENS,
        overlap: int = settings.CHUNK_OVERLAP_TOKENS,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations to avoid false sentence breaks
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|Inc|LLC|Ltd)\.\s', r'\1<DOT> ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Restore abbreviation dots
        return [s.replace('<DOT> ', '. ') for s in sentences if s.strip()]

    def chunk(self, text: str, page_map: dict[int, int] | None = None) -> list[TextChunk]:
        """
        Chunk text into overlapping token windows.
        
        Args:
            text: Full document text
            page_map: Optional {char_offset: page_number} for page attribution
        
        Returns:
            Ordered list of TextChunk objects
        """
        sentences = self._split_sentences(text)
        chunks: list[TextChunk] = []
        current_tokens: list[int] = []
        current_text_parts: list[str] = []
        chunk_index = 0
        char_cursor = 0

        for sentence in sentences:
            sentence_tokens = _ENCODER.encode(sentence)

            # If single sentence exceeds chunk size, split by tokens directly
            if len(sentence_tokens) > self.chunk_size:
                # Flush current buffer first
                if current_tokens:
                    chunk_text = " ".join(current_text_parts)
                    chunks.append(TextChunk(
                        index=chunk_index,
                        text=chunk_text.strip(),
                        token_count=len(current_tokens),
                        char_start=char_cursor - len(chunk_text),
                        char_end=char_cursor,
                    ))
                    chunk_index += 1
                    # Carry over overlap
                    overlap_tokens = current_tokens[-self.overlap:]
                    current_tokens = list(overlap_tokens)
                    current_text_parts = [_ENCODER.decode(overlap_tokens)]

                # Split long sentence into windows
                for i in range(0, len(sentence_tokens), self.chunk_size - self.overlap):
                    window = sentence_tokens[i:i + self.chunk_size]
                    window_text = _ENCODER.decode(window)
                    chunks.append(TextChunk(
                        index=chunk_index,
                        text=window_text.strip(),
                        token_count=len(window),
                        char_start=char_cursor,
                        char_end=char_cursor + len(window_text),
                    ))
                    chunk_index += 1
                    char_cursor += len(window_text)
                continue

            # Normal path: accumulate sentences into chunk
            if len(current_tokens) + len(sentence_tokens) > self.chunk_size:
                # Emit current chunk
                chunk_text = " ".join(current_text_parts)
                chunks.append(TextChunk(
                    index=chunk_index,
                    text=chunk_text.strip(),
                    token_count=len(current_tokens),
                    char_start=char_cursor - len(chunk_text),
                    char_end=char_cursor,
                ))
                chunk_index += 1

                # Overlap: keep last N tokens
                overlap_token_count = 0
                overlap_parts: list[str] = []
                for part in reversed(current_text_parts):
                    part_tokens = _ENCODER.encode(part)
                    if overlap_token_count + len(part_tokens) <= self.overlap:
                        overlap_parts.insert(0, part)
                        overlap_token_count += len(part_tokens)
                    else:
                        break
                current_text_parts = overlap_parts
                current_tokens = _ENCODER.encode(" ".join(current_text_parts))

            current_text_parts.append(sentence)
            current_tokens = _ENCODER.encode(" ".join(current_text_parts))
            char_cursor += len(sentence) + 1

        # Emit final chunk
        if current_text_parts:
            chunk_text = " ".join(current_text_parts)
            chunks.append(TextChunk(
                index=chunk_index,
                text=chunk_text.strip(),
                token_count=len(current_tokens),
                char_start=char_cursor - len(chunk_text),
                char_end=char_cursor,
            ))

        return chunks

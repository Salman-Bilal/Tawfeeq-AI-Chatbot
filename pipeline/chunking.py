"""
Token-accurate, sentence-aware chunking.

WHY THIS EXISTS
The tafsir entries in this dataset range up to ~3,800 words. Embedding a
record that long means the model silently truncates it and ~90% of the
commentary never makes it into the vector. This module splits long text
into focused, overlapping chunks that fit comfortably inside the model's
context window, so nothing is lost and each vector stays sharp.

Approach:
  1. Split into sentences on Urdu/Arabic/Latin sentence enders.
  2. Greedily pack whole sentences into a chunk until the next sentence
     would exceed MAX_CHUNK_TOKENS (measured with the real tokenizer).
  3. Carry the trailing sentences forward as overlap so context is not
     cut mid-thought between adjacent chunks.
A single sentence longer than the limit (rare) is hard-split on spaces.
"""

import re
from typing import Callable, List

# Sentence enders: Urdu full stop (۔), Urdu/Arabic question mark (؟),
# Arabic semicolon (؛), plus Latin . ? ! — split on the whitespace that
# follows one of them, or on any run of newlines.
_SENT_SPLIT = re.compile(r"(?<=[۔؟؛!?\.])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _hard_split(sentence: str, count_tokens: Callable[[str], int], max_tokens: int) -> List[str]:
    """Split an over-long single sentence on whitespace into token-bounded pieces."""
    words = sentence.split()
    pieces, cur, cur_tok = [], [], 0
    for w in words:
        wt = count_tokens(w) or 1
        if cur and cur_tok + wt > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tok = [], 0
        cur.append(w)
        cur_tok += wt
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _overlap_tail(sentences: List[str], count_tokens: Callable[[str], int], overlap_tokens: int):
    """Return the trailing sentences whose combined size is ~overlap_tokens."""
    tail, tok = [], 0
    for s in reversed(sentences):
        st = count_tokens(s)
        if tail and tok + st > overlap_tokens:
            break
        tail.insert(0, s)
        tok += st
    return tail, tok


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 400,
    overlap_tokens: int = 80,
    min_chars: int = 15,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # Fast path: short enough to embed as-is.
    if count_tokens(text) <= max_tokens:
        return [text]

    # Normalise sentences, pre-splitting any that are individually too long.
    units: List[str] = []
    for s in split_sentences(text):
        if count_tokens(s) <= max_tokens:
            units.append(s)
        else:
            units.extend(_hard_split(s, count_tokens, max_tokens))

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for unit in units:
        ut = count_tokens(unit)
        if cur and cur_tok + ut > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_tok = _overlap_tail(cur, count_tokens, overlap_tokens)
        cur.append(unit)
        cur_tok += ut
    if cur:
        chunks.append(" ".join(cur))

    cleaned = [c for c in chunks if len(c) >= min_chars]
    return cleaned or [text]

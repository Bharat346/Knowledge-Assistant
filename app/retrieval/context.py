from __future__ import annotations

import re


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def select_context(results: list[dict], max_tokens: int) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    selected: list[dict] = []
    used_ids: set[str] = set()
    tokens_used = 0
    for index, result in enumerate(results, start=1):
        if result["chunk_id"] in used_ids:
            continue
        metadata = result.get("metadata", {})
        source = metadata.get("source") or metadata.get("filename") or "unknown source"
        page = metadata.get("page")
        reference = f"[S{index}] source={source}, page={page if page is not None else 'n/a'}, chunk={result['chunk_id']}"
        block = f"{reference}\n{result['text']}"
        block_tokens = _token_count(block)
        if tokens_used + block_tokens > max_tokens:
            continue
        blocks.append(block)
        selected.append(result)
        used_ids.add(result["chunk_id"])
        tokens_used += block_tokens
    return "\n\n".join(blocks), selected


def build_context(results: list[dict], max_tokens: int) -> str:
    return select_context(results, max_tokens)[0]
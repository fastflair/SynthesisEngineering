"""
comic_book_token_budget.py
==========================

Centralised token-budget enforcement for the comic-book / novel pipeline.

WHY THIS EXISTS
---------------
Several passes assemble an LLM prompt out of free-form fields (panel
descriptions, scene-anchor setting text, per-character costume locks, continuity
summaries, few-shot examples...). Any one of those fields can balloon when an
upstream LLM returns a runaway blob, and nothing downstream clamps it. The
result is the 400 ``maximum prompt length`` / 413 ``Payload Too Large`` errors
seen in production, where a single per-panel "refinement" call ships 1.5M-6M
tokens to a model whose ceiling is 1M.

This module gives every prompt builder a cheap, dependency-free way to:

  * estimate the token cost of a string (tiktoken if available, else chars/4),
  * clamp an individual field to a sub-budget WITHOUT cutting mid-word,
  * assemble a multi-section prompt under a single hard total budget, trimming
    the lowest-priority sections first and reporting what it cut.

It is deliberately self-contained (only stdlib) so it can be imported from both
``novel_generator`` and ``comic_book_generator`` without creating an import
cycle, and so it works whether or not ``tiktoken`` is installed.

The numbers here are intentionally conservative: every limit sits far below the
1M model ceiling, because the whole point is "minimal context per subtask"
(prev page + current beat + next page + the relevant anchors) rather than
"as much as the model will physically accept".
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable budgets (override via env var or by editing here).
# ---------------------------------------------------------------------------
# Hard ceiling for ANY single LLM call, enforced at the get_openai_prompt_response
# choke point. Sits far under the 1M model limit so even a pathological prompt
# can never 400/413 again.
GLOBAL_MAX_PROMPT_TOKENS: int = int(os.getenv("LLM_GLOBAL_MAX_PROMPT_TOKENS", "200000"))

# Target size for ONE per-panel prompt-refinement call. This is the "ralph loop"
# subtask budget: a single panel only ever needs its own brief plus a thin slice
# of neighbouring context, which is a few thousand tokens, not a few million.
REFINE_CALL_TOKEN_BUDGET: int = int(os.getenv("LLM_REFINE_CALL_TOKEN_BUDGET", "24000"))

# Per-section sub-budgets inside the refinement prompt. They sum to well under
# REFINE_CALL_TOKEN_BUDGET, leaving headroom for the fixed instruction scaffold.
SECTION_BUDGETS: Dict[str, int] = {
    "components": 6000,     # the assembled brief (subject/clothing/action/...)
    "summary": 800,         # the one-line panel summary
    "continuity": 5000,     # prev page / next page / anchors / scene lock
    "few_shot": 3000,       # style examples
    "story_context": 800,   # genre / mood / themes / char list
    "single_field": 1800,   # any one continuity sub-field (a setting blurb, etc.)
    # objects is now split into two independently-budgeted sections so
    # world_state (dynamic, high-continuity-value) is preserved at higher
    # priority than visual_bible (static reference, lower priority) when
    # the total objects budget is tight.
    "visual_bible": 1200,   # SeriesVisualBible canonical prop/costume entries
    "world_state": 800,     # WorldContinuityTracker current object appearances
}

# Marker inserted where text was cut, so a human reading a manifest can see it.
TRUNCATION_MARKER = " …[trimmed]"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    """Lazily load a tiktoken encoder if the library is present; else None."""
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        import tiktoken  # type: ignore
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        _ENCODER = None
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """Best-effort token count.

    Uses tiktoken when available (accurate), otherwise a chars/4 heuristic that
    slightly OVER-estimates for English prose — which is the safe direction for
    a budget guard. Empty / non-str inputs cost 0.
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: ~4 chars per token, rounded up, with a small floor so
    # very short strings never read as 0 tokens.
    return max(1, (len(text) + 3) // 4)


def tokens_to_chars(tokens: int) -> int:
    """Rough inverse of the heuristic, used when we need a char budget."""
    return max(0, tokens) * 4


# ---------------------------------------------------------------------------
# Clamping a single field
# ---------------------------------------------------------------------------
def clamp_text(text: str, max_tokens: int, *, where: str = "") -> str:
    """Return ``text`` trimmed so it costs at most ``max_tokens`` tokens.

    Trims on a whitespace boundary where possible (no mid-word cuts) and appends
    a visible marker. Logs a warning naming ``where`` so an oversized field is
    discoverable in logs — this is how you find the real culprit in your data.
    """
    if not text:
        return text or ""
    if not isinstance(text, str):
        text = str(text)
    cost = estimate_tokens(text)
    if cost <= max_tokens:
        return text

    # Binary-search the longest prefix that fits, then back off to a word break.
    target_chars = tokens_to_chars(max_tokens) - len(TRUNCATION_MARKER)
    target_chars = max(0, target_chars)
    cut = text[:target_chars]
    # Tighten if the encoder disagrees with the heuristic (tiktoken path).
    while cut and estimate_tokens(cut + TRUNCATION_MARKER) > max_tokens:
        cut = cut[: int(len(cut) * 0.9)]
    # Prefer a clean word boundary in the last 20% of the kept text.
    sp = cut.rfind(" ", int(len(cut) * 0.8))
    if sp > 0:
        cut = cut[:sp]

    logger.warning(
        "[TokenBudget] Clamped %s from ~%d to ~%d tokens (limit %d). "
        "An upstream field is oversized; inspect it.",
        where or "a field", cost, estimate_tokens(cut + TRUNCATION_MARKER),
        max_tokens,
    )
    return cut.rstrip() + TRUNCATION_MARKER


def clamp_components(components: Dict[str, str],
                     per_section: Optional[int] = None,
                     total: Optional[int] = None) -> Dict[str, str]:
    """Clamp each value of an assembled-brief dict, then enforce a total.

    Each section is first clamped to ``per_section`` (default SECTION_BUDGETS
    single_field). If the whole dict still exceeds ``total`` (default
    SECTION_BUDGETS['components']), sections are trimmed proportionally, longest
    first, until it fits. Keys/structure are preserved so downstream merge logic
    is unaffected.
    """
    if not isinstance(components, dict):
        return components
    per_section = per_section or SECTION_BUDGETS["single_field"]
    total = total or SECTION_BUDGETS["components"]

    out = {
        k: clamp_text(str(v or ""), per_section, where=f"components.{k}")
        for k, v in components.items()
    }
    if sum(estimate_tokens(v) for v in out.values()) <= total:
        return out

    # Still too big collectively — trim the largest sections down until it fits.
    while sum(estimate_tokens(v) for v in out.values()) > total:
        # Find the currently-largest section and halve its budget.
        biggest_key = max(out, key=lambda k: estimate_tokens(out[k]))
        biggest_cost = estimate_tokens(out[biggest_key])
        if biggest_cost <= 1:
            break
        out[biggest_key] = clamp_text(
            out[biggest_key], max(1, biggest_cost // 2),
            where=f"components.{biggest_key}(total-trim)",
        )
    return out


# ---------------------------------------------------------------------------
# Assembling a whole prompt under a hard budget
# ---------------------------------------------------------------------------
class Section:
    """One labelled, priority-ranked piece of a prompt.

    priority: lower number = more important = trimmed LAST.
    min_tokens: never trim this section below this floor (0 = may drop entirely).
    """

    __slots__ = ("name", "text", "priority", "min_tokens")

    def __init__(self, name: str, text: str, priority: int = 100,
                 min_tokens: int = 0):
        self.name = name
        self.text = text or ""
        self.priority = priority
        self.min_tokens = min_tokens


def assemble_within_budget(sections: List[Section],
                           max_tokens: int,
                           joiner: str = "\n\n") -> Tuple[str, Dict[str, int]]:
    """Join ``sections`` into one string costing <= ``max_tokens`` tokens.

    Strategy: keep the highest-priority sections intact; trim or drop the
    lowest-priority ones first. Returns (assembled_text, report) where ``report``
    maps section name -> final token cost (0 = dropped), useful for diagnostics.
    """
    # Sort a working copy by priority ASC (most important first).
    work = sorted(sections, key=lambda s: s.priority)
    joiner_cost = estimate_tokens(joiner)

    # Greedily admit sections in priority order, clamping each to remaining room.
    kept: List[Tuple[Section, str]] = []
    used = 0
    report: Dict[str, int] = {s.name: 0 for s in sections}
    for s in work:
        if not s.text.strip():
            continue
        room = max_tokens - used - (joiner_cost if kept else 0)
        if room <= max(1, s.min_tokens):
            # No space left for even the floor — drop this (lower-priority) one.
            continue
        clamped = clamp_text(s.text, room, where=f"section.{s.name}")
        if estimate_tokens(clamped) < s.min_tokens and s.min_tokens > 0:
            # Couldn't honour the floor; skip rather than emit a stub.
            continue
        kept.append((s, clamped))
        used += estimate_tokens(clamped) + (joiner_cost if len(kept) > 1 else 0)
        report[s.name] = estimate_tokens(clamped)

    # Re-emit in ORIGINAL order (priority controlled trimming, not ordering).
    kept_by_name = {s.name: txt for (s, txt) in kept}
    ordered = [kept_by_name[s.name] for s in sections if kept_by_name.get(s.name)]
    return joiner.join(ordered), report


# ---------------------------------------------------------------------------
# Global guard for the single LLM choke point
# ---------------------------------------------------------------------------
def guard_prompt(prompt: str,
                 max_tokens: int = GLOBAL_MAX_PROMPT_TOKENS,
                 *, origin: str = "") -> str:
    """Last-resort clamp applied right before an API call.

    Middle-truncates (keeps the head and tail, drops the bloated middle) because
    instructions usually live at the top and the ask usually lives at the
    bottom; a runaway blob is almost always in the body. No-op when the prompt
    already fits.
    """
    cost = estimate_tokens(prompt)
    if cost <= max_tokens:
        return prompt

    notice = (
        "\n\n[NOTE: a large block of context was omitted here to fit the model's "
        "limit; the instructions above and the request below are intact.]\n\n"
    )
    # Reserve room for the notice itself so the final result stays under budget.
    body_budget = max(1, max_tokens - estimate_tokens(notice))
    head_budget = int(body_budget * 0.6)
    tail_budget = body_budget - head_budget
    head = clamp_text(prompt, head_budget, where=f"{origin or 'prompt'}:head")
    tail_chars = tokens_to_chars(tail_budget)
    tail = prompt[-tail_chars:] if tail_chars > 0 else ""
    sp = tail.find(" ")
    if 0 < sp < 40:
        tail = tail[sp + 1:]

    logger.error(
        "[TokenBudget] GLOBAL GUARD tripped for %s: prompt was ~%d tokens, "
        "hard-capped to ~%d. This means a builder upstream failed to bound its "
        "context — check the warnings above for the oversized field.",
        origin or "an LLM call", cost, max_tokens,
    )
    result = head + notice + tail
    # Final backstop: if the heuristic mis-estimated, clamp the whole thing.
    if estimate_tokens(result) > max_tokens:
        result = clamp_text(result, max_tokens, where=f"{origin or 'prompt'}:final")
    return result


# ---------------------------------------------------------------------------
# Token-limit error detection (for shrink-instead-of-retry)
# ---------------------------------------------------------------------------
_TOKEN_ERROR_SIGNATURES = (
    "maximum prompt length",
    "maximum context length",
    "payload too large",
    "413",
    "context_length_exceeded",
    "too many tokens",
    "request contains",
    "string too long",
)


def is_token_limit_error(err: object) -> bool:
    """True when an exception/string indicates a prompt-size rejection.

    These are DETERMINISTIC: re-sending the same payload will fail identically,
    so the caller should shrink the prompt rather than sleep-and-retry.
    """
    msg = str(err).lower()
    return any(sig in msg for sig in _TOKEN_ERROR_SIGNATURES)

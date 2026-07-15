"""
comic_book_repetition.py
========================
Deterministic anti-repetition guard for the comic / graphic-novel pipeline.

WHY THIS EXISTS
---------------
Several independent passes add spoken text to panels — the base script
generator, the intro-caption injector, the dialogue reviewer, and the narration
weave — and none of them compares the *text* it emits against what an adjacent
panel already says. The narration weave feeds the LLM prior captions only as
soft "maintain voice" context (and only the captions written *before* its first
target, so captions produced together in one call are never cross-checked), and
nothing hard-stops a verbatim repeat. The visible symptom is a caption like

    "Historic hotel. Twilight. Lights remember."

landing on one panel and then again on the very next one. A reader does not want
to be told the same thing twice in a row; it is better to say something new or
nothing at all and let the art carry the beat.

This module is the deterministic safeguard the pipeline was missing. It walks
the finished script in reading order and, for each spoken line, checks whether a
*recent* line of the same kind already said essentially the same thing. When it
finds an unintentional repeat it removes the later copy (the first occurrence
already did the job) and lets the panel breathe — recomputing dialogue_density
so the renderer knows the panel is now lighter or silent.

WHAT IT DELIBERATELY DOES NOT TOUCH
-----------------------------------
Repetition is a real literary device in this codebase (elegiac refrains,
recontextualising echoes, and the plant-and-payoff callbacks written by
``weave_dialogue_callbacks``). This guard protects all of those:

  * DISTANCE GATE — it only removes a repeat whose earlier twin is *nearby*
    (a handful of panels). A line that returns thirty panels later at the climax
    is a callback, not a bug, and is left alone.
  * TAG EXEMPTION — any line marked intentional (``_intentional_echo``,
    ``_callback``, ``_refrain``, ``intentional_repeat``) is never removed.
  * REFRAIN ALLOWLIST — a book-level signature/refrain declared in story_dna is
    exempt from exact-match removal wherever it recurs.
  * CATCHPHRASE ALLOWLIST — pass ``voice_profiles`` and each character's own
    declared catchphrases are exempt too (scoped to that speaker only) — a
    catchphrase recurring is the whole point of it. Whether a catchphrase is
    recurring too MUCH across the whole book is a different question, answered
    by comic_book_dialogue_director's signature-usage check, not this module.
  * ORDERING — run this BEFORE ``weave_dialogue_callbacks`` so deliberately
    planted echoes are added afterwards and never seen by the guard.

Design mirrors the other deterministic modules: stdlib only (``difflib`` +
``re``), no ML stack required, and every step fails soft — on any surprise it
leaves the script unchanged rather than risking damage.

Integration (in comic_book_generator, after weave_narration_pass and before
weave_dialogue_callbacks):

    from comic_book_repetition import dedupe_repeated_dialogue
    script, _rep = dedupe_repeated_dialogue(
        script, story_dna=project.story_dna, working_dir=project.working_dir,
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable thresholds (override via env or by editing here). Conservative on
# purpose: the cost of a missed repeat is small; the cost of eating an
# intentional line is large.
# ---------------------------------------------------------------------------

# How many panels back a NEAR-duplicate still counts as an unintentional repeat.
# Beyond this a recurrence is treated as a deliberate echo/callback and kept.
NEAR_WINDOW: int = int(os.getenv("REP_NEAR_WINDOW", "6"))

# How many panels back an EXACT (normalised-identical) repeat is still caught.
# A little wider than NEAR_WINDOW: a verbatim narration repeat within a scene is
# almost always a bug even at a slightly greater distance.
EXACT_WINDOW: int = int(os.getenv("REP_EXACT_WINDOW", "12"))

# Character dialogue is deduped more cautiously than narration (a person can
# repeat themselves on purpose), so it uses a tiny window and exact-only match.
CHAR_WINDOW: int = int(os.getenv("REP_CHAR_WINDOW", "3"))

# Similarity gates for a NEAR (not identical) narration duplicate. A repeat has
# to clear these to be removed — genuine variation ("say something new") sits
# well below them and survives.
SIM_RATIO: float = float(os.getenv("REP_SIM_RATIO", "0.82"))   # difflib ratio
SIM_JACCARD: float = float(os.getenv("REP_SIM_JACCARD", "0.72"))  # token overlap

# Very short lines ("Yes.", "No!", "Run!") are excluded from NEAR matching — a
# two-word exclamation recurring is usually rhythm, not a bug. Exact repeats of
# them are still caught within the tighter windows.
MIN_TOKENS_FOR_NEAR: int = int(os.getenv("REP_MIN_TOKENS_FOR_NEAR", "4"))

# Line keys that mark a line as an intentional recurrence — never removed.
_INTENTIONAL_KEYS = (
    "_intentional_echo", "_callback", "_refrain",
    "intentional_repeat", "_is_echo", "_planted_echo",
)


# ---------------------------------------------------------------------------
# Text normalisation & similarity
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")
# Tiny stop set so "the lights remember" ~ "lights remember" collapse together.
_STOP = frozenset((
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "is",
    "was", "were", "be", "it", "its", "this", "that", "with", "for", "as",
))


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation/ellipses, collapse whitespace."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower().replace("…", " ")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _content_tokens(norm: str) -> Tuple[str, ...]:
    """Content words (stop-words removed) for Jaccard overlap."""
    return tuple(w for w in _WORD_RE.findall(norm) if w not in _STOP)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _is_near_duplicate(norm_a: str, toks_a: Tuple[str, ...],
                       norm_b: str, toks_b: Tuple[str, ...]) -> bool:
    """True when two normalised lines say essentially the same thing.

    Catches (1) exact normalised equality, (2) full containment of the shorter
    content in the longer, and (3) high difflib ratio *and* high token overlap.
    Requiring BOTH ratio and overlap for case (3) keeps genuine variation safe.
    """
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True

    sa, sb = set(toks_a), set(toks_b)
    # Containment: one line's content words are a subset of the other's and it is
    # at least half its length (a caption that merely re-states a prior one).
    if sa and sb:
        smaller, larger = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        if smaller and smaller <= larger and len(smaller) >= max(3, len(larger) // 2):
            return True

    if len(toks_a) < MIN_TOKENS_FOR_NEAR or len(toks_b) < MIN_TOKENS_FOR_NEAR:
        return False   # too short to judge as a reworded repeat

    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if ratio >= SIM_RATIO and _jaccard(sa, sb) >= SIM_JACCARD:
        return True
    return False


# ---------------------------------------------------------------------------
# Line classification helpers
# ---------------------------------------------------------------------------
def _is_narration(line: Dict) -> bool:
    spk = str(line.get("speaker", "")).strip().upper()
    bt = str(line.get("bubble_type", "")).strip().lower()
    return spk == "NARRATOR" or bt == "caption"


def _is_intentional(line: Dict) -> bool:
    return any(bool(line.get(k)) for k in _INTENTIONAL_KEYS)


def _speaker_key(line: Dict) -> str:
    return str(line.get("speaker", "")).strip().lower()


def _recompute_density(panel: Dict) -> None:
    """Match the density convention used across the generator."""
    n = len([d for d in (panel.get("dialogue") or []) if isinstance(d, dict)])
    panel["dialogue_density"] = (
        "none" if n == 0 else
        "light" if n == 1 else
        "moderate" if n <= 3 else
        "heavy"
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class RepetitionReport:
    narration_removed: int = 0
    dialogue_removed: int = 0
    narration_rewritten: int = 0
    panels_emptied: int = 0
    examples: List[Dict] = field(default_factory=list)   # {page, panel, kind, kept, dropped}

    @property
    def total_removed(self) -> int:
        return self.narration_removed + self.dialogue_removed

    def as_dict(self) -> Dict:
        return {
            "narration_removed": self.narration_removed,
            "dialogue_removed": self.dialogue_removed,
            "narration_rewritten": self.narration_rewritten,
            "panels_emptied": self.panels_emptied,
            "total_removed": self.total_removed,
            "examples": self.examples[:40],
        }


# One remembered line in the sliding history.
@dataclass
class _Seen:
    pos: int                    # panel position in reading order
    norm: str
    tokens: Tuple[str, ...]
    text: str
    page: int
    panel: int


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------
def dedupe_repeated_dialogue(
    script: List[Dict],
    *,
    story_dna: Optional[Dict] = None,
    working_dir: Optional[str] = None,
    voice_profiles: Optional[Dict] = None,
    near_window: int = NEAR_WINDOW,
    exact_window: int = EXACT_WINDOW,
    char_window: int = CHAR_WINDOW,
    dedupe_character_lines: bool = True,
    rewrite_fn: Optional[Callable[[str, List[str]], Optional[str]]] = None,
) -> Tuple[List[Dict], RepetitionReport]:
    """Remove unintentional near-adjacent repeated dialogue / narration.

    Walks the script in reading order. For every spoken line it asks: did a
    *recent* line of the same kind already say this? If so — and the recurrence
    is not a protected intentional echo — the later copy is dropped so the beat
    is said once and the art is allowed to breathe.

    Parameters
    ----------
    story_dna : optional dict; a declared signature/refrain in it is exempted
        from exact-match removal wherever it recurs.
    voice_profiles : optional {name: CharacterVoiceProfile}; each character's
        OWN declared catchphrases are exempted from removal (scoped to that
        speaker only) — a catchphrase recurring is the point of it, not a bug.
        Overall frequency of a catchphrase across the whole book is instead
        governed by comic_book_dialogue_director's signature-usage check,
        which can flag it if it crosses from "a spice" into "a crutch".
    near_window / exact_window / char_window : panel-distance gates (see module
        docstring). Recurrences beyond these are treated as deliberate.
    dedupe_character_lines : also drop a character's *verbatim* immediate
        self-repeat (tiny window, exact only). Narration is always deduped.
    rewrite_fn : optional ``(dup_text, recent_texts) -> new_text | None``. When
        supplied, a duplicate NARRATION caption is offered to it for a fresh
        line instead of being dropped; returning None/'' falls back to dropping.
        Left None by default (drop-only) to keep the pass deterministic and to
        honour "say something new or nothing at all".

    Returns (script, RepetitionReport). Fails soft: on any error the script is
    returned unchanged.
    """
    report = RepetitionReport()
    if not isinstance(script, list) or not script:
        return script, report

    try:
        refrains = _collect_refrains(story_dna)
        catchphrases = _collect_catchphrases(voice_profiles)

        # Sliding histories. Narration shares one narrator voice → one history.
        # Character lines are tracked per speaker.
        narr_hist: List[_Seen] = []
        char_hist: Dict[str, List[_Seen]] = {}

        pos = 0  # panel position in reading order

        # Reading order: pages by page number, panels by panel_index.
        pages = [p for p in script
                 if isinstance(p, dict) and not p.get("_act_break")]
        try:
            pages.sort(key=lambda p: int(p.get("page", 0) or 0))
        except Exception:
            pass

        for page in pages:
            page_num = page.get("page", 0)
            panels = [pn for pn in (page.get("panels") or []) if isinstance(pn, dict)]
            try:
                panels.sort(key=lambda pn: int(pn.get("panel_index", 0) or 0))
            except Exception:
                pass

            for panel in panels:
                pos += 1
                panel_idx = panel.get("panel_index", 0)
                lines = [d for d in (panel.get("dialogue") or []) if isinstance(d, dict)]
                if not lines:
                    continue

                kept: List[Dict] = []
                for line in lines:
                    text = str(line.get("text", "")).strip()
                    if not text:
                        kept.append(line)
                        continue

                    norm = _normalise(text)
                    toks = _content_tokens(norm)

                    # Intentional echoes, declared refrains, and a speaker's OWN
                    # catchphrase: keep, and record so a later *accidental*
                    # repeat of them is still caught.
                    speaker_key = _speaker_key(line)
                    protected = (
                        _is_intentional(line)
                        or (norm in refrains)
                        or (norm in catchphrases.get(speaker_key, ()))
                    )

                    if _is_narration(line):
                        dup = None if protected else _find_dup(
                            narr_hist, pos, norm, toks,
                            near_window=near_window, exact_window=exact_window,
                        )
                        if dup is not None:
                            replacement = None
                            if rewrite_fn is not None:
                                replacement = _try_rewrite(
                                    rewrite_fn, text, narr_hist)
                            if replacement:
                                line["text"] = replacement
                                report.narration_rewritten += 1
                                report.examples.append({
                                    "page": page_num, "panel": panel_idx,
                                    "kind": "narration", "action": "rewritten",
                                    "was": text[:120], "now": replacement[:120],
                                    "clashed_with": dup.text[:120],
                                })
                                # Re-seed history with the fresh line.
                                rn = _normalise(replacement)
                                narr_hist.append(_Seen(
                                    pos, rn, _content_tokens(rn),
                                    replacement, page_num, panel_idx))
                                kept.append(line)
                            else:
                                report.narration_removed += 1
                                report.examples.append({
                                    "page": page_num, "panel": panel_idx,
                                    "kind": "narration", "action": "dropped",
                                    "dropped": text[:120],
                                    "kept": dup.text[:120],
                                    "kept_at": f"p{dup.page}/panel{dup.panel}",
                                })
                                dup.pos = pos   # extend the run so a 3rd repeat also goes
                            continue
                        narr_hist.append(_Seen(pos, norm, toks, text, page_num, panel_idx))
                        kept.append(line)
                        continue

                    # ---- character dialogue (incl. thought bubbles) ----
                    if dedupe_character_lines and not protected:
                        spk = _speaker_key(line)
                        hist = char_hist.setdefault(spk, [])
                        dup = _find_dup(
                            hist, pos, norm, toks,
                            near_window=char_window, exact_window=char_window,
                            exact_only=True,
                        )
                        if dup is not None:
                            report.dialogue_removed += 1
                            report.examples.append({
                                "page": page_num, "panel": panel_idx,
                                "kind": "dialogue", "action": "dropped",
                                "speaker": line.get("speaker", ""),
                                "dropped": text[:120], "kept": dup.text[:120],
                            })
                            dup.pos = pos
                            continue
                        hist.append(_Seen(pos, norm, toks, text, page_num, panel_idx))
                        kept.append(line)
                        continue

                    # protected character line: record + keep
                    if not _is_narration(line):
                        char_hist.setdefault(_speaker_key(line), []).append(
                            _Seen(pos, norm, toks, text, page_num, panel_idx))
                    kept.append(line)

                if len(kept) != len(lines):
                    panel["dialogue"] = kept
                    _recompute_density(panel)
                    if not kept:
                        report.panels_emptied += 1

        if report.total_removed or report.narration_rewritten:
            logger.info(
                "[Repetition] Removed %d narration + %d dialogue repeat(s); "
                "rewrote %d; %d panel(s) now silent (art breathes).",
                report.narration_removed, report.dialogue_removed,
                report.narration_rewritten, report.panels_emptied,
            )
        else:
            logger.info("[Repetition] No repeated dialogue/narration found.")

        if working_dir:
            try:
                with open(os.path.join(working_dir, "repetition_report.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(report.as_dict(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    except Exception as e:   # never let the guard break a build
        logger.warning("[Repetition] Guard skipped (%s); script unchanged.", e)
        return script, RepetitionReport()

    return script, report


def _find_dup(history: List[_Seen], pos: int, norm: str, toks: Tuple[str, ...],
              *, near_window: int, exact_window: int,
              exact_only: bool = False) -> Optional[_Seen]:
    """Return the most recent in-window history entry this line duplicates."""
    # Iterate newest-first so we match the closest prior occurrence.
    for seen in reversed(history):
        dist = pos - seen.pos
        if dist < 0:
            continue   # dist == 0 is allowed: two identical lines in one panel
        if dist > max(near_window, exact_window):
            break   # history is ordered; nothing older can be in window
        exact = (norm == seen.norm)
        if exact and dist <= exact_window:
            return seen
        if exact_only:
            continue
        if dist <= near_window and _is_near_duplicate(norm, toks, seen.norm, seen.tokens):
            return seen
    return None


def _try_rewrite(rewrite_fn: Callable[[str, List[str]], Optional[str]],
                 dup_text: str, narr_hist: List[_Seen]) -> Optional[str]:
    recent = [s.text for s in narr_hist[-6:]]
    try:
        new = rewrite_fn(dup_text, recent)
    except Exception as e:
        logger.warning("[Repetition] rewrite_fn failed (%s); dropping instead.", e)
        return None
    if not new:
        return None
    new = str(new).strip().strip('"').strip()
    if not new:
        return None
    # Guard against the rewrite just echoing something recent again.
    nn, nt = _normalise(new), None
    for s in narr_hist[-8:]:
        if _is_near_duplicate(nn, _content_tokens(nn), s.norm, s.tokens):
            return None
    return new


def _collect_refrains(story_dna: Optional[Dict]) -> set:
    """Normalised set of declared book-level refrains/signatures to protect."""
    out: set = set()
    if not isinstance(story_dna, dict):
        return out
    for key in ("_refrain", "refrain", "_signature_line", "signature_line",
                "_narrative_refrain"):
        val = story_dna.get(key)
        if isinstance(val, str) and val.strip():
            out.add(_normalise(val))
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, str) and v.strip():
                    out.add(_normalise(v))
    return out


def _collect_catchphrases(voice_profiles: Optional[Dict]) -> Dict[str, set]:
    """{normalised speaker key: {normalised catchphrase, ...}} from voice profiles.

    A catchphrase is a character's DELIBERATE recurring signature line — the
    whole reason it exists is to say the same thing again. Without this, the
    generic per-character dedupe (CHAR_WINDOW, exact-match) would delete a
    catchphrase the moment it recurred within a few panels of itself, fighting
    the voice system rather than serving it. This scopes the exemption to the
    OWNING speaker only (someone else saying the same words is not exempted).
    """
    out: Dict[str, set] = {}
    if not voice_profiles:
        return out
    for name, profile in voice_profiles.items():
        cps = getattr(profile, 'catchphrases', None)
        if cps is None and isinstance(profile, dict):
            cps = profile.get('catchphrases')
        if not cps:
            continue
        norm_set = {_normalise(str(c)) for c in cps if str(c).strip()}
        norm_set.discard('')
        if norm_set:
            out[str(name).strip().lower()] = norm_set
    return out


# ---------------------------------------------------------------------------
# Non-destructive audit (for validators / manifests)
# ---------------------------------------------------------------------------
def find_repetitions(script: List[Dict], *, story_dna: Optional[Dict] = None,
                     voice_profiles: Optional[Dict] = None,
                     near_window: int = NEAR_WINDOW,
                     exact_window: int = EXACT_WINDOW) -> List[Dict]:
    """Report near-adjacent repeats WITHOUT modifying the script.

    Returns a list of {kind, page, panel, text, clashes_with, at} dicts, useful
    for a validation gate or a manifest note.
    """
    findings: List[Dict] = []
    if not isinstance(script, list):
        return findings
    refrains = _collect_refrains(story_dna)
    catchphrases = _collect_catchphrases(voice_profiles)
    narr_hist: List[_Seen] = []
    char_hist: Dict[str, List[_Seen]] = {}
    pos = 0
    pages = [p for p in script if isinstance(p, dict) and not p.get("_act_break")]
    try:
        pages.sort(key=lambda p: int(p.get("page", 0) or 0))
    except Exception:
        pass
    for page in pages:
        page_num = page.get("page", 0)
        panels = [pn for pn in (page.get("panels") or []) if isinstance(pn, dict)]
        try:
            panels.sort(key=lambda pn: int(pn.get("panel_index", 0) or 0))
        except Exception:
            pass
        for panel in panels:
            pos += 1
            for line in (panel.get("dialogue") or []):
                if not isinstance(line, dict):
                    continue
                text = str(line.get("text", "")).strip()
                if not text or _is_intentional(line):
                    continue
                norm = _normalise(text)
                if norm in refrains:
                    continue
                if norm in catchphrases.get(_speaker_key(line), ()):
                    continue
                toks = _content_tokens(norm)
                if _is_narration(line):
                    dup = _find_dup(narr_hist, pos, norm, toks,
                                    near_window=near_window, exact_window=exact_window)
                    if dup is not None:
                        findings.append({
                            "kind": "narration", "page": page_num,
                            "panel": panel.get("panel_index", 0),
                            "text": text[:120], "clashes_with": dup.text[:120],
                            "at": f"p{dup.page}/panel{dup.panel}",
                        })
                    narr_hist.append(_Seen(pos, norm, toks, text, page_num,
                                           panel.get("panel_index", 0)))
                else:
                    spk = _speaker_key(line)
                    hist = char_hist.setdefault(spk, [])
                    dup = _find_dup(hist, pos, norm, toks,
                                    near_window=CHAR_WINDOW, exact_window=CHAR_WINDOW,
                                    exact_only=True)
                    if dup is not None:
                        findings.append({
                            "kind": "dialogue", "page": page_num,
                            "panel": panel.get("panel_index", 0),
                            "speaker": line.get("speaker", ""),
                            "text": text[:120], "clashes_with": dup.text[:120],
                            "at": f"p{dup.page}/panel{dup.panel}",
                        })
                    hist.append(_Seen(pos, norm, toks, text, page_num,
                                      panel.get("panel_index", 0)))
    return findings

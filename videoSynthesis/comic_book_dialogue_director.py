"""
comic_book_dialogue_director.py
===============================
The book-level DIALOGUE DIRECTOR — a managing-editor pass that runs once, LAST,
after every panel, caption, callback and delivery pass has finished, and gives
the whole book the read that no earlier pass can.

WHY A WHOLE-BOOK EDITOR (what the per-act reviewer structurally cannot do)
-------------------------------------------------------------------------
``review_and_enhance_dialog`` is already a strong senior-editor pass, but it
sees ONE ACT at a time. Four kinds of problem are only visible with the whole
book in hand, and this pass exists for exactly those:

  1. CROSS-ACT VOICE DRIFT — a character who is sharp in Act I and generic by
     Act IV. No per-act call ever holds both acts side by side; this pass
     fingerprints each voice per act and flags the drift.
  2. BOOK-WIDE PHRASING CRUTCHES — the narrator opening six captions the same
     way, or a character leaning on one word twenty times. Each act looks fine
     alone; the crutch only shows across the book.
  3. LONG-RANGE REPETITION — near-identical lines/captions far enough apart that
     the adjacency guard (comic_book_repetition) correctly leaves them, but
     which are accidental twins rather than deliberate callbacks.
  4. UNIFORM QUALITY LIFT — a single editorial voice re-polishing the entire
     book for imaginative, specific, non-clichéd writing and consistent
     character-authentic delivery, instead of ten independent act reviews.

HOW IT WORKS (deterministic analysis → token-budgeted LLM re-polish)
--------------------------------------------------------------------
The design honours this codebase's rule — the LLM does the creative writing,
plain Python holds the guarantees, and NO single call is allowed to blow the
model's context (the whole reason ``comic_book_token_budget`` exists):

  A. ``analyze_book`` reads the entire finished script CHEAPLY (pure Python) and
     builds the analytics no per-act pass has: per-character voice fingerprints
     and drift, book-wide overused phrases, narrator crutches, long-range near-
     duplicate line pairs, and on-the-nose / filler lines. This costs zero LLM
     tokens.
  B. The book is split into WINDOWS whose script text fits a hard per-call
     token budget (via ``comic_book_token_budget.estimate_tokens``), so "heaviest
     full re-polish" still means many small bounded calls, never one giant one.
  C. Each window is re-polished by the editor LLM, handed the window's script,
     the voice profiles of its speakers, the emotional-delivery guide, and ONLY
     the book-level editor notes relevant to that window (this character overuses
     "Listen"; the narrator keeps opening with "And so"; this caption is a twin
     of one on page 88). The final prompt is passed through ``guard_prompt`` as a
     last-resort backstop.
  D. Revisions are applied fail-soft with the same ``page_N_panel_M`` contract
     the act reviewer uses. Deliberate callbacks (tagged by
     ``weave_dialogue_callbacks``) are marked KEEP and reconciled back verbatim
     if the model touches them.
  E. Afterwards the delivery enforcement and the adjacency repetition guard are
     re-run, because a rewrite can lose a delivery tag or introduce a fresh
     repeat.

Everything degrades gracefully: any window that fails leaves its pages exactly
as they were, and the whole pass is wrapped so it can never break a build.

Integration (comic_book_generator, as the final dialogue pass, after callbacks
and before the density fit so rewrites get budget-fit):

    from comic_book_dialogue_director import polish_book_dialogue
    script, _ed = polish_book_dialogue(script, project)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- sibling deterministic modules (stdlib-only, no import cycle) ------------
from comic_book_token_budget import estimate_tokens, guard_prompt, clamp_text
from comic_book_repetition import (
    dedupe_repeated_dialogue,
    _normalise as _norm,
    _content_tokens as _ctoks,
    _is_near_duplicate as _near_dup,
    _is_intentional as _rep_intentional,
)

# --- tunables ----------------------------------------------------------------
# Script tokens packed into one re-polish window. Small on purpose: a window is
# a re-polish subtask, not the whole book. Sits far under any model ceiling.
WINDOW_SCRIPT_BUDGET: int = int(os.getenv("EDITOR_WINDOW_SCRIPT_BUDGET", "6000"))
# Hard cap for a single editor call's whole prompt (belt to guard_prompt).
EDITOR_CALL_MAX_TOKENS: int = int(os.getenv("EDITOR_CALL_MAX_TOKENS", "24000"))
# A content word used in at least this share of a character's lines (or this many
# times) is flagged as an overused tic for the editor to thin out.
TIC_SHARE: float = float(os.getenv("EDITOR_TIC_SHARE", "0.18"))
TIC_MIN_COUNT: int = int(os.getenv("EDITOR_TIC_MIN_COUNT", "5"))
# A repeated multi-word phrase seen at least this many times book-wide is a crutch.
PHRASE_MIN_COUNT: int = int(os.getenv("EDITOR_PHRASE_MIN_COUNT", "3"))
# Narrator caption opener (first 2-3 words) used at least this many times is a crutch.
OPENER_MIN_COUNT: int = int(os.getenv("EDITOR_OPENER_MIN_COUNT", "3"))
# Signature-usage check (catchphrases / signature_lexicon from the voice profile):
# minimum lines a character needs before we judge over/under-use — too few lines
# and one occurrence swings the share wildly, so stay quiet until there's a
# real sample.
SIGNATURE_MIN_LINES: int = int(os.getenv("EDITOR_SIGNATURE_MIN_LINES", "6"))
# Share of a character's lines containing ANY registered signature marker above
# which it reads as a crutch/caricature rather than a spice.
SIGNATURE_OVERUSE_SHARE: float = float(os.getenv("EDITOR_SIGNATURE_OVERUSE_SHARE", "0.4"))


# ---------------------------------------------------------------------------
# LLM + downstream helpers (lazy so the deterministic analysis unit-tests
# without the ML stack or the generator, and so there is no import cycle).
# ---------------------------------------------------------------------------
def _llm(prompt: str, *, temperature: float = 0.75,
         max_completion_tokens: int = 16000, cached_prefix: str = "") -> str:
    from novel_generator import (
        get_openai_prompt_response, openai_model_large, USE_GROK,
    )
    return get_openai_prompt_response(
        prompt, temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        openai_model=openai_model_large, use_grok=USE_GROK,
        cached_prefix=cached_prefix,
    )


def _parse(text: str):
    from novel_generator import parse_json_response
    return parse_json_response(text)


def _constant_blocks() -> Tuple[str, str]:
    """(delivery_guidance_block, cultural_guardrail) — identical for every window
    of a book, so they belong in the cacheable stable prefix."""
    try:
        from comic_book_dialogue import (
            delivery_guidance_block, CULTURAL_AUTHENTICITY_GUARDRAIL,
        )
        return delivery_guidance_block(), CULTURAL_AUTHENTICITY_GUARDRAIL
    except Exception:
        return "", ""


def _window_voice_guide(voice_profiles: Dict, speakers: List[str]) -> str:
    """The per-window voice profiles — VARIES by window (its speakers), so it
    trails the stable prefix, never fragments it."""
    try:
        from comic_book_dialogue import voice_profile_guide
        return voice_profile_guide(voice_profiles or {}, speakers)
    except Exception:
        return ""


def _enforce_delivery(script: List[Dict], voice_profiles: Dict) -> None:
    try:
        from comic_book_dialogue import apply_dialogue_delivery
        apply_dialogue_delivery(script, voice_profiles)
    except Exception as e:
        logger.warning("[Director] delivery re-enforcement skipped (%s).", e)


# ===========================================================================
# DETERMINISTIC WHOLE-BOOK ANALYSIS
# ===========================================================================
_STOP = frozenset((
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "is",
    "was", "were", "be", "it", "its", "i", "you", "he", "she", "they", "we",
    "me", "him", "her", "them", "us", "my", "your", "this", "that", "with",
    "for", "as", "so", "no", "not", "do", "did", "have", "has", "had", "if",
    "am", "are", "s", "t", "re", "ll", "ve", "m", "d", "o",
))
_WORD = re.compile(r"[a-z0-9']+")


def _is_narr(line: Dict) -> bool:
    return (str(line.get("speaker", "")).strip().upper() == "NARRATOR"
            or str(line.get("bubble_type", "")).strip().lower() == "caption")


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower())]


def _content(text: str) -> List[str]:
    return [w for w in _tokens(text) if w not in _STOP and len(w) > 1]


def _ngrams(words: List[str], n: int) -> List[str]:
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def _prof_get(profile: Any, attr: str) -> list:
    """Read a list-valued attribute off a CharacterVoiceProfile OR a dict."""
    if profile is None:
        return []
    v = getattr(profile, attr, None)
    if v is None and isinstance(profile, dict):
        v = profile.get(attr)
    return list(v) if isinstance(v, (list, tuple)) else []


def _registered_terms(profile: Any) -> Tuple[set, set]:
    """(single_word_terms, multi_word_phrases) normalised, for a voice profile.

    Pulls from catchphrases + signature_lexicon — the two voice-profile fields
    that are literal words/phrases a character actually says, as opposed to
    cultural_references (behavioural guidance, not matchable text) or accent/
    dialect_markers (prose descriptions of a pattern, not a literal string).
    """
    words: set = set()
    phrases: set = set()
    for item in _prof_get(profile, 'catchphrases') + _prof_get(profile, 'signature_lexicon'):
        n = _norm(str(item))
        if not n:
            continue
        toks = n.split()
        if len(toks) == 1:
            words.add(toks[0])
        else:
            phrases.add(n)
    return words, phrases


def _registered_tic_exclusions(profile: Any) -> set:
    """All individual words that belong to a registered term (word OR phrase).

    Used to keep the generic word-tic detector from ALSO flagging "ay", "dios",
    "mio" as separate overused words when "ay dios mio" is already covered,
    more precisely, by the dedicated signature-usage note.
    """
    words, phrases = _registered_terms(profile)
    out = set(words)
    for p in phrases:
        out.update(p.split())
    return out


@dataclass
class VoiceFingerprint:
    line_count: int = 0
    total_words: int = 0
    word_freq: Counter = field(default_factory=Counter)
    opener_freq: Counter = field(default_factory=Counter)
    per_act_words: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    per_act_lines: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Signature-usage tracking (populated only when voice_profiles is passed to
    # analyze_book): how many of this character's lines used ANY of their own
    # registered catchphrases/signature_lexicon, and which specific ones.
    signature_hits: int = 0
    signature_phrase_counts: Counter = field(default_factory=Counter)

    @property
    def avg_words(self) -> float:
        return (self.total_words / self.line_count) if self.line_count else 0.0

    def tics(self, exclude: Optional[set] = None) -> List[Tuple[str, int]]:
        out = []
        for w, c in self.word_freq.most_common(40):
            if exclude and w in exclude:
                continue   # this word is a REGISTERED signature term — the
                           # dedicated signature_note() covers it instead, so
                           # it isn't double-flagged as a generic "tic"
            if c >= TIC_MIN_COUNT or (self.line_count and c / self.line_count >= TIC_SHARE):
                out.append((w, c))
        return out[:8]

    def drift_note(self) -> str:
        """Compare earliest vs latest act top-word sets; report divergence."""
        acts = sorted(self.per_act_words.keys(),
                      key=lambda k: int(k) if str(k).isdigit() else 99)
        if len(acts) < 2:
            return ""
        first = {w for w, _ in self.per_act_words[acts[0]].most_common(15)}
        last = {w for w, _ in self.per_act_words[acts[-1]].most_common(15)}
        if not first or not last:
            return ""
        overlap = len(first & last) / len(first | last)
        if overlap < 0.12:
            return (f"voice may have drifted between act {acts[0]} and act "
                    f"{acts[-1]} (little shared vocabulary) — keep them the same person")
        return ""

    def signature_note(self, profile: Any) -> str:
        """Guidance on this character's use of their OWN voice-profile markers.

        Two failure modes, both worth catching:
          - OVERUSE: a catchphrase/signature word shows up in so many lines it
            reads as a crutch or caricature rather than a spice.
          - NEVER USED: the profile defines real signature markers but NONE of
            them ever appeared — the distinct voice this character was given
            never actually made it onto the page.
        Silent (returns "") when there's nothing registered, too few lines to
        judge, or usage is in the healthy middle.
        """
        words, phrases = _registered_terms(profile)
        if not words and not phrases:
            return ""
        if self.line_count < SIGNATURE_MIN_LINES:
            return ""
        share = self.signature_hits / self.line_count if self.line_count else 0.0
        if self.signature_hits == 0:
            sample = sorted(words | phrases)[:3]
            return (f"has a defined voice signature ({', '.join(sample)}) that "
                    f"never appeared across {self.line_count} lines — let it "
                    f"actually show up")
        if share > SIGNATURE_OVERUSE_SHARE:
            top = self.signature_phrase_counts.most_common(1)
            top_note = f', especially "{top[0][0]}" (\u00d7{top[0][1]})' if top else ''
            return (f"leans on their signature markers in {self.signature_hits}/"
                    f"{self.line_count} lines ({share:.0%}){top_note} — let more "
                    f"lines breathe without one, it's losing its spark")
        return ""


@dataclass
class BookDialogueAnalysis:
    fingerprints: Dict[str, VoiceFingerprint] = field(default_factory=dict)
    narrator: VoiceFingerprint = field(default_factory=VoiceFingerprint)
    overused_phrases: List[Tuple[str, int]] = field(default_factory=list)
    narrator_openers: List[Tuple[str, int]] = field(default_factory=list)
    # (page, panel, kind, text) → flagged weak line
    weak_lines: List[Dict] = field(default_factory=list)
    # long-range near-duplicate pairs: {a:{page,panel,text}, b:{...}}
    long_range_dupes: List[Dict] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "characters": {
                n: {
                    "lines": fp.line_count,
                    "avg_words": round(fp.avg_words, 1),
                    "tics": fp.tics(),
                    "drift": fp.drift_note(),
                }
                for n, fp in self.fingerprints.items()
            },
            "narrator": {
                "captions": self.narrator.line_count,
                "avg_words": round(self.narrator.avg_words, 1),
                "crutch_openers": self.narrator_openers,
                "crutch_words": self.narrator.tics(),
            },
            "overused_phrases": self.overused_phrases,
            "weak_lines": self.weak_lines[:60],
            "long_range_dupes": self.long_range_dupes[:40],
        }


# on-the-nose emotion telling + pure filler
_TELL_RE = re.compile(
    r"\bi(?:'m| am| feel| felt|'m feeling)\s+(?:so\s+|really\s+|very\s+)?"
    r"(angry|mad|sad|scared|afraid|happy|furious|terrified|nervous|worried|"
    r"excited|upset|frightened|anxious|heartbroken|devastated|thrilled|"
    r"lonely|jealous|guilty|ashamed|hopeless)\b", re.I)
_FILLER = frozenset((
    "hello", "hi", "hey", "yeah", "yes", "no", "okay", "ok", "well", "um",
    "uh", "right", "sure", "fine", "what", "huh", "oh",
))


def _ordered_pages(script: List[Dict]) -> List[Dict]:
    pages = [p for p in script if isinstance(p, dict) and not p.get("_act_break")]
    try:
        pages.sort(key=lambda p: int(p.get("page", 0) or 0))
    except Exception:
        pass
    return pages


def _ordered_panels(page: Dict) -> List[Dict]:
    panels = [pn for pn in (page.get("panels") or []) if isinstance(pn, dict)]
    try:
        panels.sort(key=lambda pn: int(pn.get("panel_index", 0) or 0))
    except Exception:
        pass
    return panels


def analyze_book(script: List[Dict],
                 voice_profiles: Optional[Dict] = None) -> BookDialogueAnalysis:
    """Cheap, deterministic whole-book read. Zero LLM tokens.

    ``voice_profiles``, when supplied, lets this also track whether each
    character actually uses their OWN declared catchphrases/signature_lexicon
    at a healthy rate (see VoiceFingerprint.signature_note) — and keeps those
    registered phrases out of the generic book-wide "overused phrase" crutch
    counter below, since a catchphrase recurring is by design, not a bug.
    """
    a = BookDialogueAnalysis()
    if not isinstance(script, list):
        return a

    voice_profiles = voice_profiles or {}
    # Registered multi-word phrases across the WHOLE cast, so the generic
    # book-wide phrase-crutch counter doesn't fight the catchphrase system by
    # telling the reviewer to "retire" a line that's supposed to recur.
    all_registered_phrases: set = set()
    for _prof in voice_profiles.values():
        _, _phrases = _registered_terms(_prof)
        all_registered_phrases |= _phrases

    phrase_counts: Counter = Counter()
    # for long-range dup detection
    seen_lines: List[Dict] = []   # {pos,norm,toks,text,page,panel,kind}
    pos = 0

    for page in _ordered_pages(script):
        act = str(page.get("act", "1"))
        page_num = page.get("page", 0)
        for panel in _ordered_panels(page):
            pos += 1
            panel_idx = panel.get("panel_index", 0)
            for line in (panel.get("dialogue") or []):
                if not isinstance(line, dict):
                    continue
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                narr = _is_narr(line)
                content = _content(text)
                words = _tokens(text)
                speaker_name = str(line.get("speaker", "")).strip() or "?"

                fp = a.narrator if narr else a.fingerprints.setdefault(
                    speaker_name, VoiceFingerprint())
                fp.line_count += 1
                fp.total_words += len(words)
                fp.word_freq.update(content)
                if not narr:
                    fp.per_act_words[act].update(content)
                    fp.per_act_lines[act] += 1
                # opener = first two content-ish words (keep short function words out)
                opener = " ".join(words[:3]).strip()
                if opener:
                    fp.opener_freq[opener] += 1

                # signature-marker usage: does this line use ANY of THIS
                # character's own registered catchphrases/signature_lexicon?
                if not narr and voice_profiles:
                    prof = voice_profiles.get(speaker_name)
                    if prof is not None:
                        sig_words, sig_phrases = _registered_terms(prof)
                        if sig_words or sig_phrases:
                            word_set = set(words)
                            matched = sorted(sig_words & word_set)
                            norm_line = _norm(text)
                            matched += [p for p in sig_phrases if p and p in norm_line]
                            if matched:
                                fp.signature_hits += 1
                                for m in matched:
                                    fp.signature_phrase_counts[m] += 1

                # book-wide phrase crutches (bi/tri-grams of content words) —
                # skip anything that's a registered signature phrase (or a
                # sub-slice of one) for ANY character; that's tracked (and
                # rate-limited) separately above.
                for n in (3, 2):
                    for g in _ngrams(content, n):
                        if any(g in rp or rp in g for rp in all_registered_phrases):
                            continue
                        phrase_counts[g] += 1

                # weak-line flags
                if not _rep_intentional(line):
                    tell = _TELL_RE.search(text)
                    if tell:
                        a.weak_lines.append({
                            "page": page_num, "panel": panel_idx,
                            "kind": "narration" if narr else "dialogue",
                            "reason": "on-the-nose emotion (show via behaviour/subtext)",
                            "text": text[:120],
                        })
                    elif content == [] and _norm(text) in _FILLER:
                        a.weak_lines.append({
                            "page": page_num, "panel": panel_idx,
                            "kind": "narration" if narr else "dialogue",
                            "reason": "filler-only line (make it earn its bubble or cut)",
                            "text": text[:120],
                        })

                # long-range near-dupe detection (skip intentional echoes)
                if not _rep_intentional(line):
                    nrm = _norm(text)
                    tks = _ctoks(nrm)
                    for s in seen_lines:
                        if pos - s["pos"] <= 8:
                            continue  # adjacency guard owns the near range
                        if s["kind"] != ("narr" if narr else "char"):
                            continue
                        if _near_dup(nrm, tks, s["norm"], s["toks"]):
                            a.long_range_dupes.append({
                                "a": {"page": s["page"], "panel": s["panel"],
                                      "text": s["text"][:100]},
                                "b": {"page": page_num, "panel": panel_idx,
                                      "text": text[:100]},
                                "kind": "narration" if narr else "dialogue",
                            })
                            break
                    seen_lines.append({
                        "pos": pos, "norm": nrm, "toks": tks, "text": text,
                        "page": page_num, "panel": panel_idx,
                        "kind": "narr" if narr else "char",
                    })

    # finalise crutch lists
    a.overused_phrases = [
        (p, c) for p, c in phrase_counts.most_common(40)
        if c >= PHRASE_MIN_COUNT and len(p.split()) >= 2
    ][:15]
    a.narrator_openers = [
        (o, c) for o, c in a.narrator.opener_freq.most_common(20)
        if c >= OPENER_MIN_COUNT
    ][:8]
    return a


# ===========================================================================
# EDITOR NOTES (filtered to a window)
# ===========================================================================
def _book_level_notes(a: BookDialogueAnalysis) -> str:
    """Notes that apply to the whole book (shown in every window)."""
    lines: List[str] = []
    if a.overused_phrases:
        lines.append(
            "OVERUSED PHRASES (book-wide) — vary or retire these; do not add more:\n  "
            + "; ".join(f'"{p}" (\u00d7{c})' for p, c in a.overused_phrases[:10]))
    if a.narrator_openers:
        lines.append(
            "NARRATOR OPENER CRUTCHES — stop starting captions this way; vary the "
            "entry:\n  " + "; ".join(f'"{o}\u2026" (\u00d7{c})' for o, c in a.narrator_openers))
    nt = a.narrator.tics()
    if nt:
        lines.append("NARRATOR OVERUSED WORDS: "
                     + ", ".join(f"{w}(\u00d7{c})" for w, c in nt))
    return "\n".join(lines)


def _window_notes(a: BookDialogueAnalysis, page_nums: set, speakers: set,
                  voice_profiles: Optional[Dict] = None) -> str:
    """Notes specific to the pages/speakers in this window."""
    voice_profiles = voice_profiles or {}
    lines: List[str] = []
    # per-character tics + drift + signature-usage for this window's speakers
    for name in sorted(speakers):
        fp = a.fingerprints.get(name)
        if not fp:
            continue
        profile = voice_profiles.get(name)
        sig_words = _registered_tic_exclusions(profile) if profile is not None else set()
        parts = []
        tics = fp.tics(exclude=sig_words)
        if tics:
            parts.append("overuses " + ", ".join(f"{w}(\u00d7{c})" for w, c in tics))
        drift = fp.drift_note()
        if drift:
            parts.append(drift)
        if profile is not None:
            sig_note = fp.signature_note(profile)
            if sig_note:
                parts.append(sig_note)
        if parts:
            lines.append(f"  {name}: " + "; ".join(parts))
    if lines:
        lines.insert(0, "CHARACTER VOICE NOTES for this window:")

    # weak lines on these pages
    wk = [w for w in a.weak_lines if w["page"] in page_nums][:12]
    if wk:
        lines.append("LINES TO REWRITE (weak / on-the-nose) on these pages:")
        for w in wk:
            lines.append(f'  p{w["page"]} panel {w["panel"]} [{w["reason"]}]: "{w["text"]}"')

    # long-range dupes touching these pages
    dp = [d for d in a.long_range_dupes
          if d["a"]["page"] in page_nums or d["b"]["page"] in page_nums][:8]
    if dp:
        lines.append("ACCIDENTAL TWINS (differentiate — make one say something new "
                     "or cut it):")
        for d in dp:
            lines.append(
                f'  p{d["a"]["page"]}:"{d["a"]["text"]}"  ~=~  '
                f'p{d["b"]["page"]}:"{d["b"]["text"]}"')
    return "\n".join(lines)


# ===========================================================================
# WINDOWING + SCRIPT RENDERING
# ===========================================================================
def _render_panel(page_num: int, panel: Dict) -> Tuple[str, set]:
    """Render one panel as addressable script text; return (text, speakers)."""
    pidx = panel.get("panel_index", 0)
    key = f"page_{page_num}_panel_{pidx}"
    desc = str(panel.get("description", "") or "")[:120]
    out = [f"[{key}] ({desc})"]
    speakers: set = set()
    dlg = [d for d in (panel.get("dialogue") or []) if isinstance(d, dict)]
    if not dlg:
        out.append("    (no dialogue)")
    for d in dlg:
        spk = str(d.get("speaker", "")).strip() or "?"
        bt = str(d.get("bubble_type", "speech")).strip()
        txt = str(d.get("text", "")).strip()
        keep = " \u27e8KEEP-VERBATIM\u27e9" if _rep_intentional(d) else ""
        out.append(f'    [{spk}/{bt}]{keep}: "{txt}"')
        if spk.upper() != "NARRATOR":
            speakers.add(spk)
    return "\n".join(out), speakers


def _windows(script: List[Dict], budget: int
             ) -> List[Tuple[List[Dict], set, set, str]]:
    """Pack pages into (pages, page_nums, speakers, rendered_text) windows that
    fit ``budget`` script tokens each. Never splits a page across windows."""
    windows = []
    cur_pages: List[Dict] = []
    cur_nums: set = set()
    cur_spk: set = set()
    cur_txt: List[str] = []
    cur_cost = 0

    def flush():
        if cur_pages:
            windows.append((list(cur_pages), set(cur_nums), set(cur_spk),
                            "\n".join(cur_txt)))

    for page in _ordered_pages(script):
        pnum = page.get("page", 0)
        blocks = [f"--- PAGE {pnum} ---"]
        pspk: set = set()
        for panel in _ordered_panels(page):
            txt, spk = _render_panel(pnum, panel)
            blocks.append(txt)
            pspk |= spk
        page_txt = "\n".join(blocks)
        cost = estimate_tokens(page_txt)
        if cur_pages and cur_cost + cost > budget:
            flush()
            cur_pages, cur_nums, cur_spk, cur_txt, cur_cost = [], set(), set(), [], 0
        cur_pages.append(page)
        cur_nums.add(pnum)
        cur_spk |= pspk
        cur_txt.append(page_txt)
        cur_cost += cost
    flush()
    return windows


# ===========================================================================
# EDITOR PROMPT
# ===========================================================================
_EDITOR_ROLE = (
    "You are the MANAGING EDITOR and DIALOGUE DIRECTOR giving a finished graphic "
    "novel its final line-edit. Every panel is already written; your job is to "
    "LIFT the writing, not to fill gaps. Re-polish the dialogue AND the narration "
    "captions in the pages below to the highest craft standard, obeying these "
    "principles:\n\n"
    "1. IMAGINATIVE, SPECIFIC WRITING — replace flat, generic, or clichéd lines "
    "with fresh, concrete, surprising language. Every line should sound authored, "
    "not defaulted.\n"
    "2. SHOW, DON'T TELL — turn on-the-nose emotion ('I'm so scared') into "
    "behaviour, subtext, or a charged image. Never name a feeling the art already "
    "shows.\n"
    "3. CHARACTER-TRUE VOICE — write each line the way THAT person actually sounds: "
    "their origin, accent, heritage and idiom (through authentic word choice and "
    "rhythm, light readable markers, never phonetic caricature) AND the emotional "
    "delivery of the moment (shout / slur / purr / stammer / cold). Use the voice "
    "profiles and delivery guide below.\n"
    "4. KILL REPETITION — never reuse a phrasing flagged as overused; vary each "
    "character's crutch words and the narrator's caption openers; differentiate "
    "any lines flagged as accidental twins. If a line or caption adds nothing the "
    "reader doesn't already have, CUT it — say something new or nothing at all and "
    "let the art breathe.\n"
    "5. NARRATION IS A VOICE, NOT A LABEL — captions must add time, interiority, "
    "irony, or judgement the images can't; they must not describe the picture or "
    "restate the dialogue, and they must vary in shape and entry.\n"
    "6. PRESERVE THE STORY — keep each line's meaning, its speaker, the beat, and "
    "any page-turn hook. Keep bubbles tight (≤15 words of dialogue). Keep "
    "speaker_side. Lines marked ⟨KEEP-VERBATIM⟩ are deliberate callbacks — return "
    "them UNCHANGED.\n\n"
    "OUTPUT CONTRACT — Return ONLY a JSON object. Keys are \"page_{N}_panel_{M}\" "
    "for panels you changed; the value is that panel's COMPLETE dialogue array "
    "(every line that should remain, in order), each line "
    "{\"speaker\",\"text\",\"bubble_type\",\"speaker_side\",\"delivery\"}. To DELETE "
    "a redundant line, simply omit it from the array (return the array without it; "
    "an empty array [] clears the panel). Omit panels you leave unchanged.\n"
)


def build_stable_prefix(story_meta: str, guardrail: str, delivery_block: str,
                        book_notes: str) -> str:
    """The book-constant head of every window prompt.

    xAI/Grok caching is prefix-based: it reuses the KV cache for the longest
    IDENTICAL leading span shared across calls on the same x-grok-conv-id. So we
    front-load everything that does not change between windows of a book — the
    editor role, story meta, the cultural guardrail, the delivery guide, and the
    book-level editor notes — into one byte-identical block. Only the per-window
    voice profiles, page notes, and script trail it, so every window after the
    first serves this whole prefix from cache (billed at the cached-token rate).

    IMPORTANT: keep this deterministic and free of per-window data (no speaker
    lists, page numbers, or counts) or the prefix stops matching and the cache
    saving evaporates.
    """
    parts = [_EDITOR_ROLE, "\n"]
    if story_meta:
        parts += [story_meta, "\n"]
    if guardrail:
        parts += [guardrail, "\n\n"]
    if delivery_block:
        parts += [delivery_block, "\n\n"]
    if book_notes:
        parts += ["=== BOOK-LEVEL EDITOR NOTES ===\n", book_notes, "\n\n"]
    return "".join(parts)


def _build_window_prompt(voice_guide: str, win_notes: str,
                         window_txt: str) -> str:
    """The variable tail for THIS window (the cache-stable prefix is passed
    separately as cached_prefix so Grok serves it from cache every window)."""
    parts = []
    if voice_guide:                               # ← varies (window speakers)
        parts += [voice_guide, "\n\n"]
    if win_notes:                                 # ← varies (window pages)
        parts += ["=== NOTES FOR THESE PAGES ===\n", win_notes, "\n\n"]
    parts += ["=== PAGES TO RE-POLISH ===\n", window_txt, "\n\n",
              "Re-polish every page above per the six principles. Return ONLY the "
              "JSON object of changed panels."]
    tail = "".join(parts)
    # Last-resort backstop so a pathological window can never 400/413. This only
    # trims when a window is oversized (rare); a normal tail is returned as-is.
    return guard_prompt(tail, EDITOR_CALL_MAX_TOKENS, origin="dialogue_director")


# ===========================================================================
# APPLY (fail-soft; protects ⟨KEEP-VERBATIM⟩ callbacks)
# ===========================================================================
def _protected_map(script: List[Dict]) -> Dict[str, List[Dict]]:
    """page_N_panel_M → list of intentional-echo lines to preserve verbatim."""
    out: Dict[str, List[Dict]] = {}
    for page in _ordered_pages(script):
        pnum = page.get("page", 0)
        for panel in _ordered_panels(page):
            key = f"page_{pnum}_panel_{panel.get('panel_index', 0)}"
            prot = [dict(d) for d in (panel.get("dialogue") or [])
                    if isinstance(d, dict) and _rep_intentional(d)]
            if prot:
                out[key] = prot
    return out


def _apply(script: List[Dict], revisions: Dict[str, Any],
           protected: Dict[str, List[Dict]]) -> int:
    """Apply page_N_panel_M → dialogue-array revisions in place. Returns panels changed."""
    if not isinstance(revisions, dict):
        return 0
    index: Dict[str, Dict] = {}
    for page in _ordered_pages(script):
        pnum = page.get("page", 0)
        for panel in _ordered_panels(page):
            index[f"page_{pnum}_panel_{panel.get('panel_index', 0)}"] = panel

    changed = 0
    for key, arr in revisions.items():
        panel = index.get(str(key))
        if panel is None or not isinstance(arr, list):
            continue
        normed: List[Dict] = []
        for e in arr:
            if not isinstance(e, dict):
                continue
            text = str(e.get("text", "") or "").strip()
            if not text:
                continue
            side = (str(e.get("speaker_side", "")) or "").lower()
            side = "right" if "right" in side else "center" if "center" in side else "left"
            normed.append({
                "speaker": e.get("speaker", ""),
                "text": text,
                "bubble_type": e.get("bubble_type", "speech"),
                "speaker_side": side,
                "delivery": e.get("delivery", ""),
            })
        # Reconcile protected callbacks: if the model dropped/altered one, re-add it.
        for prot in protected.get(str(key), []):
            pn = _norm(str(prot.get("text", "")))
            if not any(_norm(l["text"]) == pn for l in normed):
                normed.append({k: prot.get(k) for k in
                               ("speaker", "text", "bubble_type", "speaker_side",
                                "delivery")})
                # carry the intentional flags forward
                for fk, fv in prot.items():
                    if fk.startswith("_") or fk == "intentional_repeat":
                        normed[-1][fk] = fv
        panel["dialogue"] = normed
        n = len(normed)
        panel["dialogue_density"] = (
            "none" if n == 0 else "light" if n == 1 else
            "moderate" if n <= 3 else "heavy")
        changed += 1
    return changed


# ===========================================================================
# PUBLIC ENTRYPOINT
# ===========================================================================
@dataclass
class EditorResult:
    windows: int = 0
    panels_revised: int = 0
    lines_cut: int = 0
    analysis: Optional[Dict] = None


def polish_book_dialogue(
    script: List[Dict],
    project: Any = None,
    *,
    voice_profiles: Optional[Dict] = None,
    working_dir: Optional[str] = None,
    story_idea: Any = None,
    window_budget: int = WINDOW_SCRIPT_BUDGET,
    llm_fn: Optional[Callable[..., str]] = None,
    parse_fn: Optional[Callable[[str], Any]] = None,
    run_repetition_guard: bool = True,
) -> Tuple[List[Dict], EditorResult]:
    """Full book-level dialogue+narration re-polish. See module docstring.

    ``llm_fn`` / ``parse_fn`` are injectable for testing; they default to the
    project's real LLM helpers. Fails soft window-by-window and overall.
    """
    result = EditorResult()
    if not isinstance(script, list) or not script:
        return script, result

    llm_fn = llm_fn or _llm
    parse_fn = parse_fn or _parse
    if project is not None:
        voice_profiles = voice_profiles or getattr(project, "voice_profiles", None)
        working_dir = working_dir or getattr(project, "working_dir", None)
        story_idea = story_idea or getattr(project, "story_idea", None)
    voice_profiles = voice_profiles or {}

    # A) whole-book deterministic analysis (0 tokens)
    try:
        analysis = analyze_book(script, voice_profiles)
        result.analysis = analysis.as_dict()
    except Exception as e:
        logger.warning("[Director] analysis failed (%s); skipping re-polish.", e)
        return script, result

    book_notes = _book_level_notes(analysis)
    story_meta = ""
    if story_idea is not None:
        try:
            story_meta = (
                f"STORY: {getattr(story_idea,'genre','')} | "
                f"{getattr(story_idea,'mood','')}\n"
                f"THEMES: {', '.join((getattr(story_idea,'themes',None) or [])[:5])}\n"
            )
        except Exception:
            story_meta = ""

    protected = _protected_map(script)

    # Build the CACHE-STABLE PREFIX once: role + story meta + guardrail +
    # delivery guide + book-level notes are identical for every window, so
    # front-loading them into one block lets Grok serve the whole prefix from
    # its prompt cache on every window after the first (see build_stable_prefix).
    delivery_block, guardrail = _constant_blocks()
    stable_prefix = build_stable_prefix(story_meta, guardrail, delivery_block,
                                        book_notes)

    # B) token-budgeted windows
    try:
        windows = _windows(script, window_budget)
    except Exception as e:
        logger.warning("[Director] windowing failed (%s); skipping.", e)
        return script, result
    result.windows = len(windows)
    logger.info("[6.9/8] Dialogue director: full re-polish over %d window(s) "
                "(%d character voice(s) profiled)...",
                len(windows), len(analysis.fingerprints))

    lines_before = _count_lines(script)

    # C/D) per-window LLM re-polish + apply
    for wi, (pages, page_nums, speakers, window_txt) in enumerate(windows, 1):
        try:
            voice_guide = _window_voice_guide(voice_profiles, sorted(speakers))
            win_notes = _window_notes(analysis, page_nums, speakers, voice_profiles)
            prompt = _build_window_prompt(voice_guide, win_notes, window_txt)
            raw = llm_fn(prompt, temperature=0.78, max_completion_tokens=16000,
                         cached_prefix=stable_prefix)
            revisions = parse_fn(raw)
            n = _apply(script, revisions, protected)
            result.panels_revised += n
            logger.info("  [Director] window %d/%d: %d panel(s) re-polished.",
                        wi, len(windows), n)
        except Exception as e:
            logger.warning("  [Director] window %d/%d skipped (%s); pages left "
                           "as-is.", wi, len(windows), e)
            continue

    # E) re-enforce delivery, then re-run the adjacency repetition guard
    _enforce_delivery(script, voice_profiles)
    if run_repetition_guard:
        try:
            script, _ = dedupe_repeated_dialogue(
                script,
                story_dna=getattr(project, "story_dna", None) if project else None,
                working_dir=None,
                voice_profiles=voice_profiles)
        except Exception as e:
            logger.warning("[Director] post repetition guard skipped (%s).", e)

    result.lines_cut = max(0, lines_before - _count_lines(script))
    logger.info("[Director] Re-polish complete: %d panel(s) revised, %d "
                "redundant line(s) cut across %d window(s).",
                result.panels_revised, result.lines_cut, result.windows)

    if working_dir:
        try:
            with open(os.path.join(working_dir, "dialogue_editor_report.json"),
                      "w", encoding="utf-8") as f:
                json.dump({
                    "windows": result.windows,
                    "panels_revised": result.panels_revised,
                    "lines_cut": result.lines_cut,
                    "analysis": result.analysis,
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return script, result


def _count_lines(script: List[Dict]) -> int:
    n = 0
    for page in _ordered_pages(script):
        for panel in _ordered_panels(page):
            n += len([d for d in (panel.get("dialogue") or []) if isinstance(d, dict)])
    return n

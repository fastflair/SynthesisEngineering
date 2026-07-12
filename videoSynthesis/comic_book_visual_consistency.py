"""
comic_book_visual_consistency.py
================================

Post-render visual consistency verification — the "detective" layer.

WHY THIS EXISTS
---------------
Every other consistency system in the pipeline is PREVENTIVE: it writes a better
prompt (costume locks, 180-degree screen-position locks, permanent-marking locks,
scene anchors) and then TRUSTS that the diffusion model obeyed. Diffusion models
routinely do not. A prompt that says "the woman wears a blue shirt" frequently
renders a black one; a character locked to screen-left drifts to screen-right;
a tattoo on the left forearm vanishes or migrates. Nothing downstream ever looks
at the pixels, so the drift ships to the reader and has to be fixed by hand.

This module closes the loop. It does two things:

  1. build_consistency_ledger(panel_script, characters_in_frame)
     Distils the ground truth the pipeline ALREADY computed — the per-character
     `_appearance_state` (clothing, hair, conditions, held items, permanent
     markings), the resolved screen positions (180-degree rule), and the panel's
     key emotion — into a small, structured, comparable ledger. No new LLM call;
     it just joins data that already exists but was never captured in one place.

  2. check_variant_against_ledger(image, ledger, vision_fn)
     Uses a vision-capable model to score how well a RENDERED variant matches the
     ledger, returning a 0..1 score plus a list of concrete mismatches
     ("expected blue shirt, saw black"). The caller can then pick the
     best-scoring variant, or flag the panel for regeneration when every variant
     drifts badly.

DESIGN PRINCIPLES
-----------------
* Zero hard dependencies beyond the stdlib + PIL (already used everywhere). The
  vision model is INJECTED as a callable, so this module never imports a client
  and works in tests with a stub.
* Degrades to a safe no-op. If no vision function is supplied, or it errors, or
  the ledger is empty, the checker returns a neutral result and the pipeline
  behaves exactly as it does today. Verification is strictly additive.
* Cheap by construction. Only panels that HAVE a ledger worth checking (a named
  costume colour, a screen-side lock, a marking, or a held item) are worth a
  vision call; is_ledger_checkable() lets the caller skip the rest.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour vocabulary — used to pull a checkable colour term out of a costume
# description ("a faded blue denim jacket" -> "blue"). Kept deliberately small:
# these are the terms a vision model can adjudicate unambiguously.
# ---------------------------------------------------------------------------
_COLOUR_TERMS = (
    "black", "white", "grey", "gray", "silver", "red", "crimson", "scarlet",
    "maroon", "burgundy", "orange", "amber", "yellow", "gold", "green", "olive",
    "teal", "blue", "navy", "cyan", "azure", "indigo", "purple", "violet",
    "magenta", "pink", "rose", "brown", "tan", "beige", "cream", "khaki",
)
_COLOUR_RE = re.compile(r"\b(" + "|".join(_COLOUR_TERMS) + r")\b", re.IGNORECASE)

# Garment nouns we try to bind a colour to, so a mismatch can be reported
# specifically ("shirt", not just "clothing").
_GARMENT_TERMS = (
    "shirt", "blouse", "jacket", "coat", "cloak", "dress", "gown", "robe",
    "sweater", "hoodie", "vest", "tunic", "uniform", "armor", "armour",
    "trousers", "pants", "jeans", "skirt", "scarf", "hat", "cap", "hood",
    "gloves", "boots", "cape", "suit", "kimono", "sari",
)
_GARMENT_RE = re.compile(r"\b(" + "|".join(_GARMENT_TERMS) + r")\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
@dataclass
class CharacterExpectation:
    """The checkable visual facts expected of ONE character in ONE panel."""
    name: str
    gender: str = "figure"
    screen_side: str = ""                 # "left" | "right" | "center" | ""
    garment_colours: List[Tuple[str, str]] = field(default_factory=list)
    #                    ^ list of (garment, colour) e.g. ("shirt", "blue")
    conditions: List[str] = field(default_factory=list)   # wet, bloodied, muddy…
    permanent_markings: str = ""          # "jagged scar across left cheek"
    held_items: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """A short natural-language line the vision model can check against."""
        bits: List[str] = [f"{self._subject()}"]
        if self.garment_colours:
            bits.append(
                "wearing " + ", ".join(f"a {c} {g}" for g, c in self.garment_colours)
            )
        if self.permanent_markings:
            bits.append(f"with {self.permanent_markings}")
        if self.conditions:
            bits.append("appears " + ", ".join(self.conditions))
        if self.held_items:
            bits.append("holding " + ", ".join(self.held_items))
        if self.screen_side:
            bits.append(f"positioned on the {self.screen_side} of the frame")
        return " ".join(bits)

    def _subject(self) -> str:
        return f"the {self.gender}" if self.gender else "the character"

    def is_checkable(self) -> bool:
        """True when there is at least one hard, adjudicable visual fact."""
        return bool(self.garment_colours or self.permanent_markings
                    or self.held_items or self.screen_side)


@dataclass
class PanelConsistencyLedger:
    """All checkable visual expectations for a single panel."""
    page_number: int = 0
    panel_index: int = 0
    characters: List[CharacterExpectation] = field(default_factory=list)
    key_emotion: str = ""
    setting: str = ""

    def is_checkable(self) -> bool:
        return any(c.is_checkable() for c in self.characters)

    def as_check_spec(self) -> str:
        """Render the ledger into the numbered spec the vision model scores."""
        lines: List[str] = []
        for i, c in enumerate(self.characters):
            if c.is_checkable():
                lines.append(f"  C{i} — {c.describe()}.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ledger construction from existing panel state (no new LLM call)
# ---------------------------------------------------------------------------
def _extract_garment_colours(costume_text: str) -> List[Tuple[str, str]]:
    """Pull (garment, colour) pairs out of a free-text costume description.

    Strategy: for each garment noun found, take the nearest preceding colour
    within a small window (so "blue denim jacket" -> ("jacket","blue") but a
    colour far away in another clause isn't wrongly attached). Falls back to a
    bare colour with an unspecified garment when a colour appears with no noun.
    """
    if not costume_text:
        return []
    text = str(costume_text)
    pairs: List[Tuple[str, str]] = []
    for gm in _GARMENT_RE.finditer(text):
        garment = gm.group(1).lower()
        window = text[max(0, gm.start() - 40):gm.start()]
        cols = _COLOUR_RE.findall(window)
        if cols:
            pairs.append((garment, cols[-1].lower()))
    if not pairs:
        # No garment noun matched a colour; record the first colour generically
        first = _COLOUR_RE.search(text)
        if first:
            pairs.append(("clothing", first.group(1).lower()))
    # De-dup while preserving order; cap to keep the vision prompt tight.
    seen = set()
    out: List[Tuple[str, str]] = []
    for g, c in pairs:
        if (g, c) not in seen:
            seen.add((g, c))
            out.append((g, c))
    return out[:3]


def build_consistency_ledger(
    panel_script: Dict[str, Any],
    characters_in_frame: Optional[List[str]] = None,
    screen_positions: Optional[Dict[str, str]] = None,
) -> PanelConsistencyLedger:
    """Distil a checkable ledger from the panel's already-computed state.

    Reads the same `_appearance_state` the appearance_continuity_clause() uses,
    plus the resolved screen positions (180-degree rule) and the panel's arc
    emotion. Never calls an LLM; returns an empty-but-valid ledger when the
    panel has no tracked state.
    """
    ledger = PanelConsistencyLedger(
        page_number=int(panel_script.get("_page_number", 0) or 0),
        panel_index=int(panel_script.get("panel_index", 0) or 0),
        key_emotion=str(panel_script.get("_arc_emotion", "") or ""),
        setting=str(panel_script.get("setting", "") or "")[:120],
    )

    state: Dict[str, Any] = panel_script.get("_appearance_state") or {}
    frame = characters_in_frame or list(state.keys())
    positions = screen_positions or {}

    for name in frame:
        st = state.get(name, {}) if isinstance(state, dict) else {}
        exp = CharacterExpectation(
            name=name,
            gender=str((st or {}).get("gender", "figure") or "figure"),
            screen_side=str(positions.get(name, "") or "").lower(),
        )
        # Costume colours — prefer the tracked evolved clothing, else the static
        # costume/appearance description on the panel or character record.
        costume = (
            (st or {}).get("clothing")
            or (st or {}).get("costume")
            or ""
        )
        exp.garment_colours = _extract_garment_colours(str(costume))

        # Conditions (wet / bloodied / muddy …) — only the visually-adjudicable
        # ones; skip abstract state.
        conds = (st or {}).get("conditions") or []
        exp.conditions = [str(c) for c in conds if str(c).strip()][:4]

        # Permanent markings (scar / tattoo / prosthetic) — an identity lock.
        exp.permanent_markings = str((st or {}).get("permanent_markings", "") or "").strip()[:120]
        # Fold in the stamped physical-signature identity lock too, so the vision
        # checker verifies the scar/tattoo/distinctive feature actually rendered.
        sig_map = panel_script.get("_signature_lock") or {}
        if isinstance(sig_map, dict):
            sig = sig_map.get(name) or next(
                (v for k, v in sig_map.items()
                 if k.lower() == name.lower() or name.lower() in k.lower()), "")
            if sig and sig not in exp.permanent_markings:
                exp.permanent_markings = (
                    (exp.permanent_markings + "; " + sig).strip("; ")
                    if exp.permanent_markings else str(sig)[:120]
                )

        # Held items.
        held = (st or {}).get("held_items") or []
        if isinstance(held, str):
            held = [held]
        exp.held_items = [str(h) for h in held if str(h).strip()][:3]

        ledger.characters.append(exp)

    return ledger


def is_ledger_checkable(ledger: PanelConsistencyLedger) -> bool:
    """Convenience guard so callers can cheaply skip un-checkable panels."""
    return bool(ledger and ledger.is_checkable())


# ---------------------------------------------------------------------------
# Vision check
# ---------------------------------------------------------------------------
@dataclass
class VariantConsistencyResult:
    """The outcome of checking one rendered variant against the ledger."""
    score: float = 1.0                    # 0..1; 1 = perfect match, 1.0 when unchecked
    mismatches: List[str] = field(default_factory=list)
    checked: bool = False                 # False when no vision model ran
    error: str = ""

    @property
    def is_good(self) -> bool:
        return self.score >= 0.75

    def summary(self) -> str:
        if not self.checked:
            return "unchecked"
        if not self.mismatches:
            return f"score={self.score:.2f} (clean)"
        return f"score={self.score:.2f}: " + "; ".join(self.mismatches[:4])


def _image_to_data_url(image, fmt: str = "JPEG", max_side: int = 768) -> Optional[str]:
    """Encode a PIL image as a base64 data URL, downscaled to keep tokens low.

    Vision checks don't need full print resolution — a 768px longest-side JPEG
    is plenty to read a shirt colour or which side a figure stands on, and keeps
    the request small. Returns None if the image can't be encoded.
    """
    try:
        img = image
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0 and hasattr(img, "resize"):
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{b64}"
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("[VisualConsistency] image encode failed: %s", e)
        return None


_CHECK_INSTRUCTIONS = (
    "You are a comic-book continuity checker. You are shown ONE rendered panel "
    "image and a short list of expected visual facts about the character(s) in "
    "it. For EACH numbered expectation, decide whether the image matches. Judge "
    "only what is clearly visible; if something is genuinely not visible in the "
    "frame (e.g. a forearm tattoo when the arms are out of shot), treat it as "
    "'not violated', not a mismatch.\n\n"
    "Report concrete mismatches only — a wrong garment colour, a figure on the "
    "wrong side of the frame, a missing or moved permanent marking, a held item "
    "that is absent. Do NOT nitpick art style, rendering quality, or anything "
    "not in the expectation list.\n\n"
    "Return ONLY JSON:\n"
    '{"score": 0.0-1.0, "mismatches": ["expected X, saw Y", ...]}\n'
    "score is the fraction of checkable expectations the image satisfies "
    "(1.0 = all satisfied, no visible violations). JSON only."
)


def check_variant_against_ledger(
    image,
    ledger: PanelConsistencyLedger,
    vision_fn: Optional[Callable[..., str]] = None,
    *,
    parse_json: Optional[Callable[[str], Any]] = None,
) -> VariantConsistencyResult:
    """Score a single rendered variant image against the panel's ledger.

    Parameters
    ----------
    image      : a PIL.Image (or anything with .convert/.save/.size/.resize).
    ledger     : the PanelConsistencyLedger for this panel.
    vision_fn  : callable(prompt: str, image_url: str) -> str  (JSON reply).
                 INJECTED so this module never binds a client. When None, the
                 check is skipped and a neutral (unchecked) result is returned.
    parse_json : optional JSON parser (defaults to json.loads with a fence strip).

    Never raises. On any failure returns an unchecked, neutral result so the
    caller keeps the current behaviour.
    """
    result = VariantConsistencyResult()

    if vision_fn is None or ledger is None or not ledger.is_checkable():
        return result  # neutral no-op

    spec = ledger.as_check_spec()
    if not spec.strip():
        return result

    data_url = _image_to_data_url(image)
    if not data_url:
        result.error = "encode_failed"
        return result

    prompt = (
        _CHECK_INSTRUCTIONS
        + "\n\nEXPECTED VISUAL FACTS:\n" + spec
        + "\n\nCheck the image now. JSON only."
    )

    try:
        raw = vision_fn(prompt, data_url)
    except Exception as e:
        logger.debug("[VisualConsistency] vision call failed: %s", e)
        result.error = f"vision_error:{e}"
        return result

    parsed = _safe_parse(raw, parse_json)
    if not isinstance(parsed, dict):
        result.error = "unparseable"
        return result

    try:
        score = float(parsed.get("score", 1.0))
    except (TypeError, ValueError):
        score = 1.0
    score = max(0.0, min(1.0, score))
    mismatches = parsed.get("mismatches") or []
    if isinstance(mismatches, str):
        mismatches = [mismatches]
    mismatches = [str(m).strip() for m in mismatches if str(m).strip()][:8]

    result.score = score
    result.mismatches = mismatches
    result.checked = True
    return result


def pick_best_variant(
    variant_images: List[Any],
    ledger: PanelConsistencyLedger,
    vision_fn: Optional[Callable[..., str]] = None,
    *,
    parse_json: Optional[Callable[[str], Any]] = None,
    regen_threshold: float = 0.5,
) -> Tuple[int, List[VariantConsistencyResult], bool]:
    """Score every variant and return (best_index, results, needs_regen).

    * best_index   : index of the highest-scoring variant (0 if unchecked).
    * results      : per-variant VariantConsistencyResult, same order as input.
    * needs_regen  : True when even the BEST variant scores below
                     ``regen_threshold`` — i.e. every render drifted badly and
                     the panel is a good candidate to regenerate.

    Fully degrades: with no vision_fn (or an un-checkable ledger) it returns
    (0, [...unchecked], False), preserving today's "pick variant 0" behaviour.
    """
    results: List[VariantConsistencyResult] = []
    if not variant_images:
        return 0, results, False

    if vision_fn is None or not is_ledger_checkable(ledger):
        return 0, [VariantConsistencyResult() for _ in variant_images], False

    for img in variant_images:
        results.append(
            check_variant_against_ledger(img, ledger, vision_fn, parse_json=parse_json)
        )

    checked = [r for r in results if r.checked]
    if not checked:
        return 0, results, False

    best_index = max(range(len(results)), key=lambda i: results[i].score)
    needs_regen = results[best_index].score < regen_threshold
    return best_index, results, needs_regen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_parse(raw: str, parse_json: Optional[Callable[[str], Any]]):
    if parse_json is not None:
        try:
            return parse_json(raw)
        except Exception:
            pass
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Grab the first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None

"""
comic_book_period_consistency.py
================================

Time-period / setting-era consistency — the "world clock" layer.

WHY THIS EXISTS
---------------
Every visual-consistency system in the pipeline so far tracks *who* is in a
panel and *what they wear* (AppearanceContinuityTracker), *what props exist*
(SeriesVisualBible / WorldContinuityTracker), and *where* a scene is set
(scene anchors). Nothing tracks *when* the story happens. A diffusion model has
no idea a story is set in the Middle Ages: prompt it with "a scholar reads at
his desk" and it will happily add a laptop, a desk lamp, and a coffee mug,
because that is the modal "desk" in its training distribution. The result is a
knight checking a smartphone, a Roman senator beside a wall clock, a Regency
ballroom lit by LED strips.

This module gives the pipeline a single, story-derived notion of ERA and uses
it three ways, mirroring how the style-lock already works:

  1. A POSITIVE lock injected into every image prompt
     (``period_guard_clause``) — "this is <era>; only period-appropriate
     technology, materials, architecture and dress appear."

  2. A NEGATIVE-prompt term list (``period_negative_terms``) naming the
     specific anachronisms to suppress at the latent level — because a
     *negated* noun in a POSITIVE prompt ("no smartphones") activates the
     concept rather than removing it, exactly like the media/style negatives.

  3. A pre-render SCRUB of the prompt text itself (``scrub_anachronisms``):
     if an upstream description literally says "she glances at her phone" in a
     medieval story, the positive/negative locks fight a losing battle — better
     to rewrite the anachronism to a period equivalent before it reaches the
     model.

DESIGN PRINCIPLES (identical to the rest of the art-director layer)
-------------------------------------------------------------------
* The era is established ONCE per story with a single LLM call
  (``establish_period_profile``) and cached; every downstream use is free.
* The LLM is INJECTED. With no LLM available a deterministic keyword
  classifier still produces a useful profile for the common cases
  (medieval / ancient / Victorian / modern / near-future / far-future), so the
  module degrades to "coarse but safe" rather than "off".
* Story-driven, never blanket. A story can legitimately move between eras — a
  flashback, a time-machine, a framing device set in the present that narrates
  the past. ``annotate_script_with_period`` classifies each SCENE and stamps
  the panel with the era that is actually active there, so a single
  present-day framing scene in a historical film doesn't get its laptops
  scrubbed, and the medieval body of the film doesn't inherit the frame's.
* Purely additive. Every function no-ops safely on empty input, and stamping a
  ``_period_lock`` onto a panel changes nothing unless a caller reads it.

INTEGRATION (three one-line touch-points — see INTEGRATION_GUIDE.md)
--------------------------------------------------------------------
  * After the script is built, once:
        profile = establish_period_profile(story_idea)
        annotate_script_with_period(script, profile)      # stamps _period_lock
  * In the positive prompt builder (compose_panel_prompt), append:
        period_guard_clause(panel_script)
  * In _adaptive_negative_prompt(panel_script), append:
        period_negative_terms(panel_script=panel_script)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM plumbing — injected, mirrors comic_book_art_director._llm / _parse.
# ---------------------------------------------------------------------------
def _default_llm(prompt: str, temperature: float = 0.2, large: bool = True) -> str:
    """Thin wrapper around the project's LLM utility (lazy import)."""
    from novel_generator import (
        get_openai_prompt_response, openai_model, openai_model_large, USE_GROK,
    )
    model = openai_model_large if large else openai_model
    return get_openai_prompt_response(
        prompt, temperature=temperature, openai_model=model, use_grok=USE_GROK,
    )


def _default_parse(text: str):
    try:
        from novel_generator import parse_json_response
        return parse_json_response(text)
    except Exception:
        return _fallback_parse(text)


def _fallback_parse(text: str):
    """Stdlib-only JSON extraction, used when novel_generator isn't importable
    (e.g. unit tests) or its parser errors."""
    import json
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        t = m.group(0)
    try:
        return json.loads(t)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------
# Technology tiers, coarse but enough to drive the negative list. Ordered oldest
# → newest so "everything up to and including my tier is allowed".
TECH_TIERS = (
    "prehistoric",     # stone tools, fire
    "ancient",         # bronze/iron, classical antiquity
    "medieval",        # ~5th–15th c.: swords, candles, parchment, horses
    "renaissance",     # ~15th–17th c.: early firearms, printing press
    "industrial",      # ~18th–19th c.: steam, gaslight, telegraph, early rail
    "early_modern",    # ~1900–1945: early cars, radio, film, rotary phones
    "modern",          # ~1946–1995: TV, landlines, early computers
    "contemporary",    # ~1996–present: smartphones, internet, laptops
    "near_future",     # plausible extrapolation of contemporary tech
    "far_future",      # speculative / sci-fi tech
)
_TECH_TIER_INDEX = {t: i for i, t in enumerate(TECH_TIERS)}


@dataclass
class PeriodProfile:
    """The story's canonical sense of *when*, and what that permits on screen."""
    era_label: str = ""                  # human label, e.g. "High Middle Ages"
    era_summary: str = ""                # one line the LLM can reason against
    technology_tier: str = "contemporary"
    is_fantastical: bool = False         # secondary-world fantasy (period-flavoured, not literal history)
    is_historical: bool = False          # anchored to real history → strict anachronism rules
    # Vocabulary the model SHOULD lean on (period props / materials / light).
    allowed_vocabulary: List[str] = field(default_factory=list)
    # Specific nouns that must NOT appear — the negative list. Concrete objects
    # a vision/diffusion model can actually render ("smartphone", "wristwatch").
    forbidden_nouns: List[str] = field(default_factory=list)
    # Confidence 0..1 in the classification (LLM sets this; heuristic uses 0.5).
    confidence: float = 0.5

    # ---- convenience -----------------------------------------------------
    def tier_index(self) -> int:
        return _TECH_TIER_INDEX.get(self.technology_tier, _TECH_TIER_INDEX["contemporary"])

    def is_pre_modern(self) -> bool:
        return self.tier_index() < _TECH_TIER_INDEX["early_modern"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "era_label": self.era_label,
            "era_summary": self.era_summary,
            "technology_tier": self.technology_tier,
            "is_fantastical": self.is_fantastical,
            "is_historical": self.is_historical,
            "allowed_vocabulary": list(self.allowed_vocabulary),
            "forbidden_nouns": list(self.forbidden_nouns),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeriodProfile":
        if not isinstance(d, dict):
            return cls()
        return cls(
            era_label=str(d.get("era_label", "") or ""),
            era_summary=str(d.get("era_summary", "") or ""),
            technology_tier=str(d.get("technology_tier", "contemporary") or "contemporary"),
            is_fantastical=bool(d.get("is_fantastical", False)),
            is_historical=bool(d.get("is_historical", False)),
            allowed_vocabulary=[str(x) for x in (d.get("allowed_vocabulary") or []) if str(x).strip()],
            forbidden_nouns=[str(x) for x in (d.get("forbidden_nouns") or []) if str(x).strip()],
            confidence=float(d.get("confidence", 0.5) or 0.5),
        )

    def is_empty(self) -> bool:
        """A contemporary/unknown profile with no forbidden nouns imposes no
        constraint — callers can skip it entirely."""
        return (not self.forbidden_nouns) and self.technology_tier in (
            "contemporary", "near_future", "far_future"
        )


# ---------------------------------------------------------------------------
# Deterministic anachronism catalogue
# ---------------------------------------------------------------------------
# The concrete objects most commonly hallucinated into pre-modern scenes. Each
# entry maps a canonical noun to the regex that catches its common phrasings and
# the tech tier it FIRST becomes plausible at. When a story's tier is below an
# object's tier, that object is forbidden. Grouped so a heuristic profile can
# assemble a forbidden list without an LLM.
# ---------------------------------------------------------------------------
_ANACHRONISM_CATALOGUE: List[Dict[str, Any]] = [
    # noun, first-plausible tier, matching pattern
    {"noun": "smartphone",   "tier": "contemporary", "re": r"\b(smart\s?phones?|iphones?|android phones?)\b"},
    {"noun": "mobile phone", "tier": "early_modern", "re": r"\b(mobile phones?|cell\s?phones?|cellphones?)\b"},
    {"noun": "telephone",    "tier": "industrial",   "re": r"\b(telephones?|rotary phones?|landlines?|phone handsets?)\b"},
    {"noun": "laptop",       "tier": "contemporary", "re": r"\b(laptops?|notebook computers?|macbooks?)\b"},
    {"noun": "computer",     "tier": "modern",       "re": r"\b(computers?|pcs?|desktop computers?|keyboards?|computer mouse|computer mice)\b"},
    {"noun": "monitor / digital screen", "tier": "modern", "re": r"\b(monitors?|digital screens?|lcd screens?|led screens?|flat[- ]screens?|display screens?|touchscreens?)\b"},
    {"noun": "television",   "tier": "early_modern", "re": r"\b(televisions?|tv sets?|\btvs?\b|flat[- ]?screen tv)\b"},
    {"noun": "tablet device","tier": "contemporary", "re": r"\b(tablet computers?|ipads?|touch tablets?)\b"},
    {"noun": "car / automobile","tier": "early_modern","re": r"\b(cars?|automobiles?|motorcars?|sedans?|suvs?|trucks?|vans?|taxis?|buses|motorcycles?)\b"},
    {"noun": "aeroplane",    "tier": "early_modern", "re": r"\b(aeroplanes?|airplanes?|jets?|jetliners?|helicopters?)\b"},
    {"noun": "electric light","tier": "industrial",  "re": r"\b(light\s?bulbs?|electric lights?|led lights?|fluorescent (?:lights?|strips?|tubes?)|neon (?:signs?|lights?)|desk lamps?|floor lamps?|table lamps?)\b"},
    {"noun": "wristwatch",   "tier": "industrial",   "re": r"\b(wrist\s?watches?|wristwatch|digital watches?|smart\s?watches?)\b"},
    {"noun": "wall clock",   "tier": "renaissance",  "re": r"\b(wall clocks?|clock faces?|alarm clocks?|digital clocks?)\b"},
    {"noun": "camera",       "tier": "early_modern", "re": r"\b(cameras?|photographs?|selfies?|polaroids?)\b"},
    {"noun": "firearm/gun",  "tier": "renaissance",  "re": r"\b(guns?|pistols?|rifles?|revolvers?|handguns?|shotguns?|firearms?|machine guns?)\b"},
    {"noun": "eyeglasses",   "tier": "renaissance",  "re": r"\b(eye\s?glasses?|spectacles?|reading glasses?|sunglasses?)\b"},
    {"noun": "printed book", "tier": "renaissance",  "re": r"\b(printed books?|paperbacks?|magazines?|newspapers?)\b"},
    {"noun": "plastic",      "tier": "modern",       "re": r"\b(plastic (?:bottles?|bags?|cups?|containers?)|styrofoam)\b"},
    {"noun": "modern clothing","tier": "early_modern","re": r"\b(t-?shirts?|hoodies?|jeans|denim|sneakers?|trainers?|baseball caps?|business suits?|neckties?|blazers?)\b"},
    {"noun": "modern signage","tier": "early_modern","re": r"\b(traffic lights?|road signs?|street signs?|billboards?|neon signage)\b"},
    {"noun": "power lines",  "tier": "industrial",   "re": r"\b(power lines?|electric wires?|telephone poles?|utility poles?|power outlets?|electrical sockets?|light switches?)\b"},
    {"noun": "modern appliance","tier": "early_modern","re": r"\b(refrigerators?|fridges?|microwaves?|toasters?|blenders?|electric kettles?|washing machines?)\b"},
]
# Pre-compile.
for _e in _ANACHRONISM_CATALOGUE:
    _e["compiled"] = re.compile(_e["re"], re.IGNORECASE)


# Keyword → (tier, is_historical, label) map for the no-LLM heuristic path.
_ERA_KEYWORDS: List[Dict[str, Any]] = [
    {"kw": (r"stone age", r"prehistor", r"neolithic", r"paleolithic", r"ice age", r"cave\s?men"),
     "tier": "prehistoric", "historical": True, "label": "Prehistoric"},
    {"kw": (r"\bancient\b", r"antiquity", r"\broman\b", r"\bgreek\b", r"\bgreece\b", r"\begypt", r"pharaoh",
            r"babylon", r"mesopotam", r"\bbiblical\b", r"bronze age", r"iron age", r"gladiator"),
     "tier": "ancient", "historical": True, "label": "Classical Antiquity"},
    {"kw": (r"medieval", r"middle ages", r"\bdark ages\b", r"\bfeudal\b", r"crusades?", r"\bknights?\b",
            r"castle", r"kingdom of", r"\bvikings?\b", r"\bserf", r"\bfief"),
     "tier": "medieval", "historical": True, "label": "Middle Ages"},
    {"kw": (r"renaissance", r"elizabethan", r"tudor", r"\b1[45]\d\d\b", r"\b1[56]th century\b",
            r"age of sail", r"conquistador", r"musketeer"),
     "tier": "renaissance", "historical": True, "label": "Renaissance / Early Modern"},
    {"kw": (r"victorian", r"industrial revolution", r"steam[- ]?punk", r"\b18\d\d\b", r"\b19th century\b",
            r"regency", r"gaslight", r"wild west", r"\bfrontier\b", r"antebellum", r"dickensian"),
     "tier": "industrial", "historical": True, "label": "19th-Century / Industrial"},
    {"kw": (r"\b19[0-4]\d\b", r"1920s", r"1930s", r"1940s", r"world war (?:i|1|one|ii|2|two)",
            r"roaring twenties", r"great depression", r"art deco", r"noir"),
     "tier": "early_modern", "historical": True, "label": "Early 20th Century"},
    {"kw": (r"\b19[5-9]\d\b", r"cold war", r"1950s", r"1960s", r"1970s", r"1980s", r"1990s"),
     "tier": "modern", "historical": True, "label": "Mid-to-Late 20th Century"},
    {"kw": (r"contemporary", r"present[- ]day", r"modern[- ]day", r"\b20[0-2]\d\b", r"\btoday\b"),
     "tier": "contemporary", "historical": False, "label": "Contemporary"},
    {"kw": (r"near[- ]future", r"cyberpunk", r"dystopian future", r"\b20[3-9]\d\b", r"\b21\d\d\b"),
     "tier": "near_future", "historical": False, "label": "Near Future"},
    {"kw": (r"far future", r"space opera", r"\bstarship", r"interstellar", r"galactic empire",
            r"post[- ]human", r"\bmillennia hence\b"),
     "tier": "far_future", "historical": False, "label": "Far Future"},
]

# Secondary-world fantasy: period-flavoured but not literal history. It should
# still forbid modern tech, but we mark it non-historical so the rules describe
# "the world's technology level" rather than "real history", which reads better
# in the prompt and avoids demanding real-world historical accuracy of a made-up
# realm.
_FANTASY_KEYWORDS = (r"fantasy", r"high fantasy", r"sword and sorcery", r"tolkien", r"elves?",
                     r"\borcs?\b", r"\bdwarves\b", r"\bwizard", r"\bdragons?\b", r"mythic", r"fae\b")

# Allowed-vocabulary seeds per tier — a positive palette the model can lean on.
_TIER_ALLOWED: Dict[str, List[str]] = {
    "prehistoric": ["stone tools", "hide clothing", "open fire", "bone", "cave"],
    "ancient": ["bronze", "marble columns", "oil lamps", "togas or tunics", "clay amphorae", "papyrus scrolls"],
    "medieval": ["candlelight and torchlight", "parchment and quill", "wood and stone", "wrought iron",
                 "woven wool and linen", "horses and carts", "swords and shields", "tapestries"],
    "renaissance": ["candlelight", "early printed books", "oil paintings", "doublets and gowns",
                    "wooden furniture", "sailing ships", "early clockwork"],
    "industrial": ["gaslight", "brass and cast iron", "steam machinery", "horse-drawn carriages",
                   "top hats and frock coats", "telegraph wires", "cobblestone streets"],
    "early_modern": ["early electric light", "wireless radio", "vintage automobiles", "rotary telephones",
                     "tailored period suits and dresses", "film cameras"],
    "modern": ["mid-century furnishings", "landline telephones", "cathode-ray televisions", "period automobiles"],
    "contemporary": [],
    "near_future": ["sleek minimalist tech", "holographic displays", "clean energy infrastructure"],
    "far_future": ["advanced speculative technology", "energy weapons", "starship interiors"],
}


# ---------------------------------------------------------------------------
# Establishing the profile
# ---------------------------------------------------------------------------
_PROFILE_CACHE: Dict[str, PeriodProfile] = {}


def _story_fields(story_idea: Any) -> Dict[str, str]:
    """Pull the era-relevant fields off a StoryIdea object OR a dict."""
    def g(name: str) -> str:
        if isinstance(story_idea, dict):
            v = story_idea.get(name, "")
        else:
            v = getattr(story_idea, name, "")
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v)
        return str(v or "")
    return {
        "genre": g("genre"),
        "setting": g("setting"),
        "premise": g("premise"),
        "mood": g("mood"),
        "themes": g("themes"),
        "title": g("title"),
        "logline": g("logline"),
    }


def _heuristic_profile(fields: Dict[str, str]) -> PeriodProfile:
    """No-LLM classifier: keyword-match the blob to a tier, then assemble the
    forbidden list from the catalogue. Coarse but safe for common eras."""
    blob = " ".join(fields.values()).lower()
    is_fantasy = any(re.search(k, blob) for k in _FANTASY_KEYWORDS)

    chosen = None
    for entry in _ERA_KEYWORDS:
        if any(re.search(k, blob) for k in entry["kw"]):
            chosen = entry
            break
    if chosen is None:
        # Fantasy with no explicit era reads as medieval-tech by convention.
        if is_fantasy:
            chosen = {"tier": "medieval", "historical": False, "label": "Medieval-era fantasy world"}
        else:
            # Default: contemporary, no constraint.
            return PeriodProfile(
                era_label="Contemporary", era_summary="Present-day setting.",
                technology_tier="contemporary", is_historical=False, confidence=0.4,
            )

    tier = chosen["tier"]
    prof = PeriodProfile(
        era_label=("%s (fantasy world)" % chosen["label"]) if is_fantasy else chosen["label"],
        era_summary="Setting inferred from genre/premise keywords.",
        technology_tier=tier,
        is_fantastical=is_fantasy,
        is_historical=bool(chosen.get("historical")) and not is_fantasy,
        allowed_vocabulary=list(_TIER_ALLOWED.get(tier, [])),
        confidence=0.5,
    )
    prof.forbidden_nouns = _forbidden_for_tier(tier)
    return prof


def _forbidden_for_tier(tier: str) -> List[str]:
    """Every catalogue object whose first-plausible tier is newer than `tier`."""
    idx = _TECH_TIER_INDEX.get(tier, _TECH_TIER_INDEX["contemporary"])
    out: List[str] = []
    for e in _ANACHRONISM_CATALOGUE:
        if _TECH_TIER_INDEX.get(e["tier"], 99) > idx:
            out.append(e["noun"])
    return out


_PROFILE_PROMPT = """You are the historical/world consultant for a graphic novel or animated film.
From the story metadata below, determine the SINGLE dominant time-period the main story
takes place in, and what that period allows on screen.

STORY METADATA
Genre: {genre}
Setting: {setting}
Premise: {premise}
Mood: {mood}
Themes: {themes}
Title: {title}

Return ONLY a JSON object, no prose:
{{
  "era_label": "short human label, e.g. 'High Middle Ages', 'Edo-period Japan', '1920s New York', 'far-future space opera'",
  "era_summary": "one sentence a prop artist could reason against",
  "technology_tier": one of {tiers},
  "is_fantastical": true if a secondary/invented world rather than literal Earth history,
  "is_historical": true if anchored to real history (strict anachronism rules apply),
  "allowed_vocabulary": ["6-12 concrete period-appropriate materials/props/light sources/dress"],
  "forbidden_nouns": ["8-20 concrete objects that would be ANACHRONISTIC and must never appear — name the OBJECT, e.g. 'smartphone', 'wristwatch', 'car', 'electric light', 'plastic', 'digital screen'"],
  "confidence": 0.0-1.0
}}

Rules:
- Pick the tier of the MAIN story body. If the story has a framing device in a different
  era (a modern narrator recalling the past), still pick the era the bulk of scenes occur in.
- forbidden_nouns must be RENDERABLE objects a picture could contain, not abstractions.
- For a pre-modern or historical setting, always include the obvious modern intrusions
  (phones, computers, screens, cars, electric lighting, modern clothing) unless the premise
  explicitly introduces them (time-travel, anachronism-as-theme).
- For a contemporary present-day story with no period constraint, return technology_tier
  "contemporary", is_historical false, and an EMPTY forbidden_nouns list.
JSON only."""


def establish_period_profile(
    story_idea: Any,
    llm_fn: Optional[Callable[..., str]] = None,
    parse_json: Optional[Callable[[str], Any]] = None,
    *,
    use_cache: bool = True,
) -> PeriodProfile:
    """Determine the story's canonical era ONCE. LLM-first, heuristic fallback.

    Returns a PeriodProfile. Never raises. On any LLM failure falls back to the
    keyword heuristic, and merges the deterministic catalogue's forbidden list
    into the LLM's (so a model that forgets 'power lines' still gets them).
    """
    fields = _story_fields(story_idea)
    cache_key = "|".join(fields.get(k, "") for k in ("genre", "setting", "premise", "title"))
    if use_cache and cache_key in _PROFILE_CACHE:
        return _PROFILE_CACHE[cache_key]

    heuristic = _heuristic_profile(fields)

    llm = llm_fn or _default_llm
    parse = parse_json or _default_parse
    prof = heuristic
    try:
        prompt = _PROFILE_PROMPT.format(tiers=list(TECH_TIERS), **fields)
        raw = llm(prompt, temperature=0.1, large=True)
        parsed = parse(raw)
        if isinstance(parsed, dict) and parsed.get("technology_tier"):
            llm_prof = PeriodProfile.from_dict(parsed)
            # Merge the deterministic catalogue in so nothing obvious is missed,
            # but only for the tiers the LLM's own tier says are anachronistic.
            catalogue = set(_forbidden_for_tier(llm_prof.technology_tier))
            merged = list(dict.fromkeys(list(llm_prof.forbidden_nouns) + list(catalogue)))
            llm_prof.forbidden_nouns = merged
            if not llm_prof.allowed_vocabulary:
                llm_prof.allowed_vocabulary = list(_TIER_ALLOWED.get(llm_prof.technology_tier, []))
            prof = llm_prof
            logger.info("[Period] Established era: %s (tier=%s, %d forbidden nouns).",
                        prof.era_label, prof.technology_tier, len(prof.forbidden_nouns))
    except Exception as e:
        logger.warning("[Period] LLM era classification failed (%s); using heuristic '%s'.",
                       e, heuristic.era_label)

    if use_cache:
        _PROFILE_CACHE[cache_key] = prof
    return prof


def reset_profile_cache() -> None:
    _PROFILE_CACHE.clear()


# ---------------------------------------------------------------------------
# Per-scene period annotation (story-driven; handles flashbacks / time-jumps)
# ---------------------------------------------------------------------------
# Cues that a SCENE deliberately departs from the story's dominant era. When one
# fires we ask the LLM (cached per setting) which era that scene is in, so the
# lock follows the story rather than blanket-forcing one era.
_ERA_SHIFT_CUES = re.compile(
    r"\b(flash\s?back|flash\s?forward|years? (?:earlier|later|ago|before|hence)|"
    r"decades? (?:earlier|later|ago)|centuries? (?:earlier|later|ago)|"
    r"in the (?:distant )?(?:past|future)|time[- ]?machine|time travel|"
    r"present day|modern day|the year \d{3,4}|\b\d{4}\b|long ago|in ancient|"
    r"dream sequence|vision of the (?:past|future))\b",
    re.IGNORECASE,
)


def _panel_setting_blob(panel: Dict) -> str:
    return " ".join(str(panel.get(k, "") or "") for k in ("setting", "description")).strip()


def annotate_script_with_period(
    script: List[Dict],
    profile: PeriodProfile,
    llm_fn: Optional[Callable[..., str]] = None,
    parse_json: Optional[Callable[[str], Any]] = None,
    *,
    detect_shifts: bool = True,
) -> int:
    """Stamp every panel with the PeriodProfile that is active there.

    Writes ``panel['_period_lock'] = profile.as_dict()`` for each panel. When a
    panel's text carries an explicit era-shift cue (flashback, "years later",
    "the year 2087", time-machine) and an LLM is available, the shifted scene
    gets its OWN profile so its locks/negatives reflect that era, not the
    dominant one. Returns the number of panels stamped.

    Purely additive; no panel field other than ``_period_lock`` (and, on a
    detected shift, ``_period_shift``) is touched.
    """
    if not script:
        return 0
    base = profile or PeriodProfile()
    base_dict = base.as_dict()

    # Cache shifted profiles per unique setting string so a repeated flashback
    # location costs one classification, not one per panel.
    shift_cache: Dict[str, PeriodProfile] = {}
    llm = llm_fn or _default_llm
    parse = parse_json or _default_parse

    stamped = 0
    active = base  # carries forward through a scene once a shift is detected
    prev_setting = None

    for page in script:
        for panel in (page.get("panels") or []):
            blob = _panel_setting_blob(panel)
            setting = str(panel.get("setting", "") or "").strip().lower()

            # A new setting resets to the base era unless this panel re-declares
            # a shift — a flashback ends when the story returns to a normal scene.
            if setting and setting != prev_setting:
                active = base
            prev_setting = setting

            if detect_shifts and blob and _ERA_SHIFT_CUES.search(blob):
                key = setting or blob[:80].lower()
                if key in shift_cache:
                    active = shift_cache[key]
                else:
                    shifted = _classify_shifted_scene(blob, base, llm, parse)
                    shift_cache[key] = shifted
                    active = shifted
                if active is not base:
                    panel["_period_shift"] = active.era_label

            panel["_period_lock"] = active.as_dict() if active is not base else base_dict
            stamped += 1

    return stamped


_SHIFT_PROMPT = """A story is mainly set in: {base_era} (tech level: {base_tier}).
This particular scene appears to jump to a DIFFERENT time period. From the scene text,
identify the era of THIS scene only.

SCENE TEXT:
{scene}

Return ONLY JSON:
{{"same_as_base": true|false,
  "era_label": "...", "technology_tier": one of {tiers},
  "is_historical": true|false}}
If the scene is actually still in the base era (the cue was incidental), set same_as_base true.
JSON only."""


def _classify_shifted_scene(scene_text: str, base: PeriodProfile,
                            llm: Callable[..., str], parse: Callable[[str], Any]) -> PeriodProfile:
    """Return a PeriodProfile for a scene that departs from the base era. Falls
    back to the base profile on any uncertainty/failure (never invents a shift)."""
    try:
        raw = llm(_SHIFT_PROMPT.format(
            base_era=base.era_label or "an unspecified era",
            base_tier=base.technology_tier,
            scene=scene_text[:1200],
            tiers=list(TECH_TIERS),
        ), temperature=0.1, large=False)
        parsed = parse(raw)
        if isinstance(parsed, dict) and not parsed.get("same_as_base", False):
            tier = str(parsed.get("technology_tier") or base.technology_tier)
            prof = PeriodProfile(
                era_label=str(parsed.get("era_label") or "shifted era"),
                era_summary="Scene-local era shift (flashback / time-jump).",
                technology_tier=tier,
                is_fantastical=base.is_fantastical,
                is_historical=bool(parsed.get("is_historical", tier != "contemporary")),
                allowed_vocabulary=list(_TIER_ALLOWED.get(tier, [])),
                forbidden_nouns=_forbidden_for_tier(tier),
                confidence=0.6,
            )
            logger.info("[Period] Scene-local era shift → %s (tier=%s).",
                        prof.era_label, prof.technology_tier)
            return prof
    except Exception as e:
        logger.debug("[Period] shift classification failed (%s); staying in base era.", e)
    return base


# ---------------------------------------------------------------------------
# Reading the lock back off a panel
# ---------------------------------------------------------------------------
def profile_from_panel(panel_script: Optional[Dict]) -> PeriodProfile:
    """Recover the active PeriodProfile stamped on a panel (or an empty one)."""
    if not isinstance(panel_script, dict):
        return PeriodProfile()
    lock = panel_script.get("_period_lock")
    if isinstance(lock, dict):
        return PeriodProfile.from_dict(lock)
    return PeriodProfile()


# ---------------------------------------------------------------------------
# 1) Positive prompt lock
# ---------------------------------------------------------------------------
def period_guard_clause(panel_script: Optional[Dict] = None,
                        profile: Optional[PeriodProfile] = None) -> str:
    """A positive lock naming the era, to append to an image prompt.

    Pass EITHER a panel (reads its stamped ``_period_lock``) or an explicit
    profile. Returns "" when the profile imposes no constraint (contemporary),
    so contemporary stories pay nothing.
    """
    prof = profile or profile_from_panel(panel_script)
    if prof.is_empty():
        return ""
    era = prof.era_label or ("a %s-technology world" % prof.technology_tier)
    world_kind = ("invented world at a %s technology level" % prof.technology_tier
                  if prof.is_fantastical else era)
    allowed = ""
    if prof.allowed_vocabulary:
        allowed = (" Period-appropriate detail only: "
                   + ", ".join(prof.allowed_vocabulary[:10]) + ".")
    return (
        f"PERIOD LOCK — this image is set in {world_kind}. Every object, material, "
        f"garment, light source, vehicle and piece of architecture must belong to "
        f"that period; no later technology or modern intrusions of any kind."
        + allowed
    )


# ---------------------------------------------------------------------------
# 2) Negative prompt terms
# ---------------------------------------------------------------------------
def period_negative_terms(panel_script: Optional[Dict] = None,
                          profile: Optional[PeriodProfile] = None,
                          *, max_terms: int = 40) -> str:
    """Comma-joined anachronistic nouns to add to a NEGATIVE prompt.

    Returns "" for contemporary/unconstrained profiles. Mirrors
    style_negative_terms(): the caller appends this to _adaptive_negative_prompt.
    """
    prof = profile or profile_from_panel(panel_script)
    if prof.is_empty() or not prof.forbidden_nouns:
        return ""
    # Expand catalogue nouns to a few synonyms the model actually renders, so
    # the negative bites (e.g. "smartphone" also pushes "iphone", "cell phone").
    terms: List[str] = []
    seen = set()
    for noun in prof.forbidden_nouns:
        for syn in _negative_synonyms(noun):
            s = syn.strip().lower()
            if s and s not in seen:
                seen.add(s)
                terms.append(s)
        if len(terms) >= max_terms:
            break
    return ", ".join(terms[:max_terms])


# Minimal synonym expansion for the highest-value anachronisms so the negative
# prompt names the tokens a diffusion model actually latches onto.
_NEGATIVE_SYNONYMS: Dict[str, List[str]] = {
    "smartphone": ["smartphone", "cell phone", "iphone", "mobile phone"],
    "mobile phone": ["cell phone", "mobile phone"],
    "telephone": ["telephone", "phone"],
    "laptop": ["laptop", "notebook computer"],
    "computer": ["computer", "desktop computer", "keyboard", "computer monitor"],
    "monitor / digital screen": ["digital screen", "computer screen", "lcd screen", "led screen", "flat screen", "touchscreen"],
    "television": ["television", "tv"],
    "tablet device": ["tablet computer", "ipad"],
    "car / automobile": ["car", "automobile", "modern vehicle", "truck", "bus"],
    "aeroplane": ["airplane", "jet", "helicopter"],
    "electric light": ["electric light", "light bulb", "led light", "fluorescent light", "neon sign", "modern lamp"],
    "wristwatch": ["wristwatch", "smartwatch", "digital watch"],
    "wall clock": ["modern clock", "digital clock"],
    "camera": ["camera", "photograph"],
    "firearm/gun": ["gun", "pistol", "rifle", "modern firearm"],
    "eyeglasses": ["modern eyeglasses", "sunglasses"],
    "printed book": ["printed book", "paperback", "magazine", "newspaper"],
    "plastic": ["plastic", "styrofoam"],
    "modern clothing": ["t-shirt", "jeans", "hoodie", "sneakers", "modern business suit", "baseball cap"],
    "modern signage": ["traffic light", "road sign", "billboard", "neon signage"],
    "power lines": ["power lines", "electrical wires", "power outlet", "light switch"],
    "modern appliance": ["refrigerator", "microwave", "modern appliance"],
}


def _negative_synonyms(noun: str) -> List[str]:
    return _NEGATIVE_SYNONYMS.get(noun, [noun])


# ---------------------------------------------------------------------------
# 3) Pre-render anachronism scan + scrub
# ---------------------------------------------------------------------------
@dataclass
class AnachronismHit:
    noun: str
    matched_text: str
    span: tuple


def scan_anachronisms(text: str,
                      panel_script: Optional[Dict] = None,
                      profile: Optional[PeriodProfile] = None) -> List[AnachronismHit]:
    """Find forbidden objects literally named in ``text`` for the active era.

    Zero LLM cost. Only checks catalogue nouns whose tier is anachronistic for
    the profile, so contemporary stories always return []. Useful both as a QC
    signal (log which panels mention anachronisms) and to drive ``scrub``.
    """
    prof = profile or profile_from_panel(panel_script)
    if not text or prof.is_empty():
        return []
    forbidden = set(prof.forbidden_nouns)
    hits: List[AnachronismHit] = []
    for e in _ANACHRONISM_CATALOGUE:
        if e["noun"] not in forbidden:
            continue
        for m in e["compiled"].finditer(text):
            hits.append(AnachronismHit(noun=e["noun"], matched_text=m.group(0), span=m.span()))
    return hits


_SCRUB_PROMPT = """You are a continuity editor for a work set in {era} (technology level: {tier}).
The image-prompt text below contains one or more ANACHRONISMS — objects that cannot exist
in this period. Rewrite the text so every anachronistic object is replaced by the closest
PERIOD-APPROPRIATE equivalent that still serves the same narrative purpose, OR removed if it
has no equivalent. Change NOTHING else — keep the same characters, action, framing, mood, and
labelled-section structure. Preserve any 'Subject:/Clothing:/Action:/Environment:/...' labels exactly.

Anachronisms detected: {hits}

TEXT:
{text}

Return ONLY the corrected text, no commentary, no code fences."""


def scrub_anachronisms(text: str,
                       panel_script: Optional[Dict] = None,
                       profile: Optional[PeriodProfile] = None,
                       llm_fn: Optional[Callable[..., str]] = None,
                       *,
                       deterministic_fallback: bool = True) -> str:
    """Rewrite ``text`` to remove period anachronisms.

    Strategy: scan first (free). If nothing is found, return the text unchanged.
    If an LLM is available, ask it to substitute period-appropriate equivalents
    (a 'laptop' → 'a ledger and quill' in a medieval scene) — the highest-
    quality fix because it preserves narrative intent. If no LLM (or it fails)
    and ``deterministic_fallback`` is set, fall back to a blunt but safe redaction
    that strips the offending noun phrase. Never raises.
    """
    prof = profile or profile_from_panel(panel_script)
    if not text or prof.is_empty():
        return text or ""
    hits = scan_anachronisms(text, profile=prof)
    if not hits:
        return text

    llm = llm_fn or _default_llm
    try:
        raw = llm(_SCRUB_PROMPT.format(
            era=prof.era_label or "a pre-modern era",
            tier=prof.technology_tier,
            hits=", ".join(sorted({h.noun for h in hits})),
            text=text[:4000],
        ), temperature=0.2, large=True)
        cleaned = _strip_fences(raw)
        if cleaned and len(cleaned) > 10:
            # Verify the rewrite actually removed the anachronisms; if the model
            # left one in, fall through to deterministic redaction on the remainder.
            residual = scan_anachronisms(cleaned, profile=prof)
            if not residual:
                logger.info("[Period] Scrubbed %d anachronism(s) via LLM rewrite.", len(hits))
                return cleaned
            text = cleaned
            hits = residual
    except Exception as e:
        logger.debug("[Period] LLM scrub failed (%s); using deterministic redaction.", e)

    if deterministic_fallback:
        return _redact_hits(text, hits)
    return text


def _redact_hits(text: str, hits: List[AnachronismHit]) -> str:
    """Blunt fallback: replace each anachronistic noun phrase with a neutral,
    period-safe placeholder so nothing modern survives even without an LLM.

    We replace with 'an object' rather than deleting, to avoid leaving dangling
    articles ('she reached for her  and') that would confuse the image model.
    """
    spans = sorted({h.span for h in hits}, key=lambda s: s[0], reverse=True)
    out = text
    _det_re = re.compile(r"(his|her|their|its|the|a|an|our|my|your)\s+$", re.IGNORECASE)
    for start, end in spans:
        # Absorb an immediately-preceding article/possessive so we don't leave a
        # dangling determiner ("his a period-appropriate object").
        head = out[:start]
        m = _det_re.search(head)
        if m:
            start = m.start()
        out = out[:start] + "a period-appropriate object" + out[end:]
    logger.info("[Period] Redacted %d anachronism(s) deterministically (no LLM).", len(spans))
    return out


def _strip_fences(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    return t


# ---------------------------------------------------------------------------
# Post-render check hook (optional) — plugs into comic_book_visual_consistency
# ---------------------------------------------------------------------------
def period_expectation_line(panel_script: Optional[Dict] = None,
                            profile: Optional[PeriodProfile] = None) -> str:
    """A single vision-checkable expectation line for the detective layer.

    Returns "" for unconstrained eras. When non-empty, a caller can add it to
    the PanelConsistencyLedger's spec so the post-render vision pass also flags
    a rendered anachronism ("expected a medieval scene, saw a wall clock").
    """
    prof = profile or profile_from_panel(panel_script)
    if prof.is_empty():
        return ""
    banned = ", ".join(prof.forbidden_nouns[:12])
    return (f"The scene must read as {prof.era_label or prof.technology_tier}: "
            f"NONE of these may be visible — {banned}.")

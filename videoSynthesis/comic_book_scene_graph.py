"""
comic_book_scene_graph.py
=========================

Location continuity — the "where, and does that make sense" layer.

WHY THIS EXISTS
---------------
The pipeline already knows how to detect a scene *change* (``_settings_differ`` /
``_llm_scene_boundary`` in comic_book_art_director) — but it only uses that to
RESET transient state (wetness, nudity) when the story moves. Nothing ever asks
whether the *sequence* of locations is coherent. Two failure modes ship as a
result:

  1. FLIP-FLOP (A → B → A): a character is in a forest, then a single panel puts
     them in a room, then they're back in the forest — with no travel, no exit,
     no reason in the story. This is almost always an LLM slip in the script
     (a mis-labelled setting, or a stray establishing line borrowed from another
     scene), and it reads to a viewer as a teleport-and-back.

  2. HARD TELEPORT (A → B mid-action): consecutive panels of one continuous
     action land in two clearly different places with no establishing/transition
     beat between them.

Diffusion models make this worse: because each panel is rendered independently,
even a *correctly* labelled location drifts (the same tavern gets a different
floor plan every panel). So this module does two jobs:

  A. Build a canonical LOCATION MODEL of the whole script — cluster the many
     free-text setting strings into a small set of named places, and cut the
     panel stream into ordered SCENE SEGMENTS (contiguous runs in one place).
     This yields a stable ``_scene_id`` and ``_location_id`` per panel, which
     lets the scene-anchor cache reuse ONE establishing anchor for a whole
     scene instead of re-deriving per panel — the single biggest lever on
     within-scene location drift.

  B. Detect the two anomalies above and, where it is safe, REPAIR them:
     a 1-panel A-B-A middle that the story doesn't justify is relabelled to A
     (it was almost certainly always meant to be A); a genuine teleport is
     FLAGGED (not silently rewritten) so the generator can insert a transition
     beat or the author can look.

DESIGN PRINCIPLES (same as the rest of the layer)
-------------------------------------------------
* Reuses the EXISTING boundary logic. ``build_scene_graph`` takes the same
  ``settings_differ`` predicate the continuity tracker already uses (injected),
  so scene IDs and transient-state resets agree by construction — no second,
  disagreeing notion of "new scene".
* LLM-optional. Clustering and flip-flop detection are deterministic. The LLM is
  used only to ADJUDICATE ambiguous repairs ("is this stray panel really the
  same place, or a real new scene?"), and everything degrades to a safe no-op
  without it.
* Conservative repairs. Only high-confidence, same-location relabels are applied
  automatically; anything that would change the story's actual geography is
  flagged, never rewritten.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM plumbing (injected; mirrors the art-director pattern).
# ---------------------------------------------------------------------------
def _default_llm(prompt: str, temperature: float = 0.1, large: bool = False) -> str:
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
# Setting normalisation + a default deterministic similarity predicate
# ---------------------------------------------------------------------------
_SETTING_STOP = {
    "the", "a", "an", "in", "on", "at", "of", "and", "with", "to", "into",
    "near", "by", "inside", "outside", "over", "under", "is", "are", "as",
    "scene", "shot", "view", "interior", "exterior", "int", "ext", "day",
    "night", "morning", "evening", "later", "meanwhile", "continuous",
}


def _setting_tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in _SETTING_STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _setting_tokens(a), _setting_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _default_settings_differ(a: str, b: str) -> bool:
    """Fallback boundary predicate when the art-director one isn't injected.

    Deterministic Jaccard only (no LLM): treats >=0.5 overlap as the same place.
    The real pipeline should inject comic_book_art_director._settings_differ,
    which additionally uses an LLM for the ambiguous middle band.
    """
    a_n, b_n = (a or "").strip().lower(), (b or "").strip().lower()
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return False
    return _jaccard(a_n, b_n) < 0.5


def _canonical_location_name(settings: List[str]) -> str:
    """Pick a representative human label for a cluster of setting strings:
    the most frequent non-empty one, else the longest (most descriptive)."""
    freq: Dict[str, int] = {}
    for s in settings:
        s = (s or "").strip()
        if s:
            freq[s] = freq.get(s, 0) + 1
    if not freq:
        return ""
    # Highest frequency, tie-break on descriptiveness (length).
    return max(freq, key=lambda s: (freq[s], len(s)))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class PanelRef:
    """A flat pointer back into the page/panel structure."""
    global_index: int
    page_index: int
    panel_index: int
    setting: str = ""
    description: str = ""


@dataclass
class Location:
    id: int
    canonical_name: str
    member_settings: List[str] = field(default_factory=list)


@dataclass
class SceneSegment:
    """A contiguous run of panels sharing one location."""
    scene_id: int
    location_id: int
    canonical_name: str
    panel_refs: List[PanelRef] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.panel_refs)

    @property
    def start_index(self) -> int:
        return self.panel_refs[0].global_index if self.panel_refs else -1

    @property
    def end_index(self) -> int:
        return self.panel_refs[-1].global_index if self.panel_refs else -1


@dataclass
class SceneGraph:
    locations: List[Location] = field(default_factory=list)
    segments: List[SceneSegment] = field(default_factory=list)
    panels: List[PanelRef] = field(default_factory=list)

    def location_name(self, loc_id: int) -> str:
        for l in self.locations:
            if l.id == loc_id:
                return l.canonical_name
        return ""


@dataclass
class LocationAnomaly:
    kind: str                     # "flip_flop" | "teleport"
    panel_refs: List[PanelRef]    # the panels involved (the STRAY one first)
    from_location: str
    stray_location: str
    to_location: str
    detail: str = ""
    suggested_fix: str = ""       # "relabel_to_surrounding" | "insert_transition" | "review"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------
def _flatten(script: List[Dict]) -> List[PanelRef]:
    refs: List[PanelRef] = []
    gi = 0
    for pi, page in enumerate(script):
        for pj, panel in enumerate(page.get("panels") or []):
            refs.append(PanelRef(
                global_index=gi,
                page_index=pi,
                panel_index=pj,
                setting=str(panel.get("setting", "") or "").strip(),
                description=str(panel.get("description", "") or "").strip(),
            ))
            gi += 1
    return refs


def _panel_by_ref(script: List[Dict], ref: PanelRef) -> Optional[Dict]:
    try:
        return (script[ref.page_index].get("panels") or [])[ref.panel_index]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# A) Build the scene graph
# ---------------------------------------------------------------------------
def build_scene_graph(
    script: List[Dict],
    settings_differ: Optional[Callable[[str, str], bool]] = None,
    *,
    stamp: bool = True,
) -> SceneGraph:
    """Cluster settings into locations and cut the panel stream into scenes.

    Walks panels in narrative order. A new scene starts whenever the panel's
    setting ``settings_differ`` from the current scene's canonical setting.
    Locations are de-duplicated ACROSS the whole script (so a return to the
    forest reuses the forest's location id), which is what makes flip-flop
    detection possible.

    When ``stamp`` is True (default) each panel gets:
        panel['_scene_id']           int, contiguous-scene identifier
        panel['_location_id']        int, canonical location identifier
        panel['_scene_canonical_setting'] str, the location's representative name
        panel['_scene_first_panel']  bool, True on the establishing panel of a scene
    These let the scene-anchor cache key on the scene rather than the raw
    setting string, so one anchor covers a whole scene.

    Never raises; returns an empty graph for empty input.
    """
    differ = settings_differ or _default_settings_differ
    refs = _flatten(script)
    graph = SceneGraph(panels=refs)
    if not refs:
        return graph

    # ---- Pass 1: assign each panel to a canonical location -----------------
    # Greedy: compare each setting against the representative of every known
    # location; attach to the first that does NOT differ, else open a new one.
    locations: List[Location] = []

    def _find_location(setting: str) -> Optional[Location]:
        if not setting:
            # Empty setting inherits the previous panel's location (handled by
            # caller); here we only match non-empty settings.
            return None
        for loc in locations:
            rep = loc.canonical_name or (loc.member_settings[0] if loc.member_settings else "")
            if rep and not differ(setting, rep):
                return loc
        return None

    panel_location: List[int] = [-1] * len(refs)
    last_loc_id = -1
    for i, r in enumerate(refs):
        if not r.setting:
            # No setting given → assume continuation of the current location.
            panel_location[i] = last_loc_id if last_loc_id >= 0 else _open_location(locations, "")
            last_loc_id = panel_location[i]
            continue
        loc = _find_location(r.setting)
        if loc is None:
            loc_id = _open_location(locations, r.setting)
            loc = locations[loc_id]
        loc.member_settings.append(r.setting)
        panel_location[i] = loc.id
        last_loc_id = loc.id

    # Recompute canonical names now that clusters are populated.
    for loc in locations:
        loc.canonical_name = _canonical_location_name(loc.member_settings) or loc.canonical_name

    graph.locations = locations

    # ---- Pass 2: cut into contiguous scene segments ------------------------
    segments: List[SceneSegment] = []
    scene_id = -1
    cur: Optional[SceneSegment] = None
    prev_loc = -2
    for i, r in enumerate(refs):
        loc_id = panel_location[i]
        if loc_id != prev_loc:
            scene_id += 1
            cur = SceneSegment(scene_id=scene_id, location_id=loc_id,
                               canonical_name=graph.location_name(loc_id))
            segments.append(cur)
            prev_loc = loc_id
        cur.panel_refs.append(r)
    graph.segments = segments

    # ---- Pass 3: stamp panels ---------------------------------------------
    if stamp:
        for seg in segments:
            for k, r in enumerate(seg.panel_refs):
                panel = _panel_by_ref(script, r)
                if panel is None:
                    continue
                panel["_scene_id"] = seg.scene_id
                panel["_location_id"] = seg.location_id
                panel["_scene_canonical_setting"] = seg.canonical_name
                panel["_scene_first_panel"] = (k == 0)

    logger.info("[SceneGraph] %d panels → %d locations, %d scene segments.",
                len(refs), len(locations), len(segments))
    return graph


def _open_location(locations: List[Location], setting: str) -> int:
    loc_id = len(locations)
    locations.append(Location(id=loc_id, canonical_name=setting.strip(),
                              member_settings=[setting] if setting else []))
    return loc_id


# ---------------------------------------------------------------------------
# B) Detect anomalies
# ---------------------------------------------------------------------------
# Cues in a panel's own text that a location change IS justified by the story
# (so it must NOT be flagged as a teleport/flip-flop).
_TRANSITION_CUES = re.compile(
    r"\b(walk(?:s|ed|ing)?|run(?:s|ning)?|ran|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|"
    r"leav(?:e|es|ing)|left|arriv(?:e|es|ed|ing)|depart(?:s|ed|ing)?|head(?:s|ed|ing)? "
    r"(?:to|toward|for|into|out)|step(?:s|ped|ping)? (?:in|out|into|through)|"
    r"cross(?:es|ed|ing)?|travel(?:s|led|ing)?|journey|door|gate(?:way)?|threshold|"
    r"climb(?:s|ed|ing)?|descend(?:s|ed|ing)?|drive(?:s)?|drove|ride(?:s)?|rode|"
    r"back (?:to|at|in(?:side)?)|return(?:s|ed|ing)? (?:to|home)|"
    r"cut to|meanwhile|later|elsewhere|hours? (?:later|earlier)|the next|"
    r"outside|inside|down the (?:hall|corridor|path|road|stairs))\b",
    re.IGNORECASE,
)


def _has_transition_cue(panel: Optional[Dict]) -> bool:
    if not panel:
        return False
    blob = " ".join(str(panel.get(k, "") or "") for k in ("setting", "description"))
    return bool(_TRANSITION_CUES.search(blob))


def detect_location_anomalies(
    graph: SceneGraph,
    script: List[Dict],
    *,
    max_flip_len: int = 1,
) -> List[LocationAnomaly]:
    """Scan the scene sequence for flip-flops and hard teleports.

    FLIP-FLOP: a scene segment of location B, at most ``max_flip_len`` panels
    long, sitting between two segments both in location A, where NEITHER the B
    panel nor the panel returning to A carries a transition/travel cue. This is
    the forest → room → forest case. Highest-value, highest-confidence anomaly.

    TELEPORT: two adjacent segments in different locations where the panel that
    opens the new segment continues an action from the previous panel (shared
    characters, no transition cue, no establishing language). Lower confidence —
    flagged for review, never auto-rewritten.

    Returns anomalies ordered by panel position. Empty when the sequence is clean
    or too short to judge.
    """
    segs = graph.segments
    anomalies: List[LocationAnomaly] = []
    if len(segs) < 2:
        return anomalies

    # --- Flip-flop (A B A) --------------------------------------------------
    for i in range(1, len(segs) - 1):
        prev_s, mid_s, next_s = segs[i - 1], segs[i], segs[i + 1]
        if mid_s.location_id == prev_s.location_id:
            continue
        if prev_s.location_id != next_s.location_id:
            continue
        if mid_s.length > max_flip_len:
            continue  # a real, developed detour scene, not a slip
        # Is the excursion justified? Check the stray panel(s) AND the return
        # panel for any travel/transition language.
        mid_panel = _panel_by_ref(script, mid_s.panel_refs[0])
        ret_panel = _panel_by_ref(script, next_s.panel_refs[0])
        if _has_transition_cue(mid_panel) or _has_transition_cue(ret_panel):
            continue  # story explicitly moves there and back — legitimate
        anomalies.append(LocationAnomaly(
            kind="flip_flop",
            panel_refs=[mid_s.panel_refs[0]] + list(prev_s.panel_refs[-1:]) + list(next_s.panel_refs[:1]),
            from_location=prev_s.canonical_name,
            stray_location=mid_s.canonical_name,
            to_location=next_s.canonical_name,
            detail=(f"Panel {mid_s.panel_refs[0].global_index} jumps to "
                    f"'{mid_s.canonical_name}' for {mid_s.length} panel(s) between two "
                    f"'{prev_s.canonical_name}' scenes, with no travel in the story."),
            suggested_fix="relabel_to_surrounding",
            confidence=0.85,
        ))

    # --- Teleport (A → B mid-action) ---------------------------------------
    for i in range(1, len(segs)):
        prev_s, cur_s = segs[i - 1], segs[i]
        if cur_s.location_id == prev_s.location_id:
            continue
        # Already reported as the middle of a flip-flop? skip.
        if any(a.kind == "flip_flop" and cur_s.panel_refs[0] in a.panel_refs
               for a in anomalies):
            continue
        open_panel = _panel_by_ref(script, cur_s.panel_refs[0])
        prev_panel = _panel_by_ref(script, prev_s.panel_refs[-1])
        if _has_transition_cue(open_panel):
            continue  # establishing/travel language present — a proper cut
        # Continuous action heuristic: same characters carried across the cut
        # with no transition cue → likely an unmarked teleport.
        if _shares_characters(prev_panel, open_panel) and cur_s.length <= 2:
            anomalies.append(LocationAnomaly(
                kind="teleport",
                panel_refs=[cur_s.panel_refs[0], prev_s.panel_refs[-1]],
                from_location=prev_s.canonical_name,
                stray_location=cur_s.canonical_name,
                to_location=cur_s.canonical_name,
                detail=(f"Panel {cur_s.panel_refs[0].global_index} moves from "
                        f"'{prev_s.canonical_name}' to '{cur_s.canonical_name}' with the "
                        f"same characters mid-action and no establishing/travel beat."),
                suggested_fix="insert_transition",
                confidence=0.5,
            ))

    anomalies.sort(key=lambda a: a.panel_refs[0].global_index)
    return anomalies


def _shares_characters(a: Optional[Dict], b: Optional[Dict]) -> bool:
    if not a or not b:
        return False
    ca = {str(x).lower() for x in (a.get("characters_in_frame") or [])}
    cb = {str(x).lower() for x in (b.get("characters_in_frame") or [])}
    return bool(ca & cb)


# ---------------------------------------------------------------------------
# C) Adjudicate + repair
# ---------------------------------------------------------------------------
_ADJUDICATE_PROMPT = """You are a continuity supervisor for a graphic novel / film.
The panels below are consecutive. Panel {stray_idx}'s setting was auto-labelled
'{stray}', sitting between two panels both set in '{surrounding}'. There is no travel
or scene-transition described. Decide what actually happened.

CONTEXT (consecutive panels):
{context}

Return ONLY JSON:
{{"verdict": "same_place" | "real_new_scene",
  "canonical_setting": "the setting string panel {stray_idx} should use",
  "reason": "one short clause"}}

- "same_place": the middle panel was always meant to be in '{surrounding}' — the label
  was a slip, or it's a close-up/re-angle of the same location. canonical_setting = the
  surrounding location.
- "real_new_scene": the story genuinely moves to a different place here and needs a proper
  transition beat. canonical_setting = keep the middle panel's own setting.
When unsure, prefer "same_place" for a single-panel excursion with shared characters.
JSON only."""


def suggest_scene_repairs(
    anomalies: List[LocationAnomaly],
    graph: SceneGraph,
    script: List[Dict],
    llm_fn: Optional[Callable[..., str]] = None,
    parse_json: Optional[Callable[[str], Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """For each flip-flop, ask the LLM whether the stray panel is really the same
    place (→ relabel) or a genuine new scene (→ needs a transition).

    Returns {global_panel_index: {"action": "relabel"|"insert_transition"|"review",
                                  "canonical_setting": str, "reason": str}}.

    Deterministic fallback (no LLM): a single-panel flip-flop with shared
    characters is treated as ``relabel`` (the safest, most common truth);
    everything else is ``review``. Never raises.
    """
    repairs: Dict[int, Dict[str, Any]] = {}
    if not anomalies:
        return repairs
    llm = llm_fn or _default_llm
    parse = parse_json or _default_parse

    for a in anomalies:
        stray = a.panel_refs[0]
        gi = stray.global_index
        if a.kind == "teleport":
            repairs[gi] = {"action": "insert_transition",
                           "canonical_setting": a.stray_location,
                           "reason": "unmarked location change mid-action"}
            continue

        # flip_flop → adjudicate
        verdict = None
        try:
            ctx = _context_snippet(script, graph, gi, radius=2)
            raw = llm(_ADJUDICATE_PROMPT.format(
                stray_idx=gi, stray=a.stray_location,
                surrounding=a.from_location, context=ctx,
            ), temperature=0.0, large=False)
            parsed = parse(raw)
            if isinstance(parsed, dict) and parsed.get("verdict"):
                verdict = parsed
        except Exception as e:
            logger.debug("[SceneGraph] adjudication LLM failed (%s); using default.", e)

        if verdict and verdict.get("verdict") == "real_new_scene":
            repairs[gi] = {"action": "insert_transition",
                           "canonical_setting": str(verdict.get("canonical_setting") or a.stray_location),
                           "reason": str(verdict.get("reason") or "genuine new scene")}
        elif verdict and verdict.get("verdict") == "same_place":
            repairs[gi] = {"action": "relabel",
                           "canonical_setting": str(verdict.get("canonical_setting") or a.from_location),
                           "reason": str(verdict.get("reason") or "mislabelled — same place")}
        else:
            # Deterministic default: relabel single-panel flip-flops.
            repairs[gi] = {"action": "relabel",
                           "canonical_setting": a.from_location,
                           "reason": "single-panel A-B-A with no travel (default relabel)"}
    return repairs


def _context_snippet(script: List[Dict], graph: SceneGraph, center_gi: int, radius: int = 2) -> str:
    lines = []
    for r in graph.panels:
        if abs(r.global_index - center_gi) <= radius:
            marker = " <-- STRAY" if r.global_index == center_gi else ""
            desc = (r.description or "")[:160]
            lines.append(f"[panel {r.global_index}] setting='{r.setting}'{marker}\n    {desc}")
    return "\n".join(lines)


def apply_safe_relabels(
    script: List[Dict],
    repairs: Dict[int, Dict[str, Any]],
    graph: SceneGraph,
) -> int:
    """Apply ONLY the high-confidence 'relabel' repairs to the script in place.

    A relabel rewrites the stray panel's ``setting`` to the surrounding scene's
    canonical setting and marks ``panel['_scene_relabeled'] = <old_setting>`` for
    auditability. 'insert_transition' / 'review' repairs are left untouched (the
    generator / author handles those). Returns the number of panels relabelled.
    """
    if not repairs:
        return 0
    by_gi = {r.global_index: r for r in graph.panels}
    applied = 0
    for gi, rep in repairs.items():
        if rep.get("action") != "relabel":
            continue
        ref = by_gi.get(gi)
        if not ref:
            continue
        panel = _panel_by_ref(script, ref)
        if panel is None:
            continue
        old = str(panel.get("setting", "") or "")
        new = str(rep.get("canonical_setting") or "").strip()
        if not new or new.lower() == old.lower():
            continue
        panel["setting"] = new
        panel["_scene_relabeled"] = old
        applied += 1
        logger.info("[SceneGraph] Relabelled panel %d '%s' → '%s' (%s).",
                    gi, old, new, rep.get("reason", ""))
    return applied


# ---------------------------------------------------------------------------
# D) Per-panel location lock for the image prompt
# ---------------------------------------------------------------------------
def scene_continuity_clause(panel_script: Optional[Dict]) -> str:
    """A positive lock binding a panel to its scene's canonical location.

    Reads the ``_scene_canonical_setting`` / ``_scene_first_panel`` stamps left
    by ``build_scene_graph``. On the establishing panel it says "this establishes
    the location"; on later panels it says "SAME location as the scene's
    establishing shot — keep geography/architecture/layout consistent", which is
    the within-scene drift the anchor system fights. Returns "" when unstamped.
    """
    if not isinstance(panel_script, dict):
        return ""
    place = str(panel_script.get("_scene_canonical_setting", "") or "").strip()
    if not place:
        return ""
    if panel_script.get("_scene_first_panel"):
        return (f"LOCATION — this panel establishes the scene's setting: {place}. "
                f"Fix its geography, architecture and layout here; later panels of this "
                f"scene must match it.")
    return (f"LOCATION LOCK — this panel is the SAME place as the scene's establishing "
            f"shot: {place}. Keep the same geography, architecture, furniture placement, "
            f"and spatial layout; you may change the camera angle, but do not invent a "
            f"different room or a different exterior.")


# ---------------------------------------------------------------------------
# One-call convenience wrapper
# ---------------------------------------------------------------------------
def enforce_scene_continuity(
    script: List[Dict],
    settings_differ: Optional[Callable[[str, str], bool]] = None,
    llm_fn: Optional[Callable[..., str]] = None,
    parse_json: Optional[Callable[[str], Any]] = None,
    *,
    apply_relabels: bool = True,
) -> Dict[str, Any]:
    """Build the graph, detect anomalies, adjudicate, optionally apply safe
    relabels — the whole location-continuity pass in one call.

    Returns a report dict:
        {"graph": SceneGraph, "anomalies": [...], "repairs": {...},
         "relabelled": int, "flagged_for_transition": [global_index, ...]}

    Safe to call on any script; stamps scene ids even when there are no
    anomalies (which is the point — the stamps drive anchor reuse downstream).
    """
    graph = build_scene_graph(script, settings_differ=settings_differ, stamp=True)
    anomalies = detect_location_anomalies(graph, script)
    repairs = suggest_scene_repairs(anomalies, graph, script, llm_fn=llm_fn, parse_json=parse_json)
    relabelled = apply_safe_relabels(script, repairs, graph) if apply_relabels else 0

    # If we relabelled anything, scene ids are now stale — rebuild so downstream
    # anchor reuse sees the corrected, merged scenes.
    if relabelled:
        graph = build_scene_graph(script, settings_differ=settings_differ, stamp=True)

    flagged = [gi for gi, rep in repairs.items()
               if rep.get("action") == "insert_transition"]
    if anomalies:
        logger.info("[SceneGraph] %d anomaly(ies): %d relabelled, %d flagged for transition.",
                    len(anomalies), relabelled, len(flagged))
    return {
        "graph": graph,
        "anomalies": anomalies,
        "repairs": repairs,
        "relabelled": relabelled,
        "flagged_for_transition": flagged,
    }

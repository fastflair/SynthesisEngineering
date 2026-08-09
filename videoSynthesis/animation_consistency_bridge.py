"""
animation_consistency_bridge.py
===============================

Wire the comic-side consistency machinery into the ANIMATION pipeline.

WHY THIS EXISTS
---------------
``story_to_animation.py`` builds its own image prompts (``generate_image_prompts``
/ ``build_image_prompt``) from *static* character data — ``Character.costume``,
``Character.appearance`` — and never runs the mature trackers that the comic
generator uses. Concretely, in the animation path:

  * ``self.visual_bible = None`` (the SeriesVisualBible is never built);
  * ``shots_from_comic_script`` drops the panel's ``_appearance_state`` /
    ``_appearance_changes`` / ``_scene_anchor`` annotations on the floor;
  * costume is read from the registry baseline, so a jacket removed in scene 2
    is back on in scene 3, and a scene's geography drifts every shot.

That means all of the following, which already work for the printed comic, are
LOST for the film unless the comic generator happened to pre-bake a ``prompt``
string for every shot:

  * evolved wardrobe persistence ("suit stays a suit until they change"),
  * scene/location consistency (one canonical place per scene),
  * time-period consistency (no anachronisms).

This bridge closes that gap. It reconstructs a minimal panel-script view from
the flat ``Shot`` list, runs the three trackers over it
(``track_appearance_continuity`` from the art director, plus the two new
modules ``comic_book_period_consistency`` and ``comic_book_scene_graph``), and
copies the resulting per-shot locks back onto each ``Shot`` so the animation's
own prompt builder can inject them.

It also provides the I2V-specific continuity lever the comic never needed:
``should_carry_last_frame`` / ``carry_reference_for_shot`` — when consecutive
shots share a location and continuous action, seeding shot N's keyframe from the
LAST FRAME of shot N-1's clip is the strongest possible anti-drift signal,
because the model literally starts from the previous image.

DESIGN
------
* Everything is injected and optional. If the art-director module isn't
  importable, or no LLM is available, each step degrades to a safe no-op and the
  animation pipeline behaves exactly as it does today.
* Nothing here renders or calls a video engine. It only PREPARES data (locks,
  reference decisions). The animation pipeline stays in charge of generation.
* Shots are annotated via a single ``shot._consistency`` dict (added
  dynamically); no existing ``Shot`` field is touched, so it's backward
  compatible with every downstream consumer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft imports — degrade to no-op if a dependency is missing.
# ---------------------------------------------------------------------------
def _import_art_director():
    try:
        import comic_book_art_director as ad  # type: ignore
        return ad
    except Exception as e:  # pragma: no cover
        logger.info("[AnimBridge] comic_book_art_director unavailable (%s); "
                    "appearance/scene-anchor locks disabled.", e)
        return None


def _import_period():
    try:
        import comic_book_period_consistency as period  # type: ignore
        return period
    except Exception as e:  # pragma: no cover
        logger.info("[AnimBridge] comic_book_period_consistency unavailable (%s); "
                    "period locks disabled.", e)
        return None


def _import_scene_graph():
    try:
        import comic_book_scene_graph as sg  # type: ignore
        return sg
    except Exception as e:  # pragma: no cover
        logger.info("[AnimBridge] comic_book_scene_graph unavailable (%s); "
                    "scene-graph locks disabled.", e)
        return None


# ---------------------------------------------------------------------------
# Registry shim — gives the appearance tracker a wardrobe baseline from plain
# Character objects when the full CharacterAppearanceRegistry isn't supplied.
# The tracker only calls get_clothing / get_appearance / get_gender on it.
# ---------------------------------------------------------------------------
class _CharacterRegistryShim:
    """Minimal duck-typed registry built from a list of Character objects/dicts.

    Exposes exactly the three lookups ``AppearanceContinuityTracker._seed_state``
    uses. Names are matched case-insensitively and on first token, so
    'Cedric' resolves whether the frame lists 'Cedric' or 'Sir Cedric'.
    """

    def __init__(self, characters: List[Any]):
        self._clothing: Dict[str, str] = {}
        self._appearance: Dict[str, str] = {}
        self._gender: Dict[str, str] = {}
        for c in (characters or []):
            def g(attr: str) -> str:
                v = getattr(c, attr, None)
                if v is None and isinstance(c, dict):
                    v = c.get(attr)
                if isinstance(v, (list, tuple)):
                    v = ", ".join(str(x) for x in v)
                return str(v or "")
            name = g("name")
            if not name:
                continue
            key = name.strip().lower()
            self._clothing[key] = g("costume") or g("outfit") or g("clothing")
            self._appearance[key] = g("appearance") or g("physical_description")
            self._gender[key] = g("gender") or "figure"

    def _lookup(self, table: Dict[str, str], name: str) -> str:
        if not name:
            return ""
        key = name.strip().lower()
        if key in table:
            return table[key]
        # First-token / containment fallback.
        first = key.split()[0] if key.split() else key
        for k, v in table.items():
            if k == first or first in k or k in key:
                return v
        return ""

    def get_clothing(self, name: str) -> str:
        return self._lookup(self._clothing, name)

    def get_appearance(self, name: str) -> str:
        return self._lookup(self._appearance, name)

    def get_gender(self, name: str) -> str:
        return self._lookup(self._gender, name) or "figure"


# ---------------------------------------------------------------------------
# Shot <-> panel-script view
# ---------------------------------------------------------------------------
def _shots_to_script(shots: List[Any]) -> List[Dict]:
    """Reconstruct a single-page panel-script view from a flat shot list.

    The trackers operate on the comic ``script`` shape (pages → panels). A film
    is one long ordered stream, so we present it as ONE page whose panels are the
    shots in order. Each panel mirrors the fields the trackers read:
    ``setting``, ``description``, ``characters_in_frame``, ``panel_index``.
    The shot's ``index`` is stored so we can copy annotations back precisely.
    """
    panels: List[Dict] = []
    for sh in shots:
        panels.append({
            "_shot_index": getattr(sh, "index", len(panels)),
            "panel_index": getattr(sh, "index", len(panels)),
            "setting": str(getattr(sh, "setting", "") or ""),
            "description": str(getattr(sh, "description", "") or ""),
            "mood": str(getattr(sh, "mood", "") or ""),
            "characters_in_frame": list(getattr(sh, "characters_in_frame", []) or []),
        })
    return [{"panels": panels}]


def _copy_panel_to_shot(panel: Dict, shot: Any) -> None:
    """Copy the tracker-written annotations off a panel onto the shot."""
    store = getattr(shot, "_consistency", None)
    if not isinstance(store, dict):
        store = {}
        try:
            setattr(shot, "_consistency", store)
        except Exception:
            return
    for key in ("_appearance_state", "_appearance_changes", "_period_lock",
                "_period_shift", "_scene_id", "_location_id",
                "_scene_canonical_setting", "_scene_first_panel",
                "_scene_relabeled", "_scene_anchor"):
        if key in panel:
            store[key] = panel[key]
    # If the scene graph relabelled the setting, reflect it on the shot too so
    # the animation's own setting-based logic agrees with the corrected scene.
    if panel.get("_scene_relabeled") and panel.get("setting"):
        try:
            setattr(shot, "setting", panel["setting"])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The one-call preparation pass
# ---------------------------------------------------------------------------
def prepare_shots_for_animation(
    shots: List[Any],
    characters: List[Any],
    story_idea: Any = None,
    *,
    registry: Any = None,
    visual_bible: Any = None,
    graph: Any = None,
    llm_fn: Optional[Callable[..., str]] = None,
    parse_json: Optional[Callable[[str], Any]] = None,
    enable_appearance: bool = True,
    enable_period: bool = True,
    enable_scene_graph: bool = True,
) -> Dict[str, Any]:
    """Run appearance + period + scene-graph tracking over the shot list and
    annotate each shot with its resolved consistency locks.

    Call this ONCE, after the shot list exists and before ``generate_image_
    prompts`` / ``build_image_prompt``. It mutates shots in place (adding a
    ``shot._consistency`` dict) and returns a report:

        {"period_profile": PeriodProfile|None,
         "scene_report": {...}|None,
         "appearance_ran": bool,
         "shots": <same list>}

    Every stage is independently guarded: a failure or missing dependency in one
    leaves the others intact and never raises.
    """
    report: Dict[str, Any] = {
        "period_profile": None, "scene_report": None,
        "appearance_ran": False, "shots": shots,
    }
    if not shots:
        return report

    script = _shots_to_script(shots)

    # --- Scene graph FIRST: it may relabel settings, and appearance tracking
    #     keys its scene resets off settings, so we want the corrected ones. ---
    if enable_scene_graph:
        sg = _import_scene_graph()
        if sg is not None:
            try:
                ad = _import_art_director()
                differ = getattr(ad, "_settings_differ", None) if ad else None
                report["scene_report"] = sg.enforce_scene_continuity(
                    script, settings_differ=differ,
                    llm_fn=llm_fn, parse_json=parse_json, apply_relabels=True,
                )
            except Exception as e:
                logger.warning("[AnimBridge] scene-graph pass failed (%s); skipped.", e)

    # --- Period: establish once, stamp per panel (handles flashbacks). ---
    if enable_period and story_idea is not None:
        period = _import_period()
        if period is not None:
            try:
                profile = period.establish_period_profile(
                    story_idea, llm_fn=llm_fn, parse_json=parse_json)
                period.annotate_script_with_period(
                    script, profile, llm_fn=llm_fn, parse_json=parse_json)
                report["period_profile"] = profile
            except Exception as e:
                logger.warning("[AnimBridge] period pass failed (%s); skipped.", e)

    # --- Appearance: evolve wardrobe/conditions across the whole stream. ---
    if enable_appearance:
        ad = _import_art_director()
        if ad is not None and hasattr(ad, "track_appearance_continuity"):
            try:
                # Give the tracker a wardrobe baseline. Prefer the real registry;
                # otherwise build a shim from the Character objects so costume
                # persistence works without the full CharacterAppearanceRegistry.
                _registry = registry or _CharacterRegistryShim(characters or [])
                ad.track_appearance_continuity(
                    script, characters or [], registry=_registry,
                    visual_bible=visual_bible, graph=graph)
                report["appearance_ran"] = True
            except Exception as e:
                logger.warning("[AnimBridge] appearance tracking failed (%s); skipped.", e)

    # --- Copy every annotation back onto the shots. ---
    panels = script[0].get("panels") or []
    by_shot_index = {p.get("_shot_index"): p for p in panels}
    for sh in shots:
        panel = by_shot_index.get(getattr(sh, "index", None))
        if panel is not None:
            _copy_panel_to_shot(panel, sh)

    logger.info("[AnimBridge] Prepared %d shots (appearance=%s, period=%s, scene_graph=%s).",
                len(shots), report["appearance_ran"],
                report["period_profile"] is not None,
                report["scene_report"] is not None)
    return report


# ---------------------------------------------------------------------------
# Reading the locks back for the image-prompt builder
# ---------------------------------------------------------------------------
def _store(shot: Any) -> Dict[str, Any]:
    s = getattr(shot, "_consistency", None)
    return s if isinstance(s, dict) else {}


def shot_consistency_locks(shot: Any) -> str:
    """Assemble the combined lock block for one shot's image prompt.

    Concatenates, in priority order:
      1. the evolved appearance lock (wardrobe/conditions/held items/markings),
      2. the scene/location lock (same place as the scene's establishing shot),
      3. the period lock (era-appropriate objects only).

    Returns "" when the shot has no tracked locks (the bridge didn't run, or the
    shot has no constrained state), so callers can append unconditionally.
    """
    store = _store(shot)
    if not store:
        return ""
    parts: List[str] = []

    # 1. Appearance — reuse the art director's own clause builder so the wording
    #    is identical to the comic's, honouring in-panel transitions.
    ad = _import_art_director()
    if ad is not None and store.get("_appearance_state"):
        try:
            panel_view = {
                "_appearance_state": store.get("_appearance_state"),
                "_appearance_changes": store.get("_appearance_changes") or [],
            }
            chars = list((store.get("_appearance_state") or {}).keys())
            clause = ad.appearance_continuity_clause(panel_view, chars)
            if clause:
                parts.append(clause)
        except Exception as e:
            logger.debug("[AnimBridge] appearance clause failed (%s).", e)

    # 2. Scene / location.
    sg = _import_scene_graph()
    if sg is not None:
        try:
            clause = sg.scene_continuity_clause({
                "_scene_canonical_setting": store.get("_scene_canonical_setting", ""),
                "_scene_first_panel": store.get("_scene_first_panel", False),
            })
            if clause:
                parts.append(clause)
        except Exception as e:
            logger.debug("[AnimBridge] scene clause failed (%s).", e)

    # 3. Period.
    period = _import_period()
    if period is not None and store.get("_period_lock"):
        try:
            clause = period.period_guard_clause({"_period_lock": store.get("_period_lock")})
            if clause:
                parts.append(clause)
        except Exception as e:
            logger.debug("[AnimBridge] period clause failed (%s).", e)

    return "\n".join(parts)


def shot_period_negative_terms(shot: Any) -> str:
    """Era anachronism terms for a shot's NEGATIVE prompt (or "")."""
    store = _store(shot)
    if not store or not store.get("_period_lock"):
        return ""
    period = _import_period()
    if period is None:
        return ""
    try:
        return period.period_negative_terms({"_period_lock": store.get("_period_lock")})
    except Exception:
        return ""


def scrub_shot_prompt(shot: Any, image_prompt: str,
                      llm_fn: Optional[Callable[..., str]] = None) -> str:
    """Remove any period anachronisms literally present in a shot's image prompt.

    A thin wrapper over ``comic_book_period_consistency.scrub_anachronisms`` that
    reads the shot's stamped era. No-op for unconstrained eras / when the module
    is unavailable.
    """
    store = _store(shot)
    if not store or not store.get("_period_lock") or not image_prompt:
        return image_prompt
    period = _import_period()
    if period is None:
        return image_prompt
    try:
        return period.scrub_anachronisms(
            image_prompt, {"_period_lock": store.get("_period_lock")}, llm_fn=llm_fn)
    except Exception:
        return image_prompt


# ---------------------------------------------------------------------------
# I2V continuity — last-frame carry between continuous same-location shots
# ---------------------------------------------------------------------------
def should_carry_last_frame(prev_shot: Any, shot: Any) -> bool:
    """True when shot N should be seeded from shot N-1's LAST frame.

    Heuristic: same canonical scene id AND overlapping characters AND the new
    shot has no travel/transition cue — i.e. a continuous beat within one place.
    In that case starting the I2V keyframe from the previous clip's final frame
    is the strongest anti-drift signal available (the model literally continues
    the same image). We deliberately DON'T carry across a scene cut, where a
    fresh establishing keyframe is wanted.
    """
    if prev_shot is None or shot is None:
        return False
    ps, cs = _store(prev_shot), _store(shot)
    # Same scene?
    if ps.get("_scene_id") is None or cs.get("_scene_id") is None:
        # Fall back to raw setting equality when scene ids weren't stamped.
        same = (str(getattr(prev_shot, "setting", "")).strip().lower()
                == str(getattr(shot, "setting", "")).strip().lower())
        if not same:
            return False
    elif ps.get("_scene_id") != cs.get("_scene_id"):
        return False
    # First panel of a scene wants a fresh establishing frame.
    if cs.get("_scene_first_panel"):
        return False
    # Overlapping cast.
    a = {str(x).lower() for x in (getattr(prev_shot, "characters_in_frame", []) or [])}
    b = {str(x).lower() for x in (getattr(shot, "characters_in_frame", []) or [])}
    if not (a & b):
        return False
    # A transition cue means a deliberate re-frame — don't carry.
    sg = _import_scene_graph()
    if sg is not None:
        try:
            if sg._has_transition_cue({
                "setting": getattr(shot, "setting", ""),
                "description": getattr(shot, "description", ""),
            }):
                return False
        except Exception:
            pass
    return True


def carry_reference_for_shot(prev_shot: Any, shot: Any,
                             *, extract_last_frame: Optional[Callable[[str], Optional[str]]] = None
                             ) -> Optional[str]:
    """Return a reference IMAGE PATH to seed ``shot``'s keyframe, or None.

    When ``should_carry_last_frame`` holds and the previous shot produced a
    video, extract its last frame (via the injected ``extract_last_frame``
    callable, or a built-in ffmpeg one-liner) and return that path so the caller
    can pass it as the I2V init image. Returns None when carry isn't warranted or
    the previous clip is missing.
    """
    if not should_carry_last_frame(prev_shot, shot):
        return None
    prev_video = getattr(prev_shot, "video_path", None)
    prev_image = getattr(prev_shot, "image_path", None)
    if prev_video and os.path.exists(prev_video):
        extractor = extract_last_frame or _ffmpeg_last_frame
        try:
            frame = extractor(prev_video)
            if frame and os.path.exists(frame):
                return frame
        except Exception as e:
            logger.debug("[AnimBridge] last-frame extract failed (%s).", e)
    # Fall back to the previous shot's own keyframe if no video yet.
    if prev_image and os.path.exists(prev_image):
        return prev_image
    return None


def _ffmpeg_last_frame(video_path: str) -> Optional[str]:
    """Extract the final frame of a video to a sibling .png via ffmpeg."""
    import subprocess
    out = os.path.splitext(video_path)[0] + "_lastframe.png"
    try:
        # -sseof -0.1 seeks near the end; -update writes the single last frame.
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
             "-update", "1", "-frames:v", "1", out],
            check=True, capture_output=True,
        )
        return out if os.path.exists(out) else None
    except Exception as e:
        logger.debug("[AnimBridge] ffmpeg last-frame failed (%s).", e)
        return None

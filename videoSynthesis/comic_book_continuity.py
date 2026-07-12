"""
comic_book_continuity.py
========================
World-State Continuity pass for the comic book pipeline.

This module is the SECOND continuity layer. It complements the character-centric
``AppearanceContinuityTracker`` in ``comic_book_art_director.py`` by tracking the
three things that tracker deliberately does NOT:

  1. WORLD OBJECTS (props, vehicles, structures, machines, set dressing).
     The Series Visual Bible already gives every recurring object a *canonical*
     description, but that description is FROZEN — a "glossy blue sedan"
     catalogued on page 1 stays blue forever, even after the story has the
     mechanic spray it red. This module lets an object's appearance EVOLVE while
     still preventing unintentional drift, exactly the way the character tracker
     does for clothing.

  2. PROGRESSIVE TRANSITIONS (multi-stage change).
     Some changes are not instantaneous. Painting a blue car red is a sequence:
     blue → first red patch over blue → half red with blue showing through →
     freshly red with a few blue edges → fully red. A wound heals over days; a
     building is demolished floor by floor; hair greys across a time-skip. This
     module models any such change as an ordered ``ProgressiveTransition`` whose
     intermediate stages are rendered one-per-relevant-panel so the reader sees
     the change *happen* instead of teleporting from start to end state.

  3. CHARACTER AGE.
     Age lives inside the character's fixed portrait, so a few-step diffusion
     model drifts (a sixty-year-old rendered as thirty in some panels) and
     deliberate time-jumps (childhood flashbacks, "twenty years later") are not
     handled at all. This module pins a STRONG, explicit age descriptor into
     every panel and shifts it — coherently — across time-jumps, so old and
     young versions of the same person never get mixed up by accident, but DO
     change when the story intends it.

Bidirectional art-director reasoning
------------------------------------
The character tracker walks strictly forward. This pass instead gives its
per-batch LLM call BOTH the tail of the previous page AND the head of the next
page as context (see ``track_world_continuity`` / ``_detect_world_changes_llm``).
That lets the "continuity art director" (a) reconcile each panel against what
just happened, and (b) *begin a progressive change a beat early* so the end
state lands exactly where the upcoming page needs it (e.g. start the red paint
on this page so the car is fully red by the splash that opens the next page).

Outputs (annotations written onto each panel dict)
--------------------------------------------------
  panel['_world_state']   = {object_label: {appearance, category, progressing,
                                            progress_fraction, ...}}
  panel['_world_changes'] = [{label, transition, stage_descriptor}, ...]
  panel['_age_state']     = {character_name: {age_descriptor, apparent_age,
                                              life_stage, time_context}}

Integration (mirrors track_appearance_continuity)
-------------------------------------------------
Step 1 — in synthesize_comic_book(), right AFTER track_appearance_continuity():
    from comic_book_continuity import track_world_continuity
    script_raw = track_world_continuity(
        script_raw, characters,
        registry=registry, visual_bible=visual_bible,
    )

Step 2 — in compose_panel_prompt():
    from comic_book_continuity import (
        world_continuity_clause, age_continuity_clause,
        filter_visual_bible_clause,
    )
    # Subject section (age is identity):
    age_clause = age_continuity_clause(panel_script, chars_in_frame)
    if age_clause:
        _subject_bits.append(age_clause)
    # Objects section (evolving object state):
    world_clause = world_continuity_clause(panel_script)
    if world_clause:
        _objects_bits.append(world_clause)
    # And drift-guard the static Visual Bible so it can't re-assert a stale
    # colour for an object the world tracker has since evolved:
    vb_clause = filter_visual_bible_clause(vb_clause, panel_script)

Design notes
------------
* Like the character tracker, the LLM is used ONLY for the hard language task
  (detecting introductions / changes / time-jumps). The running state is held
  in plain Python so it is deterministic, inspectable, and resilient: if a batch
  LLM call fails the state simply carries forward unchanged.
* Progressive transitions ALSO auto-advance: once a paint job (etc.) is under
  way, every subsequent same-scene panel nudges it one stage forward even if the
  LLM doesn't re-mention it, so the change actually reaches its end state instead
  of stalling on the first stage.
* All prompt clauses are written as VISUAL DESCRIPTIONS (the words a diffusion
  model can render), not imperative pipeline directives, consistent with the
  rest of the prompt builder.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM / parse / scene-boundary helpers (reuse the project's utilities, exactly
# as comic_book_art_director.py does). Kept as thin wrappers so this module can
# be imported for unit-testing without the full ML stack present.
# ---------------------------------------------------------------------------

def _llm(prompt: str, temperature: float = 0.2, large: bool = True,
         cached_prefix: str = "") -> str:
    from novel_generator import (
        get_openai_prompt_response, openai_model, openai_model_large, USE_GROK,
    )
    model = openai_model_large if large else openai_model
    return get_openai_prompt_response(
        prompt, temperature=temperature, openai_model=model, use_grok=USE_GROK,
        cached_prefix=cached_prefix,
    )


def _parse(text: str):
    from novel_generator import parse_json_response
    return parse_json_response(text)


def _settings_differ(a: str, b: str) -> bool:
    """Scene-boundary check. Reuses the art director's shared detector so
    "new scene" means the same thing across every continuity pass; falls back to
    a cheap token-overlap heuristic if that import is unavailable (e.g. in an
    isolated unit test)."""
    try:
        from comic_book_art_director import _settings_differ as _sd
        return _sd(a, b)
    except Exception:
        pa = set(re.findall(r'[a-z]{3,}', (a or '').lower()))
        pb = set(re.findall(r'[a-z]{3,}', (b or '').lower()))
        if not pa or not pb:
            return False
        overlap = len(pa & pb) / len(pa | pb)
        return overlap < 0.30


def _safe_int(v, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _chars_in_frame(panel: Dict) -> List[str]:
    """Normalise a panel's characters_in_frame to a list of name strings.
    Mirrors the helper in comic_book_art_director.py."""
    out: List[str] = []
    for entry in (panel.get('characters_in_frame', []) or []):
        if isinstance(entry, dict):
            nm = (entry.get('name') or entry.get('character')
                  or entry.get('character_name') or '')
        else:
            nm = str(entry) if entry is not None else ''
        nm = nm.strip()
        if nm:
            out.append(nm)
    return out


# ===========================================================================
# PROGRESSIVE TRANSITION — the general multi-stage change machine
# ===========================================================================

@dataclass
class ProgressiveTransition:
    """An ordered, multi-panel change from one visual state to another.

    The canonical example is repainting: a blue car does not become a red car in
    one panel, it passes through visible intermediate states. ``stages`` holds
    those states as renderable visual descriptions, in order, ending at the
    final/complete state. ``index`` is the stage CURRENTLY visible:

        index = -1  → not started yet (still the from_state)
        index =  0  → first visible stage of the change
        index = len(stages) - 1 → fully complete (== to_state)

    The machine is deliberately simple and deterministic: the LLM decides WHEN a
    transition starts / continues / completes; this class decides WHAT is visible
    at each step. ``auto_step`` lets a started-but-unmentioned transition keep
    creeping forward so it actually reaches completion across a scene.
    """
    label: str                       # subject id, e.g. "rust-red sedan" or "char:Mara:hair"
    from_state: str                  # visual description of the starting state
    to_state: str                    # visual description of the finished state
    stages: List[str] = field(default_factory=list)   # ordered visible states
    index: int = -1                  # current stage (-1 = not begun)
    auto_step: int = 1               # stages advanced per same-scene panel by default
    kind: str = 'transform'          # transform | paint | build | demolish | heal | age | wear
    halted: bool = False             # explicitly paused (no auto-advance while True)

    def __post_init__(self):
        # Guarantee a usable stage list: if none supplied, fall back to a single
        # blended midpoint then the final state so there is at least one visible
        # intermediate frame between start and end.
        self.stages = [s.strip() for s in (self.stages or []) if str(s).strip()]
        if not self.stages:
            self.stages = [
                f"{self.to_state} only partially applied, the original "
                f"{self.from_state} still clearly showing through",
                self.to_state,
            ]

    # -- queries ------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self.index >= 0

    @property
    def complete(self) -> bool:
        return self.index >= len(self.stages) - 1 and self.index >= 0

    def progress_fraction(self) -> float:
        if not self.started:
            return 0.0
        if len(self.stages) <= 1:
            return 1.0
        return min(1.0, (self.index + 1) / len(self.stages))

    def current_descriptor(self) -> str:
        """Visual state to render right now."""
        if not self.started:
            return self.from_state
        idx = max(0, min(self.index, len(self.stages) - 1))
        return self.stages[idx]

    # -- mutation -----------------------------------------------------------
    def begin(self) -> bool:
        """Move from not-started to the first visible stage. Returns True if it
        actually started here (so the caller can render the *act* of change)."""
        if self.index < 0:
            self.index = 0
            return True
        return False

    def advance(self, steps: int = 1) -> bool:
        """Advance by ``steps`` stages (clamped). Returns True if anything moved."""
        if self.complete:
            return False
        before = self.index
        self.index = min(len(self.stages) - 1, max(0, self.index) + max(1, steps))
        return self.index != before

    def finish(self) -> bool:
        """Jump straight to the completed state."""
        if self.complete:
            return False
        self.index = len(self.stages) - 1
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'kind': self.kind,
            'current': self.current_descriptor(),
            'progress_fraction': round(self.progress_fraction(), 3),
            'complete': self.complete,
            'stage_index': self.index,
            'stage_count': len(self.stages),
        }


# ===========================================================================
# TRACKED WORLD ENTITY — a non-character object whose look can evolve
# ===========================================================================

# Categories of world object we track. (Characters are handled by the other
# tracker; these are the things AROUND the characters.)
_WORLD_CATEGORIES = (
    'vehicle', 'structure', 'prop', 'machine', 'furniture', 'landmark',
    'animal', 'environment', 'object',
)


@dataclass
class TrackedEntity:
    """The evolving appearance of one world object across the story.

    ``base_appearance`` is the canonical/establishing look (seeded from the
    Series Visual Bible when available). ``appearance`` is the CURRENT look,
    which diverges as the story changes the object and then persists forward.
    A ``ProgressiveTransition`` models any gradual change in flight.
    """
    label: str
    base_appearance: str = ""
    appearance: str = ""
    category: str = "object"
    aliases: List[str] = field(default_factory=list)
    transition: Optional[ProgressiveTransition] = None
    first_seen_page: int = 0
    destroyed: bool = False           # once true, render as wreckage / absent

    def __post_init__(self):
        # Auto-derive a head-noun alias so a label like "rust-red sedan" still
        # matches a later mention of just "the sedan". The head noun is the last
        # alphabetic token of the label, provided it is distinctive enough.
        head = _head_noun(self.label)
        if head and head not in self.aliases and head.lower() != self.label.lower():
            self.aliases.append(head)

    def matches(self, text: str) -> bool:
        low = (text or '').lower()
        for term in [self.label] + list(self.aliases):
            term = (term or '').strip().lower()
            if term and re.search(r'\b' + re.escape(term) + r'\b', low):
                return True
        return False

    def current_appearance(self) -> str:
        if self.destroyed:
            return f"{self.label}: destroyed / in ruins, no longer intact"
        if self.transition is not None and self.transition.started:
            return self.transition.current_descriptor()
        return self.appearance or self.base_appearance

    def snapshot(self) -> Dict[str, Any]:
        snap = {
            'label': self.label,
            'category': self.category,
            'appearance': self.current_appearance(),
            'destroyed': self.destroyed,
            'progressing': bool(self.transition and self.transition.started
                                and not self.transition.complete),
        }
        if self.transition is not None:
            snap['transition'] = self.transition.snapshot()
        return snap


# ===========================================================================
# CHARACTER AGE STATE — strong age anchor + time-jump handling
# ===========================================================================

# Life-stage bands → a renderable visual descriptor skeleton. Gender-aware
# noun is substituted in. These exist so age tracking works fully WITHOUT an
# LLM call; the LLM only ever supplies the apparent age NUMBER (or a delta).
_LIFE_STAGES: List[Tuple[int, int, str, str]] = [
    # (min_age, max_age, stage_label, visual_descriptor_template{noun})
    (0,   2,  'infant',       "an infant {noun}, soft round features, tiny build"),
    (3,   12, 'child',        "a young child, small stature, smooth unlined face, child's proportions"),
    (13,  17, 'teenager',     "a teenager, adolescent features, slim youthful build"),
    (18,  29, 'young adult',  "a young adult {noun} in their twenties, smooth youthful skin, taut features"),
    (30,  44, 'adult',        "an adult {noun} in their thirties to early forties, mature but unlined features"),
    (45,  59, 'middle-aged',  "a middle-aged {noun}, faint lines around the eyes and mouth, settling features"),
    (60,  74, 'older adult',  "an older {noun} in their sixties, lined face, greying hair, aged hands"),
    (75, 200, 'elderly',      "an elderly {noun}, deeply wrinkled face, white or thinning hair, frail aged build"),
]


def _gender_noun(gender: str) -> str:
    g = (gender or '').strip().lower()
    if g in ('woman', 'female', 'girl', 'lady'):
        return 'woman'
    if g in ('man', 'male', 'boy', 'guy'):
        return 'man'
    return 'person'


def age_to_descriptor(age: Optional[int], gender: str = 'person') -> Tuple[str, str]:
    """Map a numeric age to (life_stage_label, visual_descriptor).

    Deterministic and LLM-free. ``age`` None → ('', '') so callers can skip the
    age clause entirely rather than guess.
    """
    if age is None:
        return '', ''
    noun = _gender_noun(gender)
    for lo, hi, label, tmpl in _LIFE_STAGES:
        if lo <= age <= hi:
            desc = tmpl.format(noun=noun)
            # Append the explicit number for the strongest possible anchor —
            # diffusion models respond to "in their sixties" AND to "62-year-old".
            if label not in ('child', 'teenager', 'infant'):
                desc = f"{desc} (approximately {age} years old)"
            else:
                desc = f"{desc} (approximately {age} years old)"
            return label, desc
    return '', ''


@dataclass
class TimeContext:
    """The temporal framing of a scene, which sets every character's apparent age.

    kind:
      'present'      → characters at their canonical/base age (the default, sticky)
      'flashback'    → earlier in the timeline (younger). transient: reverts to
                       present at the next scene unless re-stated.
      'flashforward' → later in the timeline (older). transient like flashback.
    offset_years: signed years from the base age (negative = younger).
    absolute_ages: explicit {name: apparent_age} overrides (win over offset).
    label: human-readable note for logging / annotations ("childhood flashback").
    """
    kind: str = 'present'
    offset_years: int = 0
    absolute_ages: Dict[str, int] = field(default_factory=dict)
    label: str = ''

    @property
    def is_present(self) -> bool:
        return self.kind == 'present' and self.offset_years == 0 and not self.absolute_ages

    @property
    def transient(self) -> bool:
        # Present is sticky; explicit jumps last only for their own scene.
        return self.kind in ('flashback', 'flashforward')


@dataclass
class CharacterAgeState:
    name: str = ''
    gender: str = 'person'
    base_age: Optional[int] = None     # canonical present-day age

    def apparent_age(self, tctx: TimeContext) -> Optional[int]:
        if self.name in tctx.absolute_ages:
            return tctx.absolute_ages[self.name]
        if self.base_age is None:
            return None
        return max(0, self.base_age + tctx.offset_years)

    def descriptor(self, tctx: TimeContext) -> Tuple[str, Optional[int], str]:
        """Return (visual_descriptor, apparent_age, life_stage_label)."""
        a = self.apparent_age(tctx)
        label, desc = age_to_descriptor(a, self.gender)
        return desc, a, label


# ===========================================================================
# WORLD CONTINUITY TRACKER
# ===========================================================================

class WorldContinuityTracker:
    """Walks the script in narrative order, evolving world-object appearance and
    character apparent-age, and annotating every panel with the resolved state.

    Parallel in spirit to AppearanceContinuityTracker, but for the world around
    the characters rather than the characters themselves.
    """

    def __init__(self, registry=None, visual_bible=None,
                 characters: Optional[list] = None):
        self.registry = registry
        self.visual_bible = visual_bible
        self.entities: Dict[str, TrackedEntity] = {}   # label.lower() → entity
        self.ages: Dict[str, CharacterAgeState] = {}   # name → age state
        self._current_setting: str = ''
        self._scene_time: TimeContext = TimeContext()  # default present
        self._seed_from_characters(characters or [])
        self._seed_from_visual_bible(visual_bible)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def _seed_from_characters(self, characters: list) -> None:
        for c in characters:
            name = getattr(c, 'name', None) or (c.get('name') if isinstance(c, dict) else None)
            if not name:
                continue
            name = str(name).strip()
            raw_age = getattr(c, 'age', None)
            if raw_age is None and isinstance(c, dict):
                raw_age = c.get('age')
            age_int = _parse_age_value(raw_age)
            gender = getattr(c, 'gender', '') or (c.get('gender') if isinstance(c, dict) else '')
            if not gender and self.registry is not None and hasattr(self.registry, 'get_gender'):
                gender = self.registry.get_gender(name) or ''
            self.ages[name] = CharacterAgeState(
                name=name, gender=str(gender or 'person'), base_age=age_int,
            )

    def _seed_from_visual_bible(self, visual_bible) -> None:
        """Pre-register object-like Visual Bible entries as tracked entities so
        their canonical look is the evolving baseline (and the static VB clause
        can be drift-guarded once they change)."""
        if visual_bible is None or not hasattr(visual_bible, 'entries'):
            return
        for e in getattr(visual_bible, 'entries', []) or []:
            category = str(getattr(e, 'category', 'prop') or 'prop').lower()
            # Costumes belong to the character tracker; environments are handled
            # by the scene anchor. Track the physical objects: props/equipment/
            # vehicles/machines/structures.
            if category in ('costume',):
                continue
            label = str(getattr(e, 'label', '') or '').strip()
            canonical = str(getattr(e, 'canonical', '') or '').strip()
            if not label or not canonical:
                continue
            key = label.lower()
            if key in self.entities:
                continue
            self.entities[key] = TrackedEntity(
                label=label,
                base_appearance=canonical,
                appearance=canonical,
                category=_normalise_category(category),
                aliases=[str(a).lower() for a in (getattr(e, 'aliases', []) or [])],
            )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process_script(self, script: List[Dict], batch_size: int = 6) -> List[Dict]:
        ordered: List[Tuple[Dict, Dict, int, int, int]] = []  # (panel,page,page_num,panel_num,page_idx)
        for pi, page in enumerate(script):
            if not isinstance(page, dict) or page.get('_act_break'):
                continue
            page_num = page.get('page', pi + 1)
            panels = page.get('panels', []) or []
            indexed = list(enumerate(panels))
            indexed.sort(key=lambda t: (_safe_int(t[1].get('panel_index', t[0] + 1)) or (t[0] + 1)))
            for orig_pos, panel in indexed:
                panel_num = _safe_int(panel.get('panel_index', orig_pos + 1)) or (orig_pos + 1)
                ordered.append((panel, page, page_num, panel_num, pi))

        if not ordered:
            return script

        logger.info(
            f"[WorldContinuity] Tracking object + age continuity across "
            f"{len(ordered)} panels..."
        )

        total = 0
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            total += self._process_batch(batch, script)

        logger.info(
            f"[WorldContinuity] Done. {total} world/age change(s) detected and "
            f"propagated. {len(self.entities)} tracked object(s)."
        )
        return script

    # ------------------------------------------------------------------
    def _process_batch(self, batch, script) -> int:
        # The first panel of the batch carries the page index used to gather
        # bidirectional (prev page + next page) context.
        page_idx = batch[0][4]
        detections = self._detect_world_changes_llm(batch, script, page_idx)

        applied = 0
        for panel, page, page_num, panel_num, _pi in batch:
            pid = f"p{page_num}_panel{panel_num}"
            entry = detections.get(pid, {})

            # ---- scene change bookkeeping (age time-context reverts) ----
            panel_setting = (panel.get('setting') or '').strip()
            scene_changed = bool(panel_setting) and _settings_differ(
                panel_setting, self._current_setting
            )
            if scene_changed:
                # A transient time-jump (flashback) only lasts for its own scene;
                # on a new scene we revert to present unless this batch re-states
                # a jump for the new scene (handled below).
                if self._scene_time.transient:
                    logger.info(
                        f"[WorldContinuity] {pid}: scene change → time context "
                        f"reverts to present (was '{self._scene_time.label}')."
                    )
                    self._scene_time = TimeContext()
                self._current_setting = panel_setting
            elif panel_setting and not self._current_setting:
                self._current_setting = panel_setting

            # ---- apply time-context (age) signal for this panel ----
            tctx_signal = entry.get('time_context')
            if isinstance(tctx_signal, dict):
                self._apply_time_context(tctx_signal, pid)

            # ---- apply object changes ----
            for ch in entry.get('object_changes', []) or []:
                if self._apply_object_change(ch, page_num, pid):
                    applied += 1

            # ---- auto-advance any in-flight transitions for objects present ----
            present_labels = self._objects_present(panel)
            explicitly_changed = {
                str(ch.get('label', '')).strip().lower()
                for ch in (entry.get('object_changes', []) or [])
            }
            for key in present_labels:
                ent = self.entities.get(key)
                if not ent or ent.transition is None:
                    continue
                t = ent.transition
                if t.started and not t.complete and not t.halted \
                        and key not in explicitly_changed:
                    if t.advance(t.auto_step):
                        logger.info(
                            f"[WorldContinuity] {pid}: '{ent.label}' transition "
                            f"auto-advanced → {t.progress_fraction():.0%} "
                            f"({t.current_descriptor()[:50]})"
                        )

            # ---- snapshot world state for objects depicted in this panel ----
            world_snapshot: Dict[str, Dict[str, Any]] = {}
            world_changes: List[Dict[str, str]] = []
            for key in present_labels:
                ent = self.entities.get(key)
                if ent:
                    world_snapshot[ent.label] = ent.snapshot()
            for ch in entry.get('object_changes', []) or []:
                lbl = str(ch.get('label', '')).strip()
                tr = str(ch.get('transition', '')).strip()
                if lbl and tr:
                    ent = self.entities.get(lbl.lower())
                    world_changes.append({
                        'label': lbl,
                        'transition': tr,
                        'stage_descriptor': ent.current_appearance() if ent else '',
                    })

            panel['_world_state'] = world_snapshot
            panel['_world_changes'] = world_changes

            # ---- snapshot age state for characters in this panel ----
            age_snapshot: Dict[str, Dict[str, Any]] = {}
            for name in _chars_in_frame(panel):
                st = self.ages.get(name) or self._match_age_state(name)
                if not st:
                    continue
                desc, apparent, stage = st.descriptor(self._scene_time)
                if not desc:
                    continue
                age_snapshot[name] = {
                    'age_descriptor': desc,
                    'apparent_age': apparent,
                    'life_stage': stage,
                    'time_context': self._scene_time.kind,
                    'time_label': self._scene_time.label,
                }
            panel['_age_state'] = age_snapshot

        return applied

    # ------------------------------------------------------------------
    def _objects_present(self, panel: Dict) -> List[str]:
        """Keys of tracked entities whose label/alias appears in this panel."""
        text = ' '.join([
            panel.get('description', '') or '',
            panel.get('setting', '') or '',
            panel.get('sensory_anchor', '') or '',
        ])
        return [key for key, ent in self.entities.items() if ent.matches(text)]

    def _match_age_state(self, name: str) -> Optional[CharacterAgeState]:
        """Resolve an age state by first-name match when the exact key misses."""
        nl = name.lower()
        for k, st in self.ages.items():
            if k.lower() == nl:
                return st
        first = nl.split()[0] if nl.split() else nl
        cands = [st for k, st in self.ages.items()
                 if (k.lower().split()[0] if k.lower().split() else k.lower()) == first]
        return cands[0] if len(cands) == 1 else None

    # ------------------------------------------------------------------
    def _apply_time_context(self, sig: Dict, pid: str) -> None:
        kind = str(sig.get('kind', 'present')).strip().lower()
        if kind not in ('present', 'flashback', 'flashforward'):
            kind = 'present'
        if kind == 'present':
            if not self._scene_time.is_present:
                logger.info(f"[WorldContinuity] {pid}: time context → present.")
            self._scene_time = TimeContext()
            return
        offset = _safe_int(sig.get('offset_years'), 0) or 0
        if kind == 'flashback' and offset > 0:
            offset = -offset   # flashbacks are younger
        if kind == 'flashforward' and offset < 0:
            offset = -offset
        absolute: Dict[str, int] = {}
        for k, v in (sig.get('absolute_ages') or {}).items():
            iv = _safe_int(v)
            if iv is not None:
                absolute[str(k).strip()] = iv
        label = str(sig.get('label', '') or kind).strip()
        self._scene_time = TimeContext(
            kind=kind, offset_years=offset, absolute_ages=absolute, label=label,
        )
        logger.info(
            f"[WorldContinuity] {pid}: time context → {kind} "
            f"(offset {offset:+d}y{', absolutes ' + str(absolute) if absolute else ''}) "
            f"'{label}'."
        )

    # ------------------------------------------------------------------
    def _apply_object_change(self, ch: Dict, page_num: int, pid: str) -> bool:
        label = str(ch.get('label', '') or '').strip()
        if not label:
            return False
        key = label.lower()
        ctype = str(ch.get('change_type', '') or '').strip().lower()
        new_appearance = str(ch.get('new_appearance', '') or '').strip()

        ent = self.entities.get(key)
        if ent is None:
            # New object introduced by the story (not in the Visual Bible).
            ent = TrackedEntity(
                label=label,
                base_appearance=new_appearance or label,
                appearance=new_appearance or label,
                category=_normalise_category(ch.get('category', 'object')),
                aliases=[a.lower() for a in (ch.get('aliases') or []) if a],
                first_seen_page=page_num,
            )
            self.entities[key] = ent
            logger.info(f"[WorldContinuity] {pid}: registered new object '{label}'.")
            if ctype in ('introduce', 'appear', ''):
                return True

        if ctype == 'destroy':
            ent.destroyed = True
            return True
        if ctype in ('progress_start', 'progress', 'transform_start', 'paint',
                     'build', 'demolish', 'wear', 'heal', 'age'):
            return self._start_or_advance_transition(ent, ch, pid)
        if ctype in ('progress_complete', 'complete', 'finish'):
            if ent.transition is not None:
                changed = ent.transition.finish()
                ent.appearance = ent.transition.to_state or ent.appearance
                return changed
            if new_appearance:
                ent.appearance = new_appearance
                return True
            return False
        if ctype == 'halt':
            if ent.transition is not None:
                ent.transition.halted = True
                return True
            return False
        # Plain instantaneous restate of appearance.
        if new_appearance and new_appearance != ent.appearance:
            ent.appearance = new_appearance
            ent.transition = None   # an instantaneous change clears any in-flight one
            return True
        return False

    def _start_or_advance_transition(self, ent: TrackedEntity, ch: Dict, pid: str) -> bool:
        """Create (or continue) a progressive transition on an object."""
        to_state = str(ch.get('to_state') or ch.get('new_appearance') or '').strip()
        from_state = str(ch.get('from_state') or ent.current_appearance()
                         or ent.base_appearance).strip()
        stages = [str(s).strip() for s in (ch.get('stages') or []) if str(s).strip()]
        kind = str(ch.get('change_type', 'transform')).strip().lower()
        if kind.endswith('_start'):
            kind = kind[:-6]

        if ent.transition is None or ent.transition.to_state != to_state:
            # Begin a fresh transition. If the LLM gave us no stages, synthesise
            # a sensible default so the change is gradual rather than a jump.
            if not stages and to_state:
                stages = _default_stages(from_state, to_state, kind)
            ent.transition = ProgressiveTransition(
                label=ent.label, from_state=from_state, to_state=to_state or from_state,
                stages=stages, kind=kind or 'transform',
                auto_step=_safe_int(ch.get('auto_step'), 1) or 1,
            )
            ent.transition.begin()
            logger.info(
                f"[WorldContinuity] {pid}: '{ent.label}' begins {kind or 'transform'} "
                f"({from_state[:30]} → {to_state[:30]}, {len(ent.transition.stages)} stages)."
            )
            return True
        # Same transition continuing → advance one explicit step.
        t = ent.transition
        if t.halted:
            t.halted = False
        moved = t.advance(t.auto_step)
        if t.complete:
            ent.appearance = t.to_state or ent.appearance
        return moved

    # ------------------------------------------------------------------
    # LLM change detection (bidirectional: prev page + next page context)
    # ------------------------------------------------------------------
    def _detect_world_changes_llm(self, batch, script, page_idx) -> Dict[str, Dict]:
        """The continuity art-director call. Returns {panel_id: {...}}.

        Given the CURRENT tracked world state, the panels in this batch, AND the
        tail of the previous page + head of the next page (same-scene only), it
        reports per panel: object introductions/changes (incl. multi-stage
        progressions) and the scene's temporal framing (for age).
        """
        # --- current state block ---
        ent_lines: List[str] = []
        for ent in self.entities.values():
            t = ''
            if ent.transition and ent.transition.started and not ent.transition.complete:
                t = (f"  [IN PROGRESS: {ent.transition.kind} "
                     f"{ent.transition.progress_fraction():.0%} → goal: "
                     f"{ent.transition.to_state[:60]}]")
            ent_lines.append(
                f"  - {ent.label} ({ent.category}): {ent.current_appearance()[:80]}{t}"
            )
        ent_block = "\n".join(ent_lines) if ent_lines else "  (no objects catalogued yet)"

        age_lines = []
        for st in self.ages.values():
            if st.base_age is not None:
                age_lines.append(f"  - {st.name}: canonical age {st.base_age} ({_gender_noun(st.gender)})")
        age_block = "\n".join(age_lines) if age_lines else "  (no ages on record)"

        cur_time = (f"{self._scene_time.kind}"
                    + (f", offset {self._scene_time.offset_years:+d}y" if self._scene_time.offset_years else "")
                    + (f" [{self._scene_time.label}]" if self._scene_time.label else ""))

        # --- panels block ---
        panel_items = []
        for panel, page, page_num, panel_num, _pi in batch:
            desc = (panel.get('description') or '').strip()
            setting = (panel.get('setting') or '').strip()
            panel_items.append(
                f'{{"id": "p{page_num}_panel{panel_num}", '
                f'"setting": {_json_str(setting)}, '
                f'"description": {_json_str(desc)}}}'
            )
        panels_block = "\n".join(panel_items)

        # --- bidirectional context (prev page tail + next page head) ---
        prev_ctx, next_ctx = self._adjacent_context(script, page_idx)
        prev_block = "; ".join(prev_ctx) if prev_ctx else "(scene starts here / nothing carried in)"
        next_block = "; ".join(next_ctx) if next_ctx else "(scene ends here / nothing follows)"

        # Split for prompt caching: the rule/schema scaffold is byte-identical on
        # every batch, so it is passed as cached_prefix (served from Grok's cache
        # after the first batch). Only the per-batch data (tracked objects, ages,
        # scene time, adjacent context, panels) is new — it trails in `prompt`.
        _cached_prefix = f"""You are the Continuity Art Director for a graphic novel.
Track the WORLD (objects, vehicles, structures, machines, set-pieces) and the
APPARENT AGE of characters. You do NOT track clothing or wounds — another pass owns those.

GOVERNING PRINCIPLE — STATE IS STICKY AND CHANGES ARE GRADUAL:
- Once an object looks a certain way it KEEPS that look until the story changes it.
- A change that would realistically take time (repainting, building, demolishing,
  rusting, healing, ageing) must happen as an ORDERED SEQUENCE of visible stages,
  NOT an instant jump. Example: painting a blue car red passes through
  "blue car with the first red patch", "half red, blue showing through",
  "freshly red with a few blue edges", "fully red".
- Use the PREVIOUS-PAGE context to stay consistent with what just happened, and
  the NEXT-PAGE context to START a gradual change early enough that it reaches
  its finished state exactly when the next page needs it.

CONSUMABLE / QUANTITY STATE (a specific case of gradual change):
- Objects that are consumed or depleted within a scene must show their level
  DECREASING monotonically panel to panel — never refilling unless the story
  explicitly shows a refill. Track these as ordinary object_changes with ordered
  stages, using change_type "introduce" on first appearance (with the starting
  level) then "progress"/"progress_complete" as they deplete.
- Examples: a wine or water glass emptying as characters drink; a burning candle
  or cigarette getting shorter; a plate of food being eaten; a stack of papers
  shrinking as they are read; a fuel gauge dropping; a bottle draining.
- Set the object's appearance to include the CURRENT level explicitly, e.g.
  "wine glass, two-thirds full" → "wine glass, half full" → "wine glass, nearly
  empty with a red residue". This prevents a full glass reappearing after a sip.
- A consumable does NOT reset on a scene change unless the new scene plausibly
  provides a fresh one (a new round of drinks at a new location).

=== WHAT TO REPORT PER PANEL ===
Return ONLY a JSON array (no markdown, no preamble):
[
  {{
    "id": "p<page>_panel<n>",
    "time_context": {{
        "kind": "present | flashback | flashforward",
        "offset_years": <signed int, e.g. -25 for a childhood flashback, +20 for later>,
        "absolute_ages": {{"CharacterName": <int>}},   // optional explicit ages
        "label": "short note e.g. 'childhood flashback'"
    }},
    "object_changes": [
      {{
        "label": "rust-red sedan",          // stable name; reuse the catalogued label if it exists
        "category": "vehicle",              // vehicle|structure|prop|machine|furniture|landmark|animal|object|consumable
        "change_type": "introduce | progress_start | progress | progress_complete | destroy | restate | halt",
        "from_state": "glossy blue four-door sedan",     // for progress_start
        "to_state": "glossy red four-door sedan",        // the finished look
        "stages": [                          // ordered visible intermediate states (optional but preferred)
           "blue sedan with a single panel primed in red",
           "sedan half red, blue still showing along the doors and roof",
           "freshly red sedan, a few blue chips at the seams",
           "uniformly glossy red sedan"
        ],
        "new_appearance": "...",            // for introduce / restate / progress_complete
        "transition": "what is visibly happening this panel, e.g. 'spraying the first coat'"
      }}
    ]
  }}
]

RULES:
- Only emit "time_context" when the panel's temporal framing is NOT plain present
  day, OR when a scene RETURNS to the present after a jump (then kind="present").
- Only emit "object_changes" when this panel actually introduces or changes an object.
- For an ongoing change already IN PROGRESS (see state below), emit change_type
  "progress" on the panels where work visibly continues, and "progress_complete"
  on the panel where it is finished. Do NOT restate an unchanged object.
- Reuse the exact catalogued label for an object that already exists.
- Omit panels that have neither a time_context nor object_changes.

The batch to process (tracked state, scene context, and panels) follows below."""

        prompt = f"""CURRENT TRACKED OBJECTS (carry forward unless changed):
{ent_block}

CHARACTER CANONICAL AGES:
{age_block}

CURRENT SCENE TIME CONTEXT: {cur_time}

PREVIOUS PAGE ended with: {prev_block}
NEXT PAGE will show: {next_block}

PANELS IN THIS BATCH (narrative order, state flows top to bottom):
{panels_block}

Now return the JSON array for the panels above, following the rules and schema
exactly. JSON only."""

        try:
            raw = _llm(prompt, temperature=0.15, cached_prefix=_cached_prefix)
            parsed = _parse(raw)
            if not isinstance(parsed, list):
                logger.warning("[WorldContinuity] LLM returned non-list; carrying state forward.")
                return {}
            out: Dict[str, Dict] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                pid = item.get('id', '')
                if not pid:
                    continue
                rec: Dict[str, Any] = {}
                if isinstance(item.get('time_context'), dict):
                    rec['time_context'] = item['time_context']
                oc = item.get('object_changes')
                if isinstance(oc, list):
                    rec['object_changes'] = [c for c in oc if isinstance(c, dict) and c.get('label')]
                if rec:
                    out[pid] = rec
            return out
        except Exception as e:
            logger.warning(f"[WorldContinuity] Change-detection batch failed: {e}; carrying state forward.")
            return {}

    # ------------------------------------------------------------------
    def _adjacent_context(self, script, page_idx) -> Tuple[List[str], List[str]]:
        """Same-scene tail of the previous page + head of the next page.

        Reuses the generator's _gather_adjacent_page_context when available so
        the same-scene gating is identical to the rest of the pipeline; falls
        back to a local implementation for isolated testing.
        """
        try:
            from comic_book_generator import _gather_adjacent_page_context
            return _gather_adjacent_page_context(script, page_idx)
        except Exception:
            pass
        # Local fallback (no same-scene gating beyond a setting compare).
        def _real(idx):
            if 0 <= idx < len(script):
                p = script[idx]
                if isinstance(p, dict) and not p.get('_act_break'):
                    return p
            return None
        prev_s: List[str] = []
        next_s: List[str] = []
        cur = _real(page_idx)
        cur_panels = (cur or {}).get('panels', []) or []
        if cur_panels:
            pp = _real(page_idx - 1)
            if pp and (pp.get('panels') or []):
                if not _settings_differ(
                        (cur_panels[0].get('setting') or ''),
                        (pp['panels'][-1].get('setting') or '')):
                    prev_s = [(_p.get('description', '') or '')[:120]
                              for _p in pp['panels'][-2:] if _p.get('description')]
            nx = _real(page_idx + 1)
            if nx and (nx.get('panels') or []):
                if not _settings_differ(
                        (cur_panels[-1].get('setting') or ''),
                        (nx['panels'][0].get('setting') or '')):
                    next_s = [(_p.get('description', '') or '')[:120]
                              for _p in nx['panels'][:2] if _p.get('description')]
        return prev_s, next_s


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _json_str(s: str) -> str:
    """JSON-encode a string for safe embedding in an LLM prompt literal."""
    import json
    return json.dumps(s or "")


# Head-noun matching stoplist: colours / generic adjectives that should never be
# treated as a distinctive object head, plus words too generic to anchor on.
_HEAD_NOUN_STOP = frozenset({
    'the', 'a', 'an', 'old', 'new', 'big', 'small', 'red', 'blue', 'green',
    'black', 'white', 'grey', 'gray', 'yellow', 'thing', 'object', 'item',
    'one', 'his', 'her', 'their', 'its', 'left', 'right',
})


def _head_noun(label: str) -> str:
    """Last distinctive alphabetic token of a label ('rust-red sedan' -> 'sedan').
    Returns '' if no token is distinctive enough to safely match on alone."""
    toks = re.findall(r"[a-zA-Z]{3,}", label or '')
    for tok in reversed(toks):
        if tok.lower() not in _HEAD_NOUN_STOP:
            return tok
    return ''


def _normalise_category(cat: str) -> str:
    c = (cat or 'object').strip().lower()
    aliases = {
        'equipment': 'machine', 'tool': 'prop', 'weapon': 'prop',
        'building': 'structure', 'car': 'vehicle', 'ship': 'vehicle',
        'set': 'environment', 'setting': 'environment', 'creature': 'animal',
    }
    c = aliases.get(c, c)
    return c if c in _WORLD_CATEGORIES else 'object'


def _parse_age_value(raw) -> Optional[int]:
    """Best-effort parse of a Character.age value (int, '34', 'mid-thirties',
    'late 60s', 'Unknown') into a representative integer, or None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if raw > 0 else None
    s = str(raw).strip().lower()
    if not s or s in ('unknown', 'n/a', 'none', '-'):
        return None
    m = re.search(r'\d{1,3}', s)
    if m:
        n = int(m.group())
        # "60s" / "sixties" style → nudge to mid-decade.
        if re.search(r'\b' + re.escape(m.group()) + r'\s*s\b', s) or 'late' in s:
            n += 5
        elif 'early' in s:
            n += 2
        elif 'mid' in s:
            n += 5
        return n if 0 < n < 130 else None
    words = {
        'infant': 1, 'baby': 1, 'toddler': 3, 'child': 8, 'kid': 9,
        'teen': 15, 'teenager': 15, 'adolescent': 15, 'young': 22,
        'twenties': 25, 'thirties': 35, 'forties': 45, 'fifties': 55,
        'sixties': 65, 'seventies': 75, 'eighties': 85, 'elderly': 78,
        'old': 70, 'middle-aged': 50,
    }
    for w, v in words.items():
        if re.search(r'\b' + re.escape(w) + r'\b', s):
            return v
    return None


def _default_stages(from_state: str, to_state: str, kind: str) -> List[str]:
    """Synthesise a small ordered stage list when the LLM supplied none, so any
    progressive change is gradual. Kept generic across change kinds."""
    fs = (from_state or 'the original state').strip().rstrip('.')
    ts = (to_state or 'the new state').strip().rstrip('.')
    k = (kind or 'transform').lower()
    if k in ('paint', 'transform'):
        return [
            f"{ts}, only just begun — the first portion done while most of {fs} still shows",
            f"about half {ts}, with {fs} still clearly visible underneath in patches",
            f"mostly {ts}, only a few traces of {fs} left at the edges and seams",
            ts,
        ]
    if k == 'build':
        return [
            f"{ts} under construction, framework only, far from finished",
            f"{ts} roughly half-built, major structure up but incomplete",
            f"{ts} nearly complete, final details still missing",
            ts,
        ]
    if k in ('demolish', 'destroy'):
        return [
            f"{fs} beginning to come apart, first damage visible",
            f"{fs} substantially wrecked, large sections gone",
            f"{fs} reduced mostly to rubble",
            f"{ts}",
        ]
    if k in ('rust', 'wear', 'decay', 'age'):
        return [
            f"{fs} just starting to show wear toward {ts}",
            f"{fs} clearly worn, halfway to {ts}",
            f"almost fully {ts}, only faint signs of {fs} remaining",
            ts,
        ]
    if k == 'heal':
        return [
            f"{fs} freshly treated, beginning to mend",
            f"{fs} healing, noticeably improved",
            f"nearly healed toward {ts}",
            ts,
        ]
    # generic
    return [
        f"midway between {fs} and {ts}, both clearly visible",
        ts,
    ]


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

def track_world_continuity(
    script: List[Dict],
    characters: list,
    registry=None,
    visual_bible=None,
) -> List[Dict]:
    """Annotate every panel with evolved world-object state and character ages.

    Run AFTER track_appearance_continuity (so it sees disambiguated descriptions)
    and BEFORE image generation. Mutates the script in place and returns it.

    Each story panel gains:
      panel['_world_state']   - {object_label: snapshot} for objects in frame
      panel['_world_changes'] - object changes happening in THIS panel
      panel['_age_state']     - {char_name: {age_descriptor, apparent_age, ...}}
    """
    try:
        tracker = WorldContinuityTracker(
            registry=registry, visual_bible=visual_bible, characters=characters or [],
        )
        return tracker.process_script(script)
    except Exception as e:
        logger.warning(f"[WorldContinuity] Tracking failed ({e}); panels left un-annotated.")
        return script


# ===========================================================================
# PROMPT CLAUSES (read the annotations written above)
# ===========================================================================

def world_continuity_clause(panel_script: Dict) -> str:
    """Objects-section clause: the CURRENT look of every tracked object in frame,
    plus the visible action when a progressive change is happening this panel.

    Returns "" when the panel has no tracked world state.
    """
    state = panel_script.get('_world_state') or {}
    changes = panel_script.get('_world_changes') or []
    if not state and not changes:
        return ''

    changing = {str(c.get('label', '')).strip().lower(): c for c in changes}

    descriptors: List[str] = []
    transitions: List[str] = []
    for label, snap in state.items():
        appearance = (snap.get('appearance') or '').strip()
        if not appearance:
            continue
        if snap.get('destroyed'):
            descriptors.append(appearance)
            continue
        descriptors.append(appearance)
        # If a multi-stage change is under way, name the visible action too.
        ch = changing.get(label.lower())
        if ch and ch.get('transition'):
            transitions.append(f"{label} — {ch['transition']}")
        elif snap.get('progressing'):
            frac = (snap.get('transition') or {}).get('progress_fraction', 0)
            transitions.append(
                f"{label} is mid-change ({frac:.0%} of the way to its new state); "
                f"render it at exactly this partial stage, not finished"
            )

    if not descriptors and not transitions:
        return ''

    parts: List[str] = []
    if descriptors:
        parts.append(
            "Objects in this scene look exactly like this and must stay consistent "
            "with the preceding panels: " + "; ".join(descriptors) + "."
        )
    if transitions:
        parts.append("Change in progress: " + "; ".join(transitions) + ".")
    return " ".join(parts)


def age_continuity_clause(panel_script: Dict, characters_in_frame: List[str]) -> str:
    """Subject-section clause: a strong apparent-age descriptor for each character
    in frame, so the diffusion model neither drifts the age nor mixes a young and
    an old version of the same person across panels.

    Returns "" when no age is on record for anyone in frame.
    """
    age_state = panel_script.get('_age_state') or {}
    if not age_state:
        return ''

    names = characters_in_frame or list(age_state.keys())
    bits: List[str] = []
    flagged_jump = False
    for name in names:
        info = age_state.get(name)
        if not info:
            # tolerate first-name vs full-name mismatch
            nl = name.lower()
            for k, v in age_state.items():
                kl = k.lower()
                if kl == nl or kl.split()[0:1] == nl.split()[0:1]:
                    info = v
                    break
        if not info:
            continue
        desc = (info.get('age_descriptor') or '').strip()
        if not desc:
            continue
        bits.append(desc)
        if info.get('time_context') in ('flashback', 'flashforward'):
            flagged_jump = True

    if not bits:
        return ''

    clause = "Apparent age (hold this exactly, do not render younger or older): " \
             + "; ".join(bits) + "."
    if flagged_jump:
        label = ''
        for v in age_state.values():
            if v.get('time_label'):
                label = v['time_label']
                break
        clause += (f" This panel is a {label or 'time-shifted'} moment — the "
                   f"characters are deliberately shown at this age, not their "
                   f"present-day age.")
    return clause


def filter_visual_bible_clause(vb_clause: str, panel_script: Dict) -> str:
    """Drift-guard for the static Series Visual Bible clause.

    The Visual Bible clause restates an object's ORIGINAL canonical look
    ("[rust-red sedan: glossy blue four-door sedan ...]"). Once the world tracker
    has evolved that object (the car is now red), the static clause would fight
    the evolved state. This removes any "[label: ...]" segment whose label is a
    tracked object that has CHANGED (appearance diverged from base, a transition
    is in flight/complete, or it was destroyed), leaving unaffected entries
    intact. The world_continuity_clause then supplies the current truth.
    """
    if not vb_clause:
        return vb_clause
    state = panel_script.get('_world_state') or {}
    if not state:
        return vb_clause

    evolved_labels = set()
    for label, snap in state.items():
        if snap.get('destroyed') or snap.get('progressing'):
            evolved_labels.add(label.lower())
            continue
        tr = snap.get('transition')
        if tr and tr.get('complete'):
            evolved_labels.add(label.lower())
    if not evolved_labels:
        return vb_clause

    # The clause format is: "<prefix>: [label: canonical]; [label2: canonical]."
    m = re.match(r'^(.*?:\s*)(.*?)(\.?)$', vb_clause, flags=re.DOTALL)
    if not m:
        return vb_clause
    prefix, body, tail = m.group(1), m.group(2), m.group(3)
    segments = re.findall(r'\[([^\]]+)\]', body)
    kept = []
    for seg in segments:
        seg_label = seg.split(':', 1)[0].strip().lower()
        if seg_label in evolved_labels:
            continue
        kept.append(f"[{seg}]")
    if not kept:
        return ''   # every catalogued object referenced here has evolved
    return f"{prefix}{'; '.join(kept)}{tail}"


# ===========================================================================
# STYLE CONTINUITY — keep the rendering MEDIUM consistent across panels
# ---------------------------------------------------------------------------
# The reported failure: a scene drifts from the established illustrated/animated
# look into a photograph (or a 3D render) for no story reason. Root causes:
#   * the positive prompt names the art style only ONCE (in the summary line),
#     so a few-step model loses it as the prompt grows;
#   * the NEGATIVE prompt excludes text and crowds but NOT foreign media
#     (photograph, photoreal, DSLR, 3D render, CGI), so nothing pushes the
#     latent away from those modes when a description happens to mention a
#     screen, a photo, or a "cinematic" beat.
#
# The fix is a per-panel STYLE LOCK that (a) resolves the story's canonical
# medium ONCE from the Story DNA, (b) re-states it strongly and identically in
# every panel, (c) excludes every OTHER medium in the negative prompt, and
# (d) still allows a DELIBERATE shift (a flashback rendered as a faded
# photograph, a nightmare in stark photoreal) when the script asks for it — so
# intentional shifts are honoured but random drift is suppressed.
# ===========================================================================

# Foreign-media vocabulary, grouped. Each group is excluded in the negative
# prompt unless the resolved medium (or a deliberate per-panel shift) belongs to
# that group.
FOREIGN_MEDIA_TERMS: Dict[str, List[str]] = {
    'photographic': [
        "photograph", "photo", "photorealistic", "photo-realistic", "photoreal",
        "realistic photograph", "DSLR photo", "35mm photo", "film still",
        "live-action", "live action", "studio photograph", "stock photo",
        "polaroid", "candid photo", "RAW photo", "hyperrealistic photo",
    ],
    '3d_render': [
        "3D render", "3d render", "CGI", "cgi render", "octane render",
        "unreal engine", "ray-traced render", "blender render", "3d model",
        "video-game render", "rendered in 3d", "zbrush",
    ],
    'claymation': [
        "claymation", "stop motion", "stop-motion", "plasticine model",
        "felt puppet",
    ],
    'pixel': [
        "pixel art", "8-bit sprite", "16-bit sprite", "voxel art",
    ],
}

# Canonical medium families. ``allowed`` lists the FOREIGN_MEDIA_TERMS groups
# that are COMPATIBLE with the family (and so are NOT excluded). ``anchor`` is
# the strong positive medium statement re-stated in every panel.
MEDIUM_FAMILIES: Dict[str, Dict[str, Any]] = {
    'illustrated_2d': {
        'anchor': ("consistently hand-illustrated 2D comic-book art — the same "
                   "drawn medium in every panel: ink linework with painted color"),
        'allowed': set(),                       # exclude ALL photographic/3D/etc.
    },
    'watercolor': {
        'anchor': ("consistently hand-painted watercolour illustration — the same "
                   "painted medium in every panel: soft washes and visible paper grain"),
        'allowed': set(),
    },
    'cel_animation': {
        'anchor': ("consistently cel-animation / anime illustration — the same flat "
                   "cel-shaded drawn medium in every panel: clean outlines, flat color fills"),
        'allowed': set(),
    },
    '3d_animation': {
        'anchor': ("consistently stylised 3D-animation render — the same rendered "
                   "medium in every panel: soft global illumination, sculpted forms"),
        'allowed': {'3d_render'},               # 3D is the intended look here
    },
}

# Keyword → family resolution, longest/most-specific first.
_MEDIUM_KEYWORDS: List[Tuple[str, str]] = [
    ('watercolour', 'watercolor'), ('watercolor', 'watercolor'),
    ('gouache', 'watercolor'),
    ('cel-shad', 'cel_animation'), ('cel shad', 'cel_animation'),
    ('anime', 'cel_animation'), ('manga', 'cel_animation'),
    ('cartoon', 'cel_animation'), ('cel animation', 'cel_animation'),
    ('animation', 'cel_animation'),
    ('pixar', '3d_animation'), ('3d animation', '3d_animation'),
    ('3d render', '3d_animation'), ('cgi', '3d_animation'),
]


def resolve_style_medium(story_dna: Optional[Dict]) -> Tuple[str, str]:
    """Resolve the story's canonical (family_key, positive_anchor) from Story DNA.

    Defaults to illustrated 2D — the pipeline's intended look — when the DNA is
    missing or names no recognisable medium.
    """
    dna = story_dna or {}
    blob = " ".join(str(dna.get(k, '')) for k in
                    ('art_style', 'injection_clause', 'line_style',
                     'texture_motif', 'lighting_signature')).lower()
    family = 'illustrated_2d'
    for kw, fam in _MEDIUM_KEYWORDS:
        if kw in blob:
            family = fam
            break
    anchor = MEDIUM_FAMILIES.get(family, MEDIUM_FAMILIES['illustrated_2d'])['anchor']
    return family, anchor


# In-description cues that a panel deliberately RENDERS in a different medium
# (not merely shows a photo as a prop). Conservative on purpose: only explicit
# "rendered as / depicted as / in the style of" phrasing flips the medium, so an
# ordinary mention of a photo or a screen never causes drift.
_SHIFT_RENDER_CUES: List[Tuple[str, str]] = [
    (r'\brendered as (?:a |an )?(?:\w+\s+){0,4}?photograph', 'photographic'),
    (r'\bdepicted as (?:a |an )?(?:\w+\s+){0,4}?photograph', 'photographic'),
    (r'\bin the style of (?:a |an )?(?:\w+\s+){0,4}?photograph', 'photographic'),
    (r'\b(?:shifts?|cuts?) to (?:stark )?photoreal', 'photographic'),
    (r'\bphotorealistic (?:insert|flashback|panel|render)', 'photographic'),
    (r'\bas (?:a |an )?(?:\w+\s+){0,3}?(?:3d|cgi) render', '3d_render'),
    (r'\brendered in 3d', '3d_render'),
]

# Cues that a photo / screen / footage is present as an OBJECT in the scene —
# used to add the "any photo shown here is still drawn in our medium" guard so a
# photo prop doesn't flip the whole panel photoreal.
_MEDIA_PROP_CUES = (
    'photograph', 'photo', 'snapshot', 'polaroid', 'screen', 'monitor',
    'television', 'tv', 'footage', 'video', 'film reel', 'security camera',
    'projector', 'photo album', 'picture frame',
)


def _detect_intended_style_shift(panel: Dict) -> Optional[Dict[str, str]]:
    """Return a deliberate shift descriptor {medium_group, descriptor, reason} or
    None. Honours an explicit script field first, then conservative render cues."""
    # 1. Explicit author/script field wins.
    explicit = panel.get('style_shift') or panel.get('render_style')
    if isinstance(explicit, dict) and (explicit.get('medium') or explicit.get('descriptor')):
        med = str(explicit.get('medium') or explicit.get('descriptor')).strip()
        grp = str(explicit.get('group') or _classify_medium_text(med) or 'photographic')
        return {'medium_group': grp, 'descriptor': med,
                'reason': str(explicit.get('reason', 'authored style shift'))}
    if isinstance(explicit, str) and explicit.strip():
        grp = _classify_medium_text(explicit) or 'photographic'
        return {'medium_group': grp, 'descriptor': explicit.strip(),
                'reason': 'authored style shift'}
    # 2. Conservative in-description render cue.
    desc = (panel.get('description') or '').lower()
    for pat, grp in _SHIFT_RENDER_CUES:
        if re.search(pat, desc):
            return {'medium_group': grp,
                    'descriptor': f"deliberately rendered in a {grp.replace('_', ' ')} look",
                    'reason': 'dramatic/emotional style shift cued in the description'}
    return None


def _classify_medium_text(text: str) -> str:
    t = (text or '').lower()
    for grp, terms in FOREIGN_MEDIA_TERMS.items():
        if any(term.lower() in t for term in terms):
            return grp
    return ''


def track_style_continuity(script: List[Dict],
                           story_idea=None,
                           story_dna: Optional[Dict] = None) -> List[Dict]:
    """Annotate every story panel with a ``_style_lock`` so the rendering medium
    stays consistent — or shifts only when the story deliberately asks.

    Run AFTER build_story_dna() (it reads the canonical art style from the DNA)
    and BEFORE script.json is saved so the lock travels into the rebuild flow.

    Each panel gains:
      panel['_style_lock'] = {
        'medium_family', 'positive_anchor', 'exclude' (list[str]),
        'shifted' (bool), 'reason', 'guard'
      }
    """
    try:
        family, anchor = resolve_style_medium(story_dna)
        allowed = MEDIUM_FAMILIES.get(family, MEDIUM_FAMILIES['illustrated_2d'])['allowed']

        def _exclude_for(allowed_groups: set) -> List[str]:
            out: List[str] = []
            for grp, terms in FOREIGN_MEDIA_TERMS.items():
                if grp in allowed_groups:
                    continue
                out.extend(terms)
            return out

        n_shift = 0
        for page in script:
            if not isinstance(page, dict) or page.get('_act_break'):
                continue
            for panel in (page.get('panels', []) or []):
                shift = _detect_intended_style_shift(panel)
                if shift:
                    n_shift += 1
                    grp = shift['medium_group']
                    positive = (f"{shift['descriptor']} — this panel deliberately "
                                f"shifts medium for dramatic effect")
                    # Relax the excluded group so the shift can actually render.
                    relaxed = set(allowed) | {grp}
                    panel['_style_lock'] = {
                        'medium_family': family,
                        'positive_anchor': positive,
                        'exclude': _exclude_for(relaxed),
                        'shifted': True,
                        'reason': shift['reason'],
                        'guard': '',
                    }
                else:
                    # Guard: if a photo/screen is present as a PROP, keep the whole
                    # panel in our medium and draw the prop in that medium too.
                    desc_low = (panel.get('description') or '').lower()
                    guard = ''
                    if any(re.search(r'\b' + re.escape(c) + r'\b', desc_low)
                           for c in _MEDIA_PROP_CUES):
                        guard = ("Any photograph, screen, or footage shown within "
                                 "the scene is itself drawn in this same illustrated "
                                 "medium — the entire panel stays one consistent "
                                 "drawn style, never a real photo or render.")
                    panel['_style_lock'] = {
                        'medium_family': family,
                        'positive_anchor': anchor,
                        'exclude': _exclude_for(allowed),
                        'shifted': False,
                        'reason': '',
                        'guard': guard,
                    }
        logger.info(
            f"[StyleContinuity] Locked medium '{family}' across the book "
            f"({n_shift} deliberate style-shift panel(s) honoured)."
        )
        return script
    except Exception as e:
        logger.warning(f"[StyleContinuity] Style locking failed ({e}); panels left un-locked.")
        return script


def style_positive_anchor(panel_script: Dict) -> str:
    """The strong, consistent medium statement for this panel (or '' if none)."""
    lock = panel_script.get('_style_lock') or {}
    return (lock.get('positive_anchor') or '').strip()


def style_guard_clause(panel_script: Dict) -> str:
    """The in-scene media-prop guard sentence for this panel (or '')."""
    lock = panel_script.get('_style_lock') or {}
    return (lock.get('guard') or '').strip()


def style_negative_terms(panel_script: Dict) -> str:
    """Comma-joined foreign-media terms to add to this panel's negative prompt.

    Empty string when the panel has no style lock (older pickles) so callers can
    append unconditionally.
    """
    lock = panel_script.get('_style_lock') or {}
    excl = lock.get('exclude') or []
    excl = [t for t in excl if str(t).strip()]
    return ", ".join(excl)

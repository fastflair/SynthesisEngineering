"""
ltx23_prompt_guide.py
=====================

Drop-in patch that upgrades story_to_animation's prompt generation to follow
the official LTX-2.3 prompting guides:

    • https://ltx.io/blog/ltx-2-3-prompt-guide
    • https://ltx.io/blog/how-to-improve-ltx-2-3-prompt-adherence
    • https://ltx.io/blog/common-prompt-mistakes-in-ai-video-generation

USAGE — one line in your notebook, right after importing sta:
    import ltx23_prompt_guide
    ltx23_prompt_guide.apply(sta)

Or apply selectively:
    ltx23_prompt_guide.apply(sta, patch_video=True, patch_image=True)

WHAT IT CHANGES
---------------
1. Motion-prompt LLM instruction: raises target from ~30 words to ~80-150 words,
   restructured as chronological flowing paragraph with separated subject/camera
   motion, physical (not emotional) acting cues, and environment/atmosphere.

2. Few-shot motion examples: replaced with LTX-2.3-length examples (~80-120 words)
   demonstrating the recommended structure: main action → motion specifics →
   environment/lighting → camera direction → audio (when relevant).

3. Camera vocabulary (_CAMERA_BY_COMP): swapped generic descriptions for LTX-2.3's
   preferred cinematography terms (dolly in, dolly out, tracking shot, jib up,
   static camera, handheld, etc.).

4. Cinematic cues: richer lighting and camera descriptions using terms the model
   responds to (shallow depth of field, golden hour, chiaroscuro, etc.).

5. Video-prompt fallback: when no LLM motion prompt exists, the deterministic
   builder now produces a 60-100 word paragraph instead of a 15-20 word fragment.

6. Non-human scene prompts: enriched with atmosphere, texture, ambient audio cues.

7. Image-prompt LLM instruction: adds LTX-2.3-informed guidance for still images
   that will seed i2v — stronger lighting/atmosphere vocabulary, spatial layering,
   and texture detail.

8. Default motion_prompt_max_chars: raised from 900 → 1400 chars (~250 tokens),
   matching the LTX-2.3 guide's recommendation that longer, more detailed prompts
   consistently outperform short ones on 2.3.

NO BREAKING CHANGES: all patches preserve function signatures, data structures,
and the existing lip-sync/mouth-safety pipeline. The existing guards
(_sanitize_motion_prompt_for_no_dialogue, _strip_offimage_elements, etc.) still
run after these patches produce their richer output.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Shot, VideoConfig types live in the target module

logger = logging.getLogger(__name__)

# ============================================================================
# 1. REPLACEMENT FEW-SHOT MOTION EXAMPLES
# ============================================================================
# Each ~80-120 words.  Structure follows the LTX-2.3 guide:
#   main action (single sentence) → motion specifics (chronological) →
#   environment/atmosphere → camera direction.
# NO mouth/speech/singing/talking language (lip-sync pipeline constraint).

LTX23_MOTION_PROMPT_FEWSHOTS = """\
- "A man in a dark coat holds an apple in his left hand and gestures broadly \
with his right, leaning forward with intent. His head tilts slightly to one \
side, eyebrows raised, weight shifting onto his front foot. Wind catches his \
coat collar. The camera holds a steady medium shot at chest height, then \
begins a slow dolly in toward his face. Warm afternoon light from the left \
casts a long shadow behind him on the cobblestone. Ambient sound of distant \
traffic and a faint breeze."

- "Heavy rain falls on a rooftop. A bald man rests one hand on another \
person's shoulder, his expression serious, brow deeply furrowed. His jaw is \
set and he leans slightly forward, fingers tightening on the shoulder. The \
other person remains still. Rain streaks across the lens. The camera starts \
as a medium two-shot, then slowly pushes in to a close-up of the bald man's \
face. Low-key lighting from a single overhead source, water glistening on \
both figures. The sound of heavy rain drumming on concrete."

- "A woman in a red jacket walks steadily along a narrow dirt trail through \
dense pine trees. Her arms swing naturally, boots crunching on gravel, gaze \
fixed ahead. She shifts her weight slightly to navigate a gentle curve in the \
path. Dappled golden-hour sunlight filters through the canopy, casting warm \
patches on the trail. Shallow depth of field keeps the background trees soft. \
The camera tracks alongside her from a low angle, maintaining a consistent \
distance. Ambient forest sounds — birdsong, rustling leaves, distant wind."

- "A man in a suit sits on a leather sofa in a dimly lit room, leaning forward \
toward the person across from him. His hands clasp together between his knees, \
then he opens them in a deliberate, measured gesture. His jaw is set, eyes \
focused and unblinking. A desk lamp flickers once, sending a brief warm flash \
across the wall behind him. The camera holds a static medium shot, locked off \
on a tripod with only the faintest breathing movement. Soft warm sidelight \
from the lamp, cool ambient fill from a window. Room tone with a ticking clock."

- "Selfie point-of-view: a man glides through open sky beneath a parachute \
canopy, arms spread wide. His hair whips in the wind, expression elated, \
eyes wide. The harness straps pull taut against his chest as he shifts his \
weight left, the horizon tilting. Cloud wisps stream past at speed. Bright \
midday sunlight from above, lens flare catching the edge of the canopy. \
Handheld camera perspective, slight shake from turbulence. Rushing wind \
dominates the audio."\
"""

# ============================================================================
# 2. REPLACEMENT CAMERA-BY-COMPOSITION MAP
# ============================================================================
# Uses the LTX-2.3-preferred cinematography terms that the model is trained on.
# Ref: "dolly in, dolly out, dolly left, dolly right, jib up, jib down, static"

LTX23_CAMERA_BY_COMP = {
    "extreme_close": "slow dolly in, tight framing on the subject, minimal movement",
    "close_up":      "slow dolly in, shallow depth of field, face filling the frame",
    "medium_close_up": "slow push-in, medium close-up, subject centered",
    "medium_shot":   "tracking shot with subtle lateral parallax, steady handheld",
    "wide_shot":     "slow dolly out to a wide establishing frame, static hold",
    "over_shoulder": "slow dolly right past the shoulder, shallow depth of field",
    "dutch_angle":   "slight handheld drift, tilted frame, uneasy movement",
}

# ============================================================================
# 3. ENHANCED CINEMATIC CUES FUNCTION
# ============================================================================
# Richer lighting and camera descriptions using LTX-2.3's preferred vocabulary.

def _ltx23_cinematic_cues(shot, vcfg=None,
                          _energy_of=None,
                          _shot_feeling=None,
                          _matched_action_verbs=None,
                          _has_stair_motion=None,
                          _is_multi_person_contact_risk=None,
                          _shot_has_visible_people=None,
                          _HIGH_ENERGY=None,
                          _LOW_ENERGY=None,
                          _WONDER=None,
                          _MYSTERY=None):
    """Map a shot's feeling + framing to LTX-2.3-optimized lighting/camera/motion cues.

    Drop-in replacement for _cinematic_cues() with richer, more specific
    descriptions that LTX-2.3's Gemma 3 text encoder processes better.
    """
    w = _shot_feeling(shot)
    energy = _energy_of(shot)

    # ── Camera ──────────────────────────────────────────────────────────────
    camera = LTX23_CAMERA_BY_COMP.get(
        shot.composition,
        "steady tracking shot with subtle handheld movement"
    )
    if energy == "high":
        if "close" in (shot.composition or ""):
            camera = ("slow dolly in, tight on the subject's face, tension "
                      "building in the framing")
        else:
            camera = ("handheld tracking shot, deliberate and controlled, "
                      "slight shake conveying urgency")
    elif energy == "low":
        if "close" in (shot.composition or ""):
            camera = ("almost-static frame, barely perceptible breathing "
                      "movement, intimate close-up")
        else:
            camera = ("slow dolly out, lingering wide frame, static camera "
                      "with only the faintest ambient drift")

    # ── Lighting ────────────────────────────────────────────────────────────
    # Use concrete, physical lighting descriptions the model can render.
    if any(k in w for k in ("anger", "angry", "furious", "rage")):
        lighting = ("harsh directional light from above, high-contrast hard "
                    "shadows, warm red-orange cast on skin, dark background")
    elif any(k in w for k in _HIGH_ENERGY):
        lighting = ("low-key lighting with a single hard source, cool "
                    "desaturated palette, deep shadows on one side of the face, "
                    "sharp falloff at the edges")
    elif any(k in w for k in ("tender", "intimate", "love", "warm")):
        lighting = ("soft warm golden-hour light from the side, golden rim "
                    "light on hair and shoulders, shallow depth of field "
                    "with bokeh in the background")
    elif any(k in w for k in ("sad", "grief", "melancholy", "somber", "lonely", "numb")):
        lighting = ("overcast diffused light, muted desaturated colors, soft "
                    "falloff with no hard shadows, low overall exposure")
    elif any(k in w for k in _WONDER):
        lighting = ("bright lifted natural light, warm atmospheric glow, "
                    "gentle lens flare, airy high-key exposure")
    elif any(k in w for k in _MYSTERY):
        lighting = ("chiaroscuro lighting, pools of light in darkness, "
                    "atmospheric haze and volumetric beams, cool blue fill")
    else:
        lighting = ""

    # ── Motion ──────────────────────────────────────────────────────────────
    avoid_complex = vcfg is None or getattr(vcfg, "avoid_complex_motion", True)
    action_words = _matched_action_verbs(shot)

    if avoid_complex and _has_stair_motion(shot):
        motion = ("the figure holds a steady, mostly-grounded pose at the "
                  "stairs, weight shifting subtly side to side rather than "
                  "full strides. The camera's own slow dolly movement carries "
                  "the sense of vertical progress. Feet stay planted, body "
                  "sways gently. Physically coherent, no foot-placement "
                  "artifacts")
    elif avoid_complex and _is_multi_person_contact_risk(shot):
        motion = ("only the primary figure moves with simple, grounded, "
                  "single-axis motion — a step forward or a lean. Any other "
                  "person in frame stays essentially still, implying "
                  "interaction through proximity and expression rather than "
                  "coordinated physical contact. No merged or overlapping "
                  "limbs, no interpenetration")
    elif action_words:
        verb_str = ", ".join(action_words[:3])
        motion = (f"purposeful physical motion — {verb_str} — with full-body "
                  "engagement and clear weight transfer. Each movement flows "
                  "naturally into the next, maintaining physical coherence "
                  "and consistent body proportions throughout")
    else:
        motion = ("smooth cinematic motion with natural, subtle movement. "
                  "Gentle weight shifts, slight head adjustments, natural "
                  "blinking. Stable identity, consistent proportions, "
                  "physically plausible throughout")

    # ── Dialogue lip-sync safety ────────────────────────────────────────────
    lipsync_friendly = (vcfg is None) or getattr(vcfg, "lipsync_friendly_motion", True)
    is_safety_motion = avoid_complex and (
        _has_stair_motion(shot) or _is_multi_person_contact_risk(shot))
    if (lipsync_friendly and getattr(shot, "is_dialogue", False)
            and _shot_has_visible_people(shot)):
        if "close" in (shot.composition or ""):
            camera = ("almost-static close-up, locked-off frame with barely "
                      "perceptible push-in, face centered and sharply focused")
        elif shot.composition == "wide_shot":
            camera = ("static wide frame, locked off, only faint ambient drift "
                      "in the background")
        else:
            camera = ("steady near-locked frame, subtle handheld breathing, "
                      "face clearly in the center third of frame")
        if not is_safety_motion:
            motion = ("the character holds a stable, front-facing pose with "
                      "subtle eyebrow and eyelid movement, natural blinking, "
                      "slight weight shifts. Head position stays consistent, "
                      "minimal body and camera movement so the face remains "
                      "sharply in frame throughout")

    return {"energy": energy, "camera": camera, "lighting": lighting, "motion": motion}


# ============================================================================
# 4. REPLACEMENT LLM INSTRUCTION FOR generate_motion_prompts()
# ============================================================================
# This is the core upgrade: the LLM instruction that generates per-shot motion
# prompts.  The old version asked for ~30 words.  LTX-2.3's guides show that
# 80-150 word prompts structured as chronological paragraphs produce
# dramatically better adherence.

def build_ltx23_motion_extraction_prompt(blocks_str: str, fewshots: str) -> str:
    """Build the LLM instruction for motion prompt extraction, LTX-2.3 style.

    This replaces the original ~30-word instruction with one that produces
    80-150 word prompts following LTX-2.3's recommended structure.
    """
    return (
        "For each shot, write a VIDEO ANIMATION PROMPT optimized for the "
        "LTX-2.3 video generation model. Each prompt should be 80-150 words, "
        "written as a SINGLE FLOWING PARAGRAPH in present tense. LTX-2.3 "
        "responds best to long, detailed prompts — the more specific you are "
        "about subject, action, lighting, camera movement, and audio, the "
        "closer the output matches the intent.\n\n"

        "STRUCTURE (follow this order for every prompt):\n"
        "  1. MAIN ACTION: one clear sentence describing the primary physical "
        "action — what the subject does, where, and how. This anchors the "
        "model's generation.\n"
        "  2. MOTION SPECIFICS: describe body movement chronologically as it "
        "should appear on screen. Be literal and physical: 'shifts weight to "
        "left foot, turns head slowly toward the camera' not 'moves gracefully'. "
        "Use concrete mechanics the model can parse, not subjective qualities.\n"
        "  3. ENVIRONMENT & ATMOSPHERE: name textures, surfaces, weather, "
        "atmospheric elements (fog, dust, rain, particles) and any ambient "
        "environmental motion (wind, water, flickering light). Use specific "
        "spatial language.\n"
        "  4. CAMERA DIRECTION: state the camera type, position, and movement "
        "explicitly. Preferred terms: static camera, slow dolly in, dolly out, "
        "dolly left, dolly right, jib up, jib down, tracking shot, handheld, "
        "slow pan left/right, close-up, wide shot, low angle, high angle. "
        "Describe what the subject looks like AFTER the camera movement to help "
        "the model complete the motion.\n"
        "  5. LIGHTING: describe the physical light source, direction, quality "
        "(hard/soft), color temperature, and any visible effects (shadows, "
        "reflections, rim light, lens flare).\n"
        "  6. AUDIO (if the shot has any): describe ambient sound environment "
        "and any non-speech audio (footsteps, wind, rain, room tone, distant "
        "traffic). Keep this brief.\n\n"

        "CRITICAL RULES — ALWAYS FOLLOW:\n"
        "• CHRONOLOGICAL: describe actions in the order they should happen. "
        "The model maps prompt text linearly to the temporal dimension of the "
        "video. If you describe the ending before the beginning, the output "
        "scrambles the sequence.\n"
        "• ONE MAIN ACTION per 2-3 seconds of video. Do not pack five actions "
        "into one prompt — the model compresses or skips actions when overloaded.\n"
        "• BE LITERAL, NOT METAPHORICAL: describe what the camera physically "
        "sees, not what the scene means. 'Two men stand three feet apart, both "
        "leaning forward with clenched fists' not 'a tense confrontation'.\n"
        "• NO CONFLICTING DIRECTIONS: 'fast action in slow motion' sends "
        "contradictory signals. Be internally consistent.\n"
        "• EXPRESS EMOTION THROUGH PHYSICAL CUES, not abstract labels: "
        "'eyebrows drawn together, jaw set, shoulders tense' not 'angry'.\n\n"

        "GLOBAL RULE — NO MOUTH, JAW, OR SPEECH MOTION:\n"
        "This pipeline applies lip-sync after animation. The lip-sync process "
        "replaces the mouth region frame-by-frame. If the animation already "
        "contains mouth movement, the two signals compete and degrade quality.\n"
        "  • NEVER write: speaks, talks, sings, whispers, shouts, mouth moves, "
        "mouth opens, jaw shifts, lips forming words, or any synonym.\n"
        "  • For ALL shots: describe only posture, gesture, head orientation, "
        "eye movement, eyebrow expression, breathing, and body/camera motion.\n\n"

        "MATCH THE STILL: animate ONLY what is present in 'depicted in still' / "
        "'visual scene'. NEVER introduce a person, animal, vehicle, or object "
        "not already visible. Every subject that moves must already be in frame.\n\n"

        "SPEAKING characters ([SPEAKING] tag): keep the face clearly in frame, "
        "both hands away from the face. Describe posture, head tilt, eye focus, "
        "and subtle body language. Do NOT have them eat, drink, or raise anything "
        "to the mouth/chin/face.\n\n"

        "[NO PEOPLE] shots: describe only environment/object/lighting/weather/"
        "camera motion. No people, faces, expressions, or human actions.\n\n"

        "[SIMPLIFY: stairs]: do NOT ask for actual stepping — describe the figure "
        "in a steady pose near the stairs; let camera movement carry the sense "
        "of progress.\n"
        "[SIMPLIFY: multi-person contact]: give ONE person simple motion, the "
        "other stays still — imply interaction through proximity/expression.\n"
        "[ACTION BEAT]: name the SPECIFIC physical action and how the character "
        "interacts with the environment. Favor dynamic, purposeful movement.\n"
        "[WIDE/DIALOGUE-HEAVY]: face is small in frame, keep motion atmospheric "
        "(wind, light, distant movement) rather than focused on any character.\n\n"

        "Study these examples for the LEVEL of detail, structure, and length "
        "to aim for:\n"
        f"{fewshots}\n\n"

        "SHOTS:\n" + blocks_str + "\n\n"
        'Return ONLY JSON: [{"i":0,"motion":"<80-150 word motion prompt>"}]'
    )


# ============================================================================
# 5. ENHANCED VIDEO PROMPT FALLBACK
# ============================================================================

def _ltx23_video_prompt(shot, theme, vcfg=None,
                        _orig_cinematic_cues=None,
                        _clip_prompt=None,
                        _visual_safe_description=None,
                        _sanitize_motion_prompt_to_match_image=None,
                        _lipsync_motion_framing_cue=None,
                        _shot_has_visible_people=None,
                        _nonhuman_motion_prompt=None):
    """LTX-2.3-optimized fallback video prompt (used when no LLM motion_prompt).

    Produces a 60-100 word prompt instead of the original 15-20 word fragment,
    following the recommended structure: main action → environment → camera →
    lighting.
    """
    budget = int(vcfg.motion_prompt_max_chars) if vcfg is not None else 1400
    framing = _lipsync_motion_framing_cue(shot)

    # If the shot already has an LLM-authored motion prompt, use it.
    if shot.motion_prompt:
        prompt = _sanitize_motion_prompt_to_match_image(shot, shot.motion_prompt, vcfg)
        if framing and framing not in prompt:
            prompt = prompt.rstrip(". ") + ". " + framing
        return _clip_prompt(prompt, budget)

    # Non-human scenes.
    if (vcfg is not None
            and getattr(vcfg, "motion_prompts_respect_no_people_scenes", True)
            and not _shot_has_visible_people(shot)):
        return _nonhuman_motion_prompt(shot, vcfg,
                                       cinematic=(vcfg is None or getattr(vcfg, "cinematic_motion", True)))

    # ── Build a richer fallback prompt ──────────────────────────────────────
    desc = _visual_safe_description(shot).strip().rstrip(".")
    # Take the first two clauses (not just the first) for more context.
    clauses = re.split(r'(?<=[.;])\s+', desc)
    if len(clauses) > 3:
        desc = ". ".join(clauses[:3])
    words = desc.split()
    if len(words) > 40:
        desc = " ".join(words[:40])

    parts = [desc]

    if vcfg is None or getattr(vcfg, "cinematic_motion", True):
        c = _orig_cinematic_cues(shot, vcfg)

        # Environment/atmosphere
        setting = getattr(shot, "setting", "") or ""
        if setting:
            # Extract atmosphere words
            atmo_words = []
            for term in ("rain", "fog", "mist", "wind", "dust", "smoke",
                         "snow", "sunlight", "moonlight", "neon", "shadow"):
                if term in setting.lower():
                    atmo_words.append(term)
            if atmo_words:
                parts.append(f"Atmosphere: {', '.join(atmo_words)} visible in the scene")

        # Lighting (use the richer LTX-2.3 descriptions)
        if c.get("lighting"):
            parts.append(f"Lighting: {c['lighting']}")

        # Motion with physical detail
        parts.append(c["motion"])

        # Camera with LTX-2.3 vocabulary
        parts.append(f"Camera: {c['camera']}")

    if framing:
        parts.append(framing)

    result = ". ".join(b for b in parts if b).rstrip(". ") + "."
    return _clip_prompt(
        _sanitize_motion_prompt_to_match_image(shot, result, vcfg),
        budget,
    )


# ============================================================================
# 6. ENHANCED NON-HUMAN MOTION PROMPT
# ============================================================================

def _ltx23_nonhuman_motion_prompt(shot, vcfg=None, cinematic=True,
                                   _nonhuman_visual_description=None,
                                   _cinematic_cues_fn=None,
                                   _clip_prompt=None,
                                   _PEOPLE_WORD_RE=None):
    """LTX-2.3-optimized motion prompt for scenes with no visible people.

    Produces richer environment/atmosphere descriptions that LTX-2.3's
    text encoder responds to.
    """
    budget = int(vcfg.motion_prompt_max_chars) if vcfg is not None else 1400
    desc = _nonhuman_visual_description(shot).rstrip(".")
    # Keep more of the description than the original (first 2 clauses).
    clauses = re.split(r'(?<=[.;])\s+', desc)
    if len(clauses) > 3:
        desc = ". ".join(clauses[:3])
    words = desc.split()
    if len(words) > 30:
        desc = " ".join(words[:30])

    bits: List[str] = []
    if desc:
        bits.append(desc)

    # Add atmosphere and environmental motion cues.
    setting = getattr(shot, "setting", "") or ""
    mood = getattr(shot, "mood", "") or ""
    env_motion_parts = []
    if "rain" in setting.lower() or "rain" in mood.lower():
        env_motion_parts.append("rain streaks fall across the frame, puddles ripple on surfaces")
    if "wind" in setting.lower() or "storm" in setting.lower():
        env_motion_parts.append("wind moves through the scene, rustling leaves and loose debris")
    if "fire" in setting.lower() or "flame" in setting.lower():
        env_motion_parts.append("flames flicker and dance, casting shifting orange light")
    if "water" in setting.lower() or "ocean" in setting.lower() or "sea" in setting.lower():
        env_motion_parts.append("water moves with gentle waves, light plays across the surface")
    if "fog" in setting.lower() or "mist" in setting.lower():
        env_motion_parts.append("fog drifts slowly through the scene, diffusing light sources")
    if not env_motion_parts:
        env_motion_parts.append("subtle ambient motion — light shifts, particles drift, surfaces catch changing reflections")
    bits.append(". ".join(env_motion_parts))

    if cinematic:
        c = _cinematic_cues_fn(shot, vcfg)
        env_motion = c["motion"]
        # Remove human-centric motion fragments.
        env_motion = _PEOPLE_WORD_RE.sub(" ", env_motion)
        env_motion = re.sub(
            r"(?i)\b(face|mouth|eyes?|expression|speaking|talking|singing|gesturing|"
            r"figure|person|character|pose|posture|eyebrow|eyelid|blinking)\b",
            " ", env_motion)
        env_motion = re.sub(r"\s+", " ", env_motion).strip(" ,;:-")
        bits.append(f"Camera: {c['camera']}")
        if c.get("lighting"):
            bits.append(f"Lighting: {c['lighting']}")
        if env_motion:
            bits.append(env_motion)

    bits.append("Only environmental, object, lighting, weather, or camera motion; no people appear")
    return _clip_prompt(". ".join(b for b in bits if b) + ".", budget)


# ============================================================================
# 7. IMAGE PROMPT INSTRUCTION ADDON
# ============================================================================
# Extra guidance injected into the generate_image_prompts() cached prefix
# to improve still images that will seed LTX-2.3's i2v pipeline.

LTX23_IMAGE_PROMPT_ADDON = (
    "\n\nIMAGE-TO-VIDEO OPTIMIZATION (these stills seed LTX-2.3 video generation):\n"
    "• LIGHTING SPECIFICITY: describe the physical light source, its direction, "
    "quality (hard or soft), and color temperature. 'Warm golden-hour sunlight "
    "from the left' is actionable; 'nice lighting' is not. Name rim lights, "
    "fill lights, and any visible light effects (lens flare, caustics, volumetric "
    "beams, reflections on wet surfaces).\n"
    "• ATMOSPHERE & TEXTURE: include surface materials (rough stone, smooth metal, "
    "worn fabric, glossy surfaces) and atmospheric elements (fog, mist, rain, dust "
    "particles, smoke) — these give the video model physical properties to animate.\n"
    "• DEPTH LAYERS: specify foreground, midground, and background with concrete "
    "elements in each — the video model uses depth separation to generate "
    "parallax and natural motion.\n"
    "• COLOR PALETTE: name specific colors and their relationships (high contrast, "
    "muted, monochromatic, complementary warm-cool) rather than abstract mood words.\n"
    "• CAMERA FRAMING for VIDEO: the still establishes the camera position for the "
    "video. Match the composition to the intended motion: close-ups for dialogue "
    "and subtle emotion, wide shots for establishing and environment, medium shots "
    "for action. State the lens character: 'shallow depth of field' for intimate "
    "shots, 'deep focus' for establishing shots.\n"
)

# ============================================================================
# 8. APPLY PATCH
# ============================================================================

def apply(sta_module, *, patch_video=True, patch_image=True, verbose=True):
    """Monkey-patch story_to_animation with LTX-2.3-optimized prompt generation.

    Parameters
    ----------
    sta_module : module
        The imported story_to_animation module (typically ``import
        story_to_animation as sta``).
    patch_video : bool
        Patch motion/video prompt generation.
    patch_image : bool
        Patch image prompt generation with i2v-aware guidance.
    verbose : bool
        Log what was patched.
    """
    patched = []

    # ── Grab originals we need to wrap ──────────────────────────────────────
    orig_cinematic_cues = sta_module._cinematic_cues
    orig_clip_prompt = sta_module._clip_prompt
    orig_visual_safe_desc = sta_module._visual_safe_description
    orig_sanitize_motion = sta_module._sanitize_motion_prompt_to_match_image
    orig_lipsync_framing = sta_module._lipsync_motion_framing_cue
    orig_shot_has_people = sta_module._shot_has_visible_people
    orig_nonhuman_desc = sta_module._nonhuman_visual_description
    orig_nonhuman_motion = sta_module._nonhuman_motion_prompt
    orig_energy_of = sta_module._energy_of
    orig_shot_feeling = sta_module._shot_feeling
    orig_matched_actions = sta_module._matched_action_verbs
    orig_has_stairs = sta_module._has_stair_motion
    orig_is_contact = sta_module._is_multi_person_contact_risk
    orig_people_re = sta_module._PEOPLE_WORD_RE

    HIGH = sta_module._HIGH_ENERGY
    LOW = sta_module._LOW_ENERGY
    WONDER = sta_module._WONDER
    MYSTERY = sta_module._MYSTERY

    if patch_video:
        # ── A. Increase motion_prompt_max_chars default ─────────────────────
        # The VideoConfig dataclass default can't be changed after definition,
        # but we can set it on any existing VideoConfig instances and change
        # the class-level default for new ones.
        try:
            old_default = sta_module.VideoConfig.motion_prompt_max_chars
            sta_module.VideoConfig.motion_prompt_max_chars = 1400
            patched.append(f"VideoConfig.motion_prompt_max_chars: {old_default} → 1400")
        except Exception:
            pass  # dataclass may be frozen

        # ── B. Replace _CAMERA_BY_COMP ──────────────────────────────────────
        sta_module._CAMERA_BY_COMP = LTX23_CAMERA_BY_COMP
        patched.append("_CAMERA_BY_COMP → LTX-2.3 camera vocabulary")

        # ── C. Replace _MOTION_PROMPT_FEWSHOTS ──────────────────────────────
        sta_module._MOTION_PROMPT_FEWSHOTS = LTX23_MOTION_PROMPT_FEWSHOTS
        patched.append("_MOTION_PROMPT_FEWSHOTS → LTX-2.3 style (80-120 words each)")

        # ── D. Replace _cinematic_cues ──────────────────────────────────────
        def new_cinematic_cues(shot, vcfg=None):
            return _ltx23_cinematic_cues(
                shot, vcfg,
                _energy_of=orig_energy_of,
                _shot_feeling=orig_shot_feeling,
                _matched_action_verbs=orig_matched_actions,
                _has_stair_motion=orig_has_stairs,
                _is_multi_person_contact_risk=orig_is_contact,
                _shot_has_visible_people=orig_shot_has_people,
                _HIGH_ENERGY=HIGH, _LOW_ENERGY=LOW,
                _WONDER=WONDER, _MYSTERY=MYSTERY,
            )
        sta_module._cinematic_cues = new_cinematic_cues
        patched.append("_cinematic_cues → LTX-2.3 lighting/camera vocabulary")

        # ── E. Replace _video_prompt ────────────────────────────────────────
        def new_video_prompt(shot, theme, vcfg=None):
            return _ltx23_video_prompt(
                shot, theme, vcfg,
                _orig_cinematic_cues=new_cinematic_cues,
                _clip_prompt=orig_clip_prompt,
                _visual_safe_description=orig_visual_safe_desc,
                _sanitize_motion_prompt_to_match_image=orig_sanitize_motion,
                _lipsync_motion_framing_cue=orig_lipsync_framing,
                _shot_has_visible_people=orig_shot_has_people,
                _nonhuman_motion_prompt=sta_module._nonhuman_motion_prompt,
            )
        sta_module._video_prompt = new_video_prompt
        patched.append("_video_prompt → LTX-2.3 fallback (60-100 words)")

        # ── F. Replace _nonhuman_motion_prompt ──────────────────────────────
        def new_nonhuman_motion(shot, vcfg=None, cinematic=True):
            return _ltx23_nonhuman_motion_prompt(
                shot, vcfg, cinematic,
                _nonhuman_visual_description=orig_nonhuman_desc,
                _cinematic_cues_fn=new_cinematic_cues,
                _clip_prompt=orig_clip_prompt,
                _PEOPLE_WORD_RE=orig_people_re,
            )
        sta_module._nonhuman_motion_prompt = new_nonhuman_motion
        patched.append("_nonhuman_motion_prompt → LTX-2.3 atmosphere-rich")

        # ── G. Patch generate_motion_prompts LLM instruction ───────────────
        # This is the most impactful change: the LLM instruction that creates
        # per-shot motion prompts. We wrap the original function and replace
        # the prompt text inside it.
        orig_generate_motion = sta_module.generate_motion_prompts

        def patched_generate_motion_prompts(shots, theme, vcfg, batch_size=6, cinematic=True):
            """LTX-2.3-enhanced motion prompt generation.

            Temporarily bumps motion_prompt_max_chars to 1400 and replaces the
            LLM instruction with the LTX-2.3-optimized version, then delegates
            to the original function for all the safety/sanitization logic.
            """
            # Bump the budget on the vcfg instance for this run.
            old_budget = getattr(vcfg, "motion_prompt_max_chars", 900)
            vcfg.motion_prompt_max_chars = max(old_budget, 1400)

            # The original function reads _MOTION_PROMPT_FEWSHOTS at module
            # level — we already patched that. It also reads the LLM prompt
            # text inline. We can't easily replace the inline text, but we CAN
            # ensure the few-shots and budget are set correctly.
            # The inline prompt says "max ~30 words" — this is baked in the
            # function body. To truly override it, we need to replace the
            # function's prompt construction.

            # Strategy: patch the function's LLM call by temporarily replacing
            # the get_openai_prompt_response function with a wrapper that
            # intercepts and rewrites the motion-prompt instruction.
            try:
                ng = sta_module.ng
                orig_gpr = ng.get_openai_prompt_response

                def intercepted_gpr(prompt_text, *args, **kwargs):
                    # If this looks like a motion-prompt extraction call,
                    # replace the instruction header.
                    if ("SHORT video-animation prompt" in prompt_text
                            or "max ~30" in prompt_text
                            or "CAMERA movement" in prompt_text):
                        # Extract the SHOTS block from the original prompt.
                        shots_marker = "SHOTS:\n"
                        shots_idx = prompt_text.find(shots_marker)
                        if shots_idx >= 0:
                            shots_block = prompt_text[shots_idx + len(shots_marker):]
                            # Remove the trailing JSON instruction.
                            json_marker = "\nReturn ONLY JSON:"
                            json_idx = shots_block.find(json_marker)
                            if json_idx >= 0:
                                shots_block = shots_block[:json_idx]
                            # Build the LTX-2.3 replacement prompt.
                            prompt_text = build_ltx23_motion_extraction_prompt(
                                shots_block.strip(),
                                LTX23_MOTION_PROMPT_FEWSHOTS,
                            )
                    return orig_gpr(prompt_text, *args, **kwargs)

                ng.get_openai_prompt_response = intercepted_gpr
                orig_generate_motion(shots, theme, vcfg, batch_size=batch_size, cinematic=cinematic)
            finally:
                try:
                    ng.get_openai_prompt_response = orig_gpr
                except Exception:
                    pass
                vcfg.motion_prompt_max_chars = old_budget

        sta_module.generate_motion_prompts = patched_generate_motion_prompts
        patched.append("generate_motion_prompts → LTX-2.3 LLM instruction (80-150 words)")

    if patch_image:
        # ── H. Enhance generate_image_prompts instruction ───────────────────
        # Inject the LTX-2.3 i2v-aware guidance into the cached prefix.
        orig_generate_images = sta_module.generate_image_prompts

        def patched_generate_image_prompts(shots, characters, theme,
                                            batch_size=5, cinematic=True,
                                            expressive=True,
                                            visual_treatment=None,
                                            arc_map=None,
                                            use_continuity_context=True):
            """LTX-2.3-enhanced image prompt generation.

            Intercepts the LLM call to inject i2v-optimized guidance for
            lighting, atmosphere, texture, and depth layering.
            """
            try:
                ng = sta_module.ng
                orig_gpr = ng.get_openai_prompt_response

                def intercepted_gpr(prompt_text, *args, **kwargs):
                    # The generate_image_prompts function uses cached_prefix.
                    # We inject our addon into the cached_prefix kwarg.
                    if "cached_prefix" in kwargs:
                        cp = kwargs["cached_prefix"]
                        if (cp and "labelled-section format" in cp
                                and LTX23_IMAGE_PROMPT_ADDON not in cp):
                            kwargs["cached_prefix"] = cp + LTX23_IMAGE_PROMPT_ADDON
                    return orig_gpr(prompt_text, *args, **kwargs)

                ng.get_openai_prompt_response = intercepted_gpr
                orig_generate_images(
                    shots, characters, theme,
                    batch_size=batch_size, cinematic=cinematic,
                    expressive=expressive, visual_treatment=visual_treatment,
                    arc_map=arc_map,
                    use_continuity_context=use_continuity_context,
                )
            finally:
                try:
                    ng.get_openai_prompt_response = orig_gpr
                except Exception:
                    pass

        sta_module.generate_image_prompts = patched_generate_image_prompts
        patched.append("generate_image_prompts → LTX-2.3 i2v lighting/atmosphere/texture guidance")

    if verbose and patched:
        logger.info("[LTX-2.3 Prompt Guide] Applied %d patches:\n  • %s",
                    len(patched), "\n  • ".join(patched))
        # Also print for notebook visibility.
        print(f"[LTX-2.3 Prompt Guide] Applied {len(patched)} patches:")
        for p in patched:
            print(f"  ✓ {p}")

    return patched

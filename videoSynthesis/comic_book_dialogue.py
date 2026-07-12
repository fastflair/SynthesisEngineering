"""
comic_book_dialogue.py
======================
Character-voice and emotional-delivery layer for the comic dialogue pass.

The dialogue reviewer already rewrites lines for distinctiveness, rhythm and
subtext, but two things were missing, and they are exactly what makes a reader
*hear* a character:

  1. WHO the speaker is — where they're from and what they carry into English.
     A character from Marseille, from Kingston, from a Yiddish-speaking
     household sounds nothing like a flat narrator voice. This module builds a
     CharacterVoiceProfile (origin, accent, heritage, dialect markers, a small
     signature lexicon, register) so the reviewer can write each line in that
     real voice — through authentic word choice, idiom and rhythm, with light
     readable accent markers, never mocking phonetic caricature.

  2. HOW they sound RIGHT NOW — the emotional delivery. An angry character
     SHOUTS. A drunk one slurs. A delighted one bubbles over. A seducer purrs.
     A terrified one stammers. This module models that as a general, data-driven
     DELIVERY system: every emotional register maps to (a) guidance the reviewer
     writes the words to, and (b) the bubble shape the renderer already supports
     (shout / angry / excited / sarcastic / scared / tender / cold / whisper),
     plus a small, SAFE deterministic enforcement pass (a shout is upper-cased;
     the bubble shape is synced to the stated delivery) so the look always
     matches the feeling even when the LLM forgets.

Design mirrors the continuity modules: the LLM does the hard creative writing,
plain Python holds the deterministic guarantees, and every step fails soft.

Integration
-----------
Step 1 — build profiles once (cached on the project):
    from comic_book_dialogue import build_voice_profiles
    project.voice_profiles = build_voice_profiles(
        project.characters, project.story_idea, graph=project.character_graph)

Step 2 — feed the reviewer the richer voice + delivery guidance:
    from comic_book_dialogue import voice_profile_guide, delivery_guidance_block
    # add voice_profile_guide(profiles, speakers) and delivery_guidance_block()
    # into the _review_act_dialog prompt, and ask the reviewer to tag each line
    # with a "delivery" field.

Step 3 — enforce delivery deterministically AFTER revisions are applied:
    from comic_book_dialogue import apply_dialogue_delivery
    apply_dialogue_delivery(script, project.voice_profiles)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM helpers (reuse the project's utilities; lazy so the module imports without
# the ML stack for unit-testing the deterministic parts).
# ---------------------------------------------------------------------------

def _llm(prompt: str, temperature: float = 0.4, large: bool = True) -> str:
    from novel_generator import (
        get_openai_prompt_response, openai_model, openai_model_large, USE_GROK,
    )
    model = openai_model_large if large else openai_model
    return get_openai_prompt_response(
        prompt, temperature=temperature, openai_model=model, use_grok=USE_GROK,
    )


def _parse(text: str):
    from novel_generator import parse_json_response
    return parse_json_response(text)


def _get(obj, attr, default=''):
    """Read an attribute from a Character object OR a dict."""
    if obj is None:
        return default
    v = getattr(obj, attr, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(attr)
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return v


# ===========================================================================
# CHARACTER VOICE PROFILE
# ===========================================================================

@dataclass
class CharacterVoiceProfile:
    """A resolved, structured spec for how one character sounds."""
    name: str = ''
    gender: str = ''
    origin: str = ''              # place/culture of origin ("Marseille, France")
    accent: str = ''              # how it colours their English ("French-accented")
    heritage: str = ''            # cultural/linguistic background informing idiom
    dialect_markers: List[str] = field(default_factory=list)   # light grammar/word patterns
    signature_lexicon: List[str] = field(default_factory=list)  # words/exclamations they use
    register: str = ''            # casual | formal | vulgar | refined | streetwise | …
    cadence: str = ''             # rhythm of speech
    verbal_tics: List[str] = field(default_factory=list)
    catchphrases: List[str] = field(default_factory=list)
    humor_style: str = ''
    speech_signature: str = ''    # a synthesised paragraph for the LLM

    def guide_block(self) -> str:
        """Compact multi-line block describing this voice for the reviewer LLM."""
        out = [f"  {self.name}:"]
        if self.origin:
            out.append(f"    from: {self.origin}")
        if self.accent:
            out.append(f"    accent: {self.accent}")
        if self.heritage:
            out.append(f"    heritage/idiom: {self.heritage}")
        if self.register:
            out.append(f"    register: {self.register}")
        if self.cadence:
            out.append(f"    cadence: {self.cadence}")
        if self.dialect_markers:
            out.append(f"    dialect markers (use lightly, stay readable): "
                       f"{'; '.join(self.dialect_markers[:5])}")
        if self.signature_lexicon:
            out.append(f"    signature words/exclamations (sprinkle, don't overuse): "
                       f"{', '.join(self.signature_lexicon[:8])}")
        if self.verbal_tics:
            out.append(f"    verbal tics: {', '.join(self.verbal_tics[:4])}")
        if self.catchphrases:
            out.append(f"    catchphrases: {'; '.join(self.catchphrases[:3])}")
        if self.humor_style:
            out.append(f"    humor: {self.humor_style}")
        if self.speech_signature:
            out.append(f"    voice: {self.speech_signature[:200]}")
        return "\n".join(out)


# Deterministic seed: pull whatever voice fields the Character already carries.
def _seed_profile(c) -> CharacterVoiceProfile:
    name = str(_get(c, 'name', '')).strip()
    p = CharacterVoiceProfile(
        name=name,
        gender=str(_get(c, 'gender', '')),
        accent=str(_get(c, 'dialect', '')),          # Character.dialect often holds accent
        register=str(_get(c, 'vocabulary_level', '')),
        cadence=str(_get(c, 'cadence', '')),
        humor_style=str(_get(c, 'humor_style', '')),
    )
    tics = _get(c, 'verbal_tics', [])
    p.verbal_tics = list(tics) if isinstance(tics, list) else []
    cps = _get(c, 'catchphrases', [])
    p.catchphrases = list(cps) if isinstance(cps, list) else []
    sp = str(_get(c, 'speech_pattern', '')) or str(_get(c, 'voice_profile', ''))
    p.speech_signature = sp
    return p


def build_voice_profiles(characters: list, story_idea=None, graph=None,
                         enrich: bool = True, batch_size: int = 8
                         ) -> Dict[str, CharacterVoiceProfile]:
    """Build {name: CharacterVoiceProfile} for the cast.

    Seeds deterministically from each Character's existing voice fields, then
    (when ``enrich``) runs ONE batched LLM pass that reads each character's
    backstory + traits and fills in origin / accent / heritage / dialect markers
    / signature lexicon / register. Resilient: any failure leaves the
    deterministic seed in place, so dialogue generation never blocks on this.
    """
    profiles: Dict[str, CharacterVoiceProfile] = {}
    seeds: List = []
    for c in (characters or []):
        nm = str(_get(c, 'name', '')).strip()
        if not nm:
            continue
        profiles[nm] = _seed_profile(c)
        seeds.append(c)

    if not enrich or not seeds:
        return profiles

    for start in range(0, len(seeds), batch_size):
        batch = seeds[start:start + batch_size]
        try:
            _enrich_batch(batch, profiles, story_idea)
        except Exception as e:
            logger.warning(f"[Voice] Enrichment batch failed ({e}); using seeds.")
    logger.info(f"[Voice] Built {len(profiles)} character voice profile(s).")
    return profiles


def _enrich_batch(batch: list, profiles: Dict[str, CharacterVoiceProfile],
                  story_idea) -> None:
    char_blocks = []
    for c in batch:
        nm = str(_get(c, 'name', '')).strip()
        backstory = str(_get(c, 'backstory', ''))[:400]
        role = str(_get(c, 'role', ''))
        traits = _get(c, 'traits', [])
        traits_s = ', '.join(traits[:6]) if isinstance(traits, list) else str(traits)
        age = str(_get(c, 'age', ''))
        existing_dialect = str(_get(c, 'dialect', ''))
        char_blocks.append(
            f'- name: {nm}\n'
            f'  role: {role}\n  age: {age}\n  traits: {traits_s}\n'
            f'  existing dialect note: {existing_dialect or "(none)"}\n'
            f'  backstory: {backstory}'
        )
    chars_text = "\n".join(char_blocks)
    genre = _get(story_idea, 'genre', '') if story_idea is not None else ''
    setting = _get(story_idea, 'premise', '')[:200] if story_idea is not None else ''

    prompt = f"""You are a dialect and dialogue coach for a graphic novel. For each
character, infer how they actually SOUND when they speak, grounded in their
background. Be specific and AUTHENTIC, never a mocking caricature.

STORY GENRE: {genre}
SETTING/PREMISE: {setting}

CHARACTERS:
{chars_text}

For EACH character return:
  - origin: place/culture they're from (city + country/region), or "" if unclear
  - accent: how their origin colours their English (e.g. "French-accented English,
    soft 'h's, melodic"; "Jamaican Patois inflection"; "flat Midwestern American"),
    or "" if a neutral standard speaker
  - heritage: cultural/linguistic background that flavours their idiom (e.g.
    "Ashkenazi Jewish — Yiddish loanwords and rhetorical questions"; "devout
    rural Catholic"), or ""
  - dialect_markers: 2-4 LIGHT, READABLE speech patterns (grammar/syntax/word
    choices), NOT heavy phonetic spelling. e.g. ["drops the subject pronoun",
    "ends statements as questions, no?", "double negatives"]
  - signature_lexicon: 3-8 authentic words/exclamations this person would
    genuinely use (slang, oaths, loanwords, fillers) — real, specific, not
    stereotype confetti. e.g. ["oy", "bubbeleh", "feh"] or ["putain", "alors",
    "non?"] or ["wah gwaan", "irie", "bredren"]
  - register: casual | formal | vulgar | refined | streetwise | clinical | folksy | …
  - speech_signature: ONE vivid sentence summarising how they sound.

RULES:
- Authenticity over stereotype. A few true markers beat a costume of clichés.
- Keep accent renderable through WORD CHOICE and light markers; the reader must
  still read it easily.
- If a character is plainly a neutral-standard speaker, say so (empty accent).

Return ONLY a JSON array:
[{{"name":"...","origin":"...","accent":"...","heritage":"...",
   "dialect_markers":["..."],"signature_lexicon":["..."],"register":"...",
   "speech_signature":"..."}}]
"""
    raw = _llm(prompt, temperature=0.5)
    parsed = _parse(raw)
    if not isinstance(parsed, list):
        return
    for item in parsed:
        if not isinstance(item, dict):
            continue
        nm = str(item.get('name', '')).strip()
        prof = profiles.get(nm) or _match_profile(nm, profiles)
        if not prof:
            continue
        if item.get('origin'):   prof.origin = str(item['origin']).strip()
        if item.get('accent'):   prof.accent = str(item['accent']).strip()
        if item.get('heritage'): prof.heritage = str(item['heritage']).strip()
        if item.get('register'): prof.register = str(item['register']).strip()
        if item.get('speech_signature'):
            prof.speech_signature = str(item['speech_signature']).strip()
        dm = item.get('dialect_markers')
        if isinstance(dm, list):
            prof.dialect_markers = [str(x).strip() for x in dm if str(x).strip()][:5]
        lex = item.get('signature_lexicon')
        if isinstance(lex, list):
            prof.signature_lexicon = [str(x).strip() for x in lex if str(x).strip()][:8]


def _match_profile(name: str, profiles: Dict[str, CharacterVoiceProfile]
                   ) -> Optional[CharacterVoiceProfile]:
    nl = name.lower()
    for k, v in profiles.items():
        if k.lower() == nl:
            return v
    first = nl.split()[0] if nl.split() else nl
    cands = [v for k, v in profiles.items()
             if (k.lower().split()[0] if k.lower().split() else k.lower()) == first]
    return cands[0] if len(cands) == 1 else None


# ===========================================================================
# EMOTIONAL DELIVERY — the general "how they sound right now" model
# ===========================================================================

@dataclass
class DeliveryMode:
    """One emotional delivery style: how to WRITE it and how to DRAW it."""
    name: str
    bubble_type: str              # renderer shape: shout|angry|excited|sarcastic|
                                  #   scared|tender|cold|whisper|speech
    guidance: str                 # instruction the reviewer writes the line to obey
    upper: bool = False           # deterministic: UPPER-CASE the line (shouting)
    force_excl: bool = False      # deterministic: ensure it ends on '!'
    overrides_speech_only: bool = True  # only change bubble if it's generic 'speech'


# The delivery table. ORDER is the resolution priority for ambiguous emotions.
DELIVERY_MODES: Dict[str, DeliveryMode] = {
    'shout': DeliveryMode(
        'shout', 'shout',
        "FULL SHOUT — raised voice. Short, blasting lines; UPPERCASE the words; "
        "harsh, blunt diction; exclamation. Cut everything that isn't the impact.",
        upper=True, force_excl=True),
    'angry': DeliveryMode(
        'angry', 'angry',
        "Cold or seething anger (not yelling): clipped, cutting, controlled. "
        "Short sentences. Threats stated flatly. Contempt in the word choice.",
        force_excl=False),
    'excited': DeliveryMode(
        'excited', 'excited',
        "Delighted / hyped: energetic, tumbling, exclamatory; words rush together; "
        "superlatives; the speaker can barely contain it.",
        force_excl=True),
    'happy': DeliveryMode(
        'happy', 'excited',
        "Warm and happy: bright, open, easy; light exclamations; generous, "
        "affectionate word choice.",
        force_excl=False),
    'drunk': DeliveryMode(
        'drunk', 'speech',
        "Drunk: slurred and loose — run words together, drop the odd final "
        "consonant ('jus'', 'lemme', 'wha'?'), repeat words, lose the thread "
        "mid-sentence, a hiccup or two (*hic*). Keep it READABLE, not gibberish.",
        force_excl=False),
    'seductive': DeliveryMode(
        'seductive', 'tender',
        "Seductive/sultry: low, slow, teasing; loaded pauses; suggestion and "
        "innuendo working an angle. Calibrate heat to the scene — flirtatious by "
        "default; explicit only when the scene is already an adult one.",
        force_excl=False),
    'tender': DeliveryMode(
        'tender', 'tender',
        "Tender/loving: soft, unhurried, sincere; small intimate words; the "
        "guard is down.",
        force_excl=False),
    'scared': DeliveryMode(
        'scared', 'scared',
        "Frightened: trembling and fragmented — stammered starts ('I— I can't—'), "
        "broken syntax, caught breath, trailing dread.",
        force_excl=False),
    'sad': DeliveryMode(
        'sad', 'speech',
        "Grief/sadness: quiet, halting, spare; sentences trail off (…); the "
        "weight sits in what's NOT said.",
        force_excl=False),
    'sarcastic': DeliveryMode(
        'sarcastic', 'sarcastic',
        "Sarcastic/dry: deadpan, ironic, the loaded word italicised in intent; "
        "says the opposite of what's meant.",
        force_excl=False),
    'cold': DeliveryMode(
        'cold', 'cold',
        "Cold/menacing: flat affect, measured, quietly threatening; no wasted "
        "words; the calm is the threat.",
        force_excl=False),
    'whisper': DeliveryMode(
        'whisper', 'whisper',
        "Whispered/secret: hushed, conspiratorial, close; kept low so only the "
        "listener hears.",
        force_excl=False),
    'commanding': DeliveryMode(
        'commanding', 'speech',
        "Commanding: firm imperatives, no hedging, total authority.",
        force_excl=False),
    'neutral': DeliveryMode(
        'neutral', 'speech',
        "Natural conversational delivery.",
        force_excl=False),
}

# Emotion / mood vocabulary → delivery mode. Conservative, word-boundary matched.
EMOTION_TO_MODE: List[Tuple[Tuple[str, ...], str]] = [
    (('shout', 'yell', 'scream', 'roar', 'bellow', 'furious', 'enraged',
      'rage', 'screaming', 'shouting', 'hollering'), 'shout'),
    (('angry', 'anger', 'seething', 'livid', 'irate', 'incensed', 'hostile',
      'snarling', 'venom', 'contempt'), 'angry'),
    (('excited', 'thrilled', 'ecstatic', 'elated', 'exhilarated', 'giddy',
      'hyped', 'overjoyed'), 'excited'),
    (('happy', 'joy', 'joyful', 'delighted', 'cheerful', 'pleased', 'warm',
      'content'), 'happy'),
    (('drunk', 'drunken', 'intoxicated', 'wasted', 'tipsy', 'sloshed',
      'inebriated', 'slurring', 'slurred'), 'drunk'),
    (('seductive', 'sultry', 'sensual', 'flirtatious', 'flirty', 'seducing',
      'seduce', 'sexy', 'coy', 'come-hither', 'come hither'), 'seductive'),
    (('tender', 'loving', 'affectionate', 'gentle', 'intimate', 'fond'), 'tender'),
    (('scared', 'afraid', 'terrified', 'fearful', 'panicked', 'frightened',
      'petrified', 'trembling', 'dread'), 'scared'),
    (('sad', 'grief', 'sorrow', 'mournful', 'heartbroken', 'despair',
      'weeping', 'tearful', 'melancholy', 'devastated'), 'sad'),
    (('sarcastic', 'sardonic', 'snide', 'dry', 'ironic', 'mocking', 'wry'), 'sarcastic'),
    (('cold', 'icy', 'menacing', 'threatening', 'sinister', 'ominous',
      'callous'), 'cold'),
    (('whisper', 'whispering', 'hushed', 'secret', 'conspiratorial',
      'murmur', 'murmured'), 'whisper'),
    (('command', 'commanding', 'authoritative', 'order', 'imperious'), 'commanding'),
]


def _emotion_word_to_mode(text: str) -> Optional[str]:
    t = (text or '').lower()
    if not t.strip():
        return None
    for words, mode in EMOTION_TO_MODE:
        for w in words:
            if re.search(r'\b' + re.escape(w) + r'\b', t):
                return mode
    return None


def resolve_delivery(panel: Dict, line: Dict) -> str:
    """Resolve the delivery mode for one dialogue line.

    Priority (most authoritative first):
      1. an explicit 'delivery' or 'emotion' field on the line (reviewer-set);
      2. the existing bubble_type, if it is already an emotional one;
      3. strong textual cues IN THE LINE (already shouting / stammered / …);
      4. the panel's emotional register (mood / arc emotion).
    Defaults to 'neutral'. Never guesses an emotional shape from panel mood for a
    line that reads plainly — keeps the deterministic pass conservative.
    """
    # 1. explicit per-line field
    for fld in ('delivery', 'emotion'):
        m = _emotion_word_to_mode(str(line.get(fld, '')))
        if m:
            return m
        raw = str(line.get(fld, '')).strip().lower()
        if raw in DELIVERY_MODES:
            return raw

    # 2. an already-emotional bubble_type
    bt = str(line.get('bubble_type', '')).strip().lower()
    if bt in ('shout', 'angry', 'excited', 'sarcastic', 'scared', 'tender',
              'cold', 'whisper'):
        # map bubble shapes back to a delivery mode of the same name
        return bt

    # 3. textual cues in the line itself
    text = str(line.get('text', ''))
    letters = re.sub(r'[^A-Za-z]', '', text)
    if len(letters) >= 4 and letters.isupper():
        return 'shout'
    if text.count('!') >= 2:
        return 'excited'

    # 4. panel register (only used as a weak hint; still returns neutral unless a
    #    clear emotion word is present)
    panel_emotion = ' '.join(str(panel.get(k, '')) for k in
                             ('mood', '_arc_emotion', 'emotion'))
    m = _emotion_word_to_mode(panel_emotion)
    return m or 'neutral'


# ===========================================================================
# REVIEWER-PROMPT BLOCKS
# ===========================================================================

def voice_profile_guide(profiles: Dict[str, CharacterVoiceProfile],
                        speaker_names: Optional[List[str]] = None) -> str:
    """Rich voice block for the reviewer prompt, limited to the act's speakers."""
    if not profiles:
        return ''
    if speaker_names:
        wanted = set()
        for s in speaker_names:
            p = profiles.get(s) or _match_profile(s, profiles)
            if p:
                wanted.add(p.name)
        items = [p for p in profiles.values() if p.name in wanted]
    else:
        items = list(profiles.values())
    if not items:
        return ''
    blocks = [p.guide_block() for p in items]
    return ("CHARACTER VOICE PROFILES (write every line so THIS person is "
            "unmistakably the one speaking — their origin, accent, idiom and "
            "rhythm):\n" + "\n".join(blocks))


def delivery_guidance_block() -> str:
    """The general emotion→delivery instructions for the reviewer prompt."""
    rows = []
    for key in ('shout', 'angry', 'excited', 'happy', 'drunk', 'seductive',
                'tender', 'scared', 'sad', 'sarcastic', 'cold', 'whisper'):
        m = DELIVERY_MODES[key]
        rows.append(f"  - {key} (bubble_type \"{m.bubble_type}\"): {m.guidance}")
    return (
        "EMOTIONAL DELIVERY — write each line the way the character SOUNDS in "
        "that moment, then tag it. Match the panel's emotion and the speaker's "
        "state:\n" + "\n".join(rows) + "\n"
        "For EVERY line also return a \"delivery\" field set to one of: "
        "shout, angry, excited, happy, drunk, seductive, tender, scared, sad, "
        "sarcastic, cold, whisper, commanding, neutral. Set bubble_type to match "
        "(shout→shout, scared→scared, whisper→whisper, seductive/tender→tender, "
        "menacing→cold, sarcastic→sarcastic, excited/happy→excited, otherwise "
        "speech/thought as appropriate)."
    )


# Cultural-authenticity guardrail, appended to the reviewer's dialect rules.
CULTURAL_AUTHENTICITY_GUARDRAIL = (
    "ACCENT & HERITAGE — render a character's origin and culture through "
    "AUTHENTIC, specific word choice, idiom, rhythm and a few signature terms "
    "from their voice profile. Light, readable accent markers are welcome where "
    "they define the voice (a drunk's slur, a strong regional lilt, a heritage "
    "loanword) — enough to HEAR it, never so heavy it's hard to read and never a "
    "mocking phonetic caricature or a pile of stereotypes. A real person uses "
    "their own words naturally; write them with that dignity."
)


# ===========================================================================
# DETERMINISTIC DELIVERY ENFORCEMENT (post-review)
# ===========================================================================

_SENTENCE_END_RE = re.compile(r'[.!?]+$')


def _shout_caseify(text: str) -> str:
    """Upper-case a line for shouting while keeping it tidy. Idempotent."""
    if not text:
        return text
    up = text.upper()
    # Collapse accidental doubled punctuation, ensure it ends emphatic.
    up = re.sub(r'\s+', ' ', up).strip()
    if not _SENTENCE_END_RE.search(up):
        up += '!'
    elif up.endswith('.'):
        up = up[:-1] + '!'
    return up


def _ensure_excl(text: str) -> str:
    if not text:
        return text
    t = text.rstrip()
    if t.endswith('.'):
        return t[:-1] + '!'
    if not _SENTENCE_END_RE.search(t):
        return t + '!'
    return t


def apply_dialogue_delivery(script: List[Dict],
                            profiles: Optional[Dict[str, CharacterVoiceProfile]] = None
                            ) -> int:
    """Deterministic, conservative enforcement pass run AFTER dialogue review.

    For every dialogue line it resolves the delivery mode and:
      • syncs the bubble shape to the delivery (only upgrading the generic
        'speech' bubble — never clobbering an intentional thought/caption/etc.),
        so an angry line gets the burst shape, a whisper the dashed one, …;
      • makes a shout LOOK like a shout (UPPER-CASE, '!'); a delight end on '!';
      • records the resolved delivery on the line ('delivery') for transparency.

    Text rewriting for accent/slur/seduction is the LLM reviewer's job (guided
    by voice_profile_guide / delivery_guidance_block); this pass only enforces
    the small, safe, reliable things the LLM is inconsistent about.

    Returns the number of lines adjusted.
    """
    changed = 0
    for page in script:
        if not isinstance(page, dict) or page.get('_act_break'):
            continue
        for panel in (page.get('panels') or []):
            for line in (panel.get('dialogue') or []):
                if not isinstance(line, dict):
                    continue
                speaker = str(line.get('speaker', '')).strip()
                if speaker.upper() == 'NARRATOR':
                    continue   # captions are narration, not character voice
                bt = str(line.get('bubble_type', 'speech')).strip().lower()
                if bt in ('caption', 'thought'):
                    # respect deliberate inner-monologue / narration containers
                    line.setdefault('delivery', 'neutral')
                    continue

                mode_key = resolve_delivery(panel, line)
                mode = DELIVERY_MODES.get(mode_key, DELIVERY_MODES['neutral'])
                line_changed = False

                # 1) bubble shape sync (only upgrade the generic speech bubble)
                if mode.bubble_type != 'speech':
                    if bt == 'speech' or not bt:
                        if line.get('bubble_type') != mode.bubble_type:
                            line['bubble_type'] = mode.bubble_type
                            line_changed = True

                # 2) text emphasis for shout / excited
                text = str(line.get('text', ''))
                if text:
                    if mode.upper:
                        new = _shout_caseify(text)
                    elif mode.force_excl:
                        new = _ensure_excl(text)
                    else:
                        new = text
                    if new != text:
                        line['text'] = new
                        line_changed = True

                # 3) record resolved delivery
                if line.get('delivery') != mode_key:
                    line['delivery'] = mode_key
                    line_changed = True

                if line_changed:
                    changed += 1
    if changed:
        logger.info(f"[Delivery] Enforced delivery on {changed} dialogue line(s).")
    return changed

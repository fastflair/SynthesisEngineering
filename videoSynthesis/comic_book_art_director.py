"""
comic_book_art_director.py
==========================
Art Director pass for the comic book pipeline.

Solves the "mask ambiguity" class of errors: panel descriptions written by the
script LLM use shorthand ("the mask", "her suit", "the device") that the image
generation model interprets against its training distribution — a medical mask,
a wetsuit, a smartphone — rather than the specific prop established in the story.

Two components
--------------
1. SeriesVisualBible
   Built once per project from the story concept and character list.
   Maintains a canonical full-description for every recurring prop, costume
   piece, and piece of equipment in the series, keyed by shorthand alias.

   Example entry:
     aliases  : ["mask", "the mask", "diving mask", "her mask"]
     canonical: "blue diving goggle mask with wide tempered glass lens, black
                 silicone skirt, and an adjustable neoprene head strap —
                 distinctively NOT a medical or cloth face covering"

2. art_director_review_panels(script, visual_bible, story_idea, characters)
   Runs after _expand_pages_to_panels (and after _validate_and_repair_panels)
   inside generate_comic_script.  For every panel it:
     a) Detects shorthand references to Visual Bible entries.
     b) Rewrites the description with the full canonical text so the image
        model always receives unambiguous visual instructions.
     c) Returns a corrected script and a change-log for inspection.

Integration
-----------
Step 1 — in synthesize_comic_story() (after registry is built):
    from comic_book_art_director import SeriesVisualBible, build_series_visual_bible
    visual_bible = build_series_visual_bible(story_idea_str, characters, story_idea)

Step 2 — attach to project so it travels with the pickle:
    project.visual_bible = visual_bible      # add field to ComicBookProject

Step 3 — in generate_comic_script() (after _validate_and_repair_panels):
    from comic_book_art_director import art_director_review_panels
    full_script, ad_log = art_director_review_panels(
        full_script, visual_bible, story_idea, characters
    )

Step 4 — in compose_panel_prompt() (add optional parameter):
    def compose_panel_prompt(..., visual_bible=None):
        ...
        if visual_bible:
            vb_clause = visual_bible.prompt_clause_for(description)
            if vb_clause:
                parts.append(vb_clause)

Step 5 — in generate_panel_images(), pass visual_bible:
    prompt = compose_panel_prompt(
        panel_script, project.story_idea, project.character_registry,
        project.story_dna, page_dna, panel_dna,
        prev_panel_ctx=prev_panel_ctx,
        page_anchor_ctx=page_anchor_ctx,
        visual_bible=getattr(project, 'visual_bible', None),   # <-- add this
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descriptor compression — concise, image-model-friendly appearance text.
# ---------------------------------------------------------------------------
# The SOFT descriptor line ("Mara: a short skirt and denim blue jeans with
# black leather gloves over her hands, hair loose, a worried expression") is the
# text the diffusion model reads. Image models parse comma-separated noun
# phrases ("short skirt, blue jeans, black leather gloves") more reliably than
# full prose, and the prose form burns tokens we would rather spend on DETAIL.
# This helper strips grammatical filler the image model does not need while
# preserving every word that carries visual or spatial meaning.
#
# It is deliberately CONSERVATIVE and only ever applied to the soft descriptor
# line — NEVER to the hard-lock "must still …" predicates (which the LLM
# refinement pass reads and which rely on natural grammar), and never to the
# panel action/description prose. So "gloves over her hands" (redundant tail)
# compresses away, but "gloves tucked into her belt" (meaningful placement) is
# kept, because only a short list of KNOWN-REDUNDANT tails are removed.

# Leading filler verbs/phrases that add nothing for an image model.
_DESC_LEAD_FILLER_RE = re.compile(
    r'^(?:she|he|they|it)?\s*(?:is|are|was|were)?\s*'
    r'(?:currently\s+)?(?:wearing|dressed in|clad in|adorned (?:with|in)|'
    r'attired in|garbed in|outfitted in|sporting|donning)\s+',
    re.IGNORECASE)

# Redundant possessive tails that restate where a garment obviously sits. These
# are safe to drop because they add no visual information a model would miss
# ("gloves over her hands", "boots on her feet", "hat on his head"). Placement
# phrases that ARE informative ("gloves tucked into her belt", "scarf wound
# twice around the neck") are NOT in this list and are preserved verbatim.
_DESC_REDUNDANT_TAIL_RE = re.compile(
    r'\s+(?:over|on|upon)\s+(?:her|his|their|its)\s+'
    r'(?:hands?|feet|head|face|body|frame|form|shoulders?|legs?|arms?|torso)\b'
    r'(?=\s*(?:,|;|and\b|$))',
    re.IGNORECASE)

# Filler connectors between garments -> a plain comma reads better for images.
_DESC_AND_CONNECTOR_RE = re.compile(r'\s+(?:along )?with\s+|\s+and\s+', re.IGNORECASE)
# Only strip an article that BEGINS a garment phrase: at the very start of the
# string or immediately after a comma/semicolon. This preserves articles that
# live inside a placement phrase ("around the neck", "into the belt"), which
# read naturally and carry the phrase's grammar.
_DESC_LEADING_ARTICLE_RE = re.compile(r'(^|,\s*|;\s*)(?:a|an|the)\s+', re.IGNORECASE)
_DESC_MULTISPACE_RE = re.compile(r'\s{2,}')
_DESC_MULTICOMMA_RE = re.compile(r'\s*,\s*(?:,\s*)+')


def _compress_descriptor(text: str, drop_articles: bool = True) -> str:
    """Compress a soft appearance descriptor into concise, comma-separated,
    image-model-friendly phrasing WITHOUT losing visual detail.

    "she is wearing a short skirt and denim blue jeans with black leather gloves
    over her hands" -> "short skirt, denim blue jeans, black leather gloves".

    Conservative by construction: only strips a fixed set of known-redundant
    lead-ins, possessive tails, connectors, and (optionally) articles. Anything
    it does not recognise is passed through unchanged, so meaningful material,
    colour, count, and placement detail always survives. Idempotent.
    """
    if not text:
        return text
    t = text.strip()
    # Strip a leading "she is wearing / dressed in / …" BEFORE padding, so the
    # ^ anchor matches the real start of the string.
    t = _DESC_LEAD_FILLER_RE.sub('', t)
    t = ' ' + t + ' '                       # pad so boundary regexes fire cleanly
    # Remove redundant possessive tails BEFORE collapsing connectors, so the
    # "over her hands" in "gloves over her hands, and a hat" is caught.
    prev = None
    while prev != t:                        # apply repeatedly for chained tails
        prev = t
        t = _DESC_REDUNDANT_TAIL_RE.sub('', t)
    t = _DESC_AND_CONNECTOR_RE.sub(', ', t)
    t = _DESC_MULTICOMMA_RE.sub(', ', t)
    t = _DESC_MULTISPACE_RE.sub(' ', t)
    t = t.strip(' ,;')
    if drop_articles:
        # Drop only articles that begin a garment phrase (start-of-string or
        # right after a comma), NOT articles buried inside a placement phrase
        # like "around the neck" or "into the belt", which read naturally.
        prev = None
        while prev != t:                    # repeat: stripping one can expose next
            prev = t
            t = _DESC_LEADING_ARTICLE_RE.sub(r'\1', t)
        t = t.strip(' ,;')
    return t


# ---------------------------------------------------------------------------
# Lazy imports — the novel_generator utilities are available at runtime but
# we avoid a hard module-level dependency so this file can be imported for
# type-checking without the full ML stack present.
# ---------------------------------------------------------------------------

def _llm(prompt: str, temperature: float = 0.3, large: bool = True) -> str:
    """Thin wrapper around the project's LLM utility."""
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


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VisualBibleEntry:
    """One catalogued visual element."""
    label: str                        # short human-readable label, e.g. "diving mask"
    canonical: str                    # full unambiguous description for image gen
    aliases: List[str] = field(default_factory=list)   # shorthand terms to match
    owner: str = ""                   # character name, or "" for setting/shared prop
    category: str = "prop"           # prop | costume | equipment | environment

    def matches(self, text: str) -> bool:
        """True if any alias appears (case-insensitively, as whole word) in text."""
        lower = text.lower()
        for alias in self.aliases:
            pattern = r'\b' + re.escape(alias.lower()) + r'\b'
            if re.search(pattern, lower):
                return True
        return False

    def expand(self, text: str) -> Tuple[str, bool]:
        """
        Replace the FIRST alias occurrence in *text* with the canonical description.
        Returns (new_text, changed).
        """
        lower = text.lower()
        for alias in sorted(self.aliases, key=len, reverse=True):  # longest first
            pattern = re.compile(r'\b' + re.escape(alias.lower()) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                replacement = self.canonical
                new_text = pattern.sub(replacement, text, count=1)
                return new_text, new_text != text
        return text, False


class SeriesVisualBible:
    """
    Catalog of every persistent visual element in the series — props, costumes,
    equipment, recurring set-pieces — each with a canonical full description.

    Usage
    -----
    # Build once:
    vb = build_series_visual_bible(story_idea_str, characters, story_idea)

    # In panel-description expansion:
    cleaned, changed = vb.expand_description(panel_description)

    # In compose_panel_prompt:
    clause = vb.prompt_clause_for(panel_description)
    if clause:
        parts.append(clause)
    """

    def __init__(self):
        self.entries: List[VisualBibleEntry] = []

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def add_entry(self, entry: VisualBibleEntry):
        self.entries.append(entry)

    def add_from_dict(self, d: dict):
        """Populate from a dict returned by the LLM."""
        try:
            self.add_entry(VisualBibleEntry(
                label    = str(d.get("label", "")),
                canonical= str(d.get("canonical", "")),
                aliases  = [str(a).lower().strip() for a in d.get("aliases", []) if a],
                owner    = str(d.get("owner", "")),
                category = str(d.get("category", "prop")),
            ))
        except Exception as e:
            logger.warning(f"VisualBible: skipped bad entry ({e}): {d}")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def matching_entries(self, text: str) -> List[VisualBibleEntry]:
        """Return all entries whose aliases appear in *text*."""
        return [e for e in self.entries if e.matches(text)]

    def expand_description(self, text: str) -> Tuple[str, List[str]]:
        """
        Expand every alias occurrence in *text* to its canonical description.
        Returns (expanded_text, list_of_change_notes).
        """
        changes = []
        for entry in self.entries:
            new_text, changed = entry.expand(text)
            if changed:
                changes.append(
                    f"'{entry.label}': shorthand → canonical ({entry.canonical[:60]}…)"
                )
                text = new_text
        return text, changes

    def prompt_clause_for(self, description: str) -> str:
        """
        Build a compact image-prompt clause reinforcing all Visual Bible entries
        referenced in *description*.  Empty string if nothing matches.
        """
        hits = self.matching_entries(description)
        if not hits:
            return ""
        clauses = []
        for e in hits:
            clauses.append(f"[{e.label}: {e.canonical}]")
        return (
            "SERIES VISUAL BIBLE — draw these elements exactly as described: "
            + "; ".join(clauses) + "."
        )

    def summary(self) -> str:
        """Human-readable summary for logging."""
        lines = [f"SeriesVisualBible ({len(self.entries)} entries):"]
        for e in self.entries:
            owner_tag = f" [{e.owner}]" if e.owner else ""
            lines.append(
                f"  • {e.label}{owner_tag} ({e.category}): "
                f"{e.canonical[:80]}{'…' if len(e.canonical) > 80 else ''}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Coverage queries — used by the audit / backfill pass
    # ------------------------------------------------------------------

    def entries_for_owner(self, owner: str) -> List[VisualBibleEntry]:
        """All entries owned by a given character (case-insensitive match)."""
        o = (owner or "").strip().lower()
        if not o:
            return []
        return [e for e in self.entries if (e.owner or "").strip().lower() == o]

    def owner_has_category(self, owner: str, category: str) -> bool:
        """True if `owner` already has at least one entry in `category`."""
        cat = (category or "").strip().lower()
        return any(
            (e.category or "").strip().lower() == cat
            for e in self.entries_for_owner(owner)
        )

    def has_label(self, label: str) -> bool:
        """True if an entry with this label (case-insensitive) already exists."""
        l = (label or "").strip().lower()
        return any((e.label or "").strip().lower() == l for e in self.entries)

    def has_environment_for(self, location_text: str) -> bool:
        """True if any environment/setting entry's aliases or label match the
        given location text — used to detect uncovered named locations."""
        if not location_text:
            return False
        low = location_text.lower()
        for e in self.entries:
            if (e.category or "").strip().lower() != "environment":
                continue
            # label word-overlap or alias hit
            if e.matches(location_text):
                return True
            lbl = (e.label or "").strip().lower()
            if lbl and (lbl in low or low in lbl):
                return True
        return False

    def all_owners(self) -> set:
        """Set of distinct owners that have at least one entry."""
        return {(e.owner or "").strip() for e in self.entries if (e.owner or "").strip()}


# ---------------------------------------------------------------------------
# Factory: build the bible from the story concept
# ---------------------------------------------------------------------------

def ensure_distinct_appearances(characters: List, story_idea=None) -> int:
    """Make sure no two characters read as visual look-alikes.

    A common failure in AI comics is several characters sharing the same hair,
    build, and face — the reader can't tell the cast apart, which quietly breaks
    immersion and continuity. This pass scans the cast's appearance descriptors,
    detects collisions on the most identity-bearing visual axes (hair colour,
    hair length/style, build, and overall silhouette), and asks the LLM to
    re-differentiate ONLY the characters that clash — preserving each one's
    gender, age, and role while giving them a distinct, memorable look.

    Mutates each clashing Character's ``appearance`` in place. Returns the number
    of characters whose appearance was revised. Safe + idempotent: a cast that's
    already distinct triggers no LLM call.
    """
    # Need at least two same-gender characters to have a collision worth fixing.
    chars = [c for c in (characters or []) if getattr(c, 'appearance', '')]
    if len(chars) < 2:
        return 0

    # --- LLM-based collision detection ------------------------------------
    # Rather than extracting keyword signatures (which miss unusual vocabulary
    # and misfire on partial overlaps), ask the LLM to read the full appearance
    # strings and flag any characters a reader could confuse at a glance.
    # This catches clashes on colour, silhouette, and style that keyword lists
    # miss entirely.
    roster = []
    for c in chars:
        roster.append(
            f"- {c.name} ({getattr(c,'gender','') or 'unspecified'}, "
            f"age {getattr(c,'age','?') or '?'}): {c.appearance}"
        )
    genre = getattr(story_idea, 'genre', '') if story_idea else ''

    clash_prompt = (
        "You are a character designer reviewing a graphic-novel cast for visual "
        "distinctiveness. A reader must tell every character apart at a glance.\n\n"
        + ("GENRE: " + genre + "\n" if genre else "")
        + "CAST:\n" + "\n".join(roster) + "\n\n"
        + "List any characters whose appearances are so similar that a reader "
        "could confuse them — same gender AND broadly similar hair colour/style "
        "AND similar build/silhouette. Only flag pairs that would genuinely be "
        "hard to distinguish in black-outlined comic art.\n\n"
        "Return ONLY a JSON array of clashing character names (those that "
        "need redesigning), or [] if no clashes. JSON only, no prose."
    )
    clashing_names: List[str] = []
    try:
        raw_clash = _llm(clash_prompt, temperature=0.1, large=False)
        parsed_clash = _parse(raw_clash)
        if isinstance(parsed_clash, list):
            clashing_names = [str(n) for n in parsed_clash if n]
    except Exception as e:
        logger.warning(f"  [Distinctiveness] Collision detection failed ({e}); skipping.")
        return 0

    clashing = [c for c in chars if c.name in clashing_names]
    if not clashing:
        return 0

    logger.info(
        f"  [Distinctiveness] {len(clashing)} character(s) share a look; "
        f"re-differentiating..."
    )

    # Provide the FULL cast as context so new looks don't collide with anyone,
    # then ask the model to rewrite only the clashing characters.
    roster = []
    for c in chars:
        roster.append(
            f"- {c.name} ({getattr(c,'gender','') or 'unspecified'}, "
            f"age {getattr(c,'age','?')}): {c.appearance}"
        )
    to_fix = [c.name for c in clashing]
    genre = getattr(story_idea, 'genre', '') if story_idea else ''
    prompt = (
        "You are a character designer ensuring a graphic-novel cast is visually "
        "DISTINCT — a reader must tell everyone apart at a glance.\n\n"
        f"{'GENRE: ' + genre + chr(10) if genre else ''}"
        "FULL CAST (for reference — avoid clashing with ANY of these):\n"
        + "\n".join(roster) + "\n\n"
        "These characters currently look too similar to others and need a "
        f"distinct appearance: {', '.join(to_fix)}\n\n"
        "For EACH of them, write a NEW one-line appearance that keeps their "
        "gender, age, and role plausible but gives a clearly different and "
        "memorable look — vary hair colour/length/style, build/silhouette, "
        "skin tone, and 1-2 signature facial features or visual signatures so "
        "no two characters in the cast could be confused. Keep it concrete and "
        "renderable.\n\n"
        "Return a JSON object mapping each listed name to its new appearance "
        "line. JSON only."
    )
    try:
        parsed = _parse(_llm(prompt, temperature=0.7, large=True)) or {}
    except Exception as e:
        logger.warning(f"  [Distinctiveness] pass skipped ({e}).")
        return 0
    if not isinstance(parsed, dict):
        return 0

    revised = 0
    by_name = {c.name: c for c in chars}
    for name, new_appearance in parsed.items():
        c = by_name.get(name)
        if c and isinstance(new_appearance, str) and len(new_appearance.strip()) >= 8:
            c.appearance = new_appearance.strip()
            revised += 1
    logger.info(f"  [Distinctiveness] revised {revised} character appearance(s).")
    return revised


def build_series_visual_bible(
    story_idea_str: str,
    characters: list,
    story_idea,
    max_retries: int = 2,
) -> SeriesVisualBible:
    """
    Ask the LLM to catalogue every recurring visual element in the story and
    produce canonical image-generation descriptions for each one.

    Called once in synthesize_comic_story(), after the character registry is built.

    Returns a populated SeriesVisualBible.  On failure returns an empty bible
    (graceful degradation — the pipeline still works, just without disambiguation).
    """
    logger.info("[AD] Building Series Visual Bible...")

    # Build a compact character equipment summary for the LLM prompt.
    char_summaries = []
    for c in characters[:15]:  # cap for prompt size
        appearance = getattr(c, 'appearance', '') or ''
        char_summaries.append(f"  {c.name} ({c.role}): {appearance[:200]}")

    char_block = "\n".join(char_summaries) if char_summaries else "  (none)"

    genre  = getattr(story_idea, 'genre', '') or ''
    premise = getattr(story_idea, 'premise', '') or ''

    prompt = f"""You are the Art Director for a graphic novel series.
Your job: catalogue every recurring VISUAL ELEMENT that will appear across multiple panels
(props, costumes, equipment, recurring environment features, scientific/technical objects)
and write a CANONICAL, UNAMBIGUOUS image-generation description for each one.

THE PROBLEM YOU ARE SOLVING:
When a panel description says "the mask", an image model doesn't know if it's a diving mask,
a Halloween mask, a medical mask, or a gas mask.  Similarly, when it says "the reactor", the
model may draw a nuclear plant when the story means a chemistry flask; "the device" could be
a phone or a particle accelerator.  You must write descriptions so specific that no ambiguity
is possible.

For each element, include:
  - What TYPE of object it is (no generic words)
  - Distinctive colour, material, shape, markings
  - Structural/component details (number of parts, arrangement, proportions)
  - What it is NOT (if the word is commonly misread — e.g. "NOT a medical mask")
  - Who owns/wears it and how it looks when in use

SPECIAL INSTRUCTION — SCIENTIFIC AND TECHNICAL OBJECTS:
If the story involves any scientific, mechanical, biological, chemical, or astronomical objects,
you MUST catalogue them with structural precision.  Examples:
  - A "water molecule" must specify: one red oxygen sphere bonded to two white hydrogen spheres at 104.5°
  - A "microscope" must specify: binocular eyepiece, rotating objective turret, stage platform, arm
  - A "rocket" must specify: which rocket (Saturn V? Falcon 9?), number of stages/engines, fins shape
  - A "compass" must specify: circular housing, rotating needle (red tip = north), degree markings
  - "The machine" must be named and every key component listed
Do NOT catalogue abstract concepts like "hope" or "justice" — only physical, drawable objects.

STORY CONCEPT:
Genre: {genre}
Premise: {premise[:400]}

FULL STORY IDEA:
{story_idea_str[:1200]}

CHARACTER APPEARANCES:
{char_block}

INSTRUCTIONS:
1. Read the story carefully.
2. List EVERY recurring prop, costume piece, piece of equipment, distinctive environment feature,
   and technical/scientific object that will appear in multiple panels.
3. For each item write a canonical visual description (≤90 words) precise enough for an image
   model that has never read the story.
4. List ALL shorthand aliases that a writer might use to refer to it.

Return a JSON array — no markdown, no preamble:
[
  {{
    "label":     "short human label, e.g. 'diving mask'",
    "canonical": "full unambiguous visual description including shape, colour, material, part count",
    "aliases":   ["mask", "the mask", "her mask", "diving mask"],
    "owner":     "character name or empty string for shared/setting item",
    "category":  "prop | costume | equipment | environment | scientific | mechanical | biological"
  }},
  ...
]

Be thorough.  A science-fiction story might need entries for: the spaceship interior, the alien
weapon, the space suit, the control panel, the planet surface, AND any scientific equipment
or phenomena shown on screen.
"""

    vb = SeriesVisualBible()

    for attempt in range(max_retries + 1):
        try:
            raw = _llm(prompt, temperature=0.2)
            parsed = _parse(raw)
            if isinstance(parsed, list) and parsed:
                for item in parsed:
                    if isinstance(item, dict) and item.get("canonical"):
                        vb.add_from_dict(item)
                logger.info(
                    f"[AD] Visual Bible built: {len(vb.entries)} entries "
                    f"({'retried' if attempt > 0 else 'first attempt'})"
                )
                logger.info(vb.summary())
                return vb
            else:
                logger.warning(
                    f"[AD] Visual Bible parse returned unexpected type "
                    f"({type(parsed).__name__}), attempt {attempt+1}"
                )
        except Exception as e:
            logger.warning(f"[AD] Visual Bible build error (attempt {attempt+1}): {e}")

    logger.warning("[AD] Visual Bible build failed after retries; returning empty bible.")
    return vb


# ---------------------------------------------------------------------------
# Coverage audit + backfill — guarantees the Visual Bible isn't missing the
# costume of a main character, the description of a named location, or a
# signature prop. Run AFTER build_series_visual_bible (and after the script
# exists, so we can scan for named locations the LLM may have skipped).
# ---------------------------------------------------------------------------

def _collect_named_locations(script: List[Dict], cap: int = 25) -> List[str]:
    """Pull distinct setting strings from the script's pages/panels.

    Returns short location phrases (deduplicated, capped) so the audit can
    check whether each recurring location has an environment entry.
    """
    seen = {}
    for page in (script or []):
        if not isinstance(page, dict):
            continue
        # Skip structural / text pages — they have no real location.
        if (page.get('_act_break') or page.get('_prologue')
                or page.get('_narration_splash') or page.get('_act_interstitial')):
            continue
        setting = str(page.get('setting', '') or '').strip()
        if not setting:
            continue
        # Normalise to a short key for dedup (first ~8 words).
        key = ' '.join(setting.lower().split()[:8])
        if key and key not in seen:
            seen[key] = setting[:160]
        if len(seen) >= cap:
            break
    return list(seen.values())


def audit_and_backfill_visual_bible(
    visual_bible: 'SeriesVisualBible',
    characters: list,
    script: Optional[List[Dict]] = None,
    story_idea=None,
    story_idea_str: str = "",
    max_main_chars: int = 12,
) -> int:
    """Verify Visual Bible coverage and fill gaps with canonical descriptions.

    Three coverage checks:

      1. CHARACTER COSTUMES — every corporeal main/supporting character should
         have at least one 'costume' entry. A character whose face is locked by
         the appearance registry but whose CLOTHING is uncatalogued will drift
         between panels. We backfill a canonical costume for each missing one.

      2. SIGNATURE PROPS — characters whose appearance text names a held/worn
         item (bag, weapon, instrument, etc.) but who have no prop/equipment
         entry get one generated.

      3. NAMED LOCATIONS — recurring settings named in the script that have no
         'environment' entry get a canonical description so the place looks the
         same every time it appears.

    All gaps are sent to the LLM in ONE batched call (cheap) and the returned
    canonical descriptions are added to the bible. Returns the number of
    entries added. Degrades gracefully: on any failure the bible is left as-is.
    """
    if visual_bible is None:
        return 0

    # ── Identify corporeal characters worth cataloguing ──────────────────────
    def _is_corporeal(c) -> bool:
        # Non-human / abstract presences (light forms, forces, spirits) are
        # catalogued separately as scientific/environment entries by the main
        # builder; they don't wear costumes, so skip them here.
        for attr in ('is_non_human', 'non_human', 'is_abstract', 'incorporeal'):
            if getattr(c, attr, False):
                return False
        # Heuristic: a character with an explicit gender token is corporeal.
        return True

    corporeal = [c for c in (characters or []) if _is_corporeal(c)][:max_main_chars]

    # ── Build the list of gaps ───────────────────────────────────────────────
    costume_gaps = []   # characters missing a costume entry
    prop_hints   = []   # (character, appearance) where a prop is implied but missing
    for c in corporeal:
        name = getattr(c, 'name', '') or ''
        if not name:
            continue
        if not visual_bible.owner_has_category(name, 'costume'):
            costume_gaps.append(c)
        # Prop hint: appearance mentions a carryable/wearable item but the
        # character has no prop/equipment entry.
        appearance = (getattr(c, 'appearance', '') or '').lower()
        _PROP_WORDS = ('bag', 'backpack', 'satchel', 'briefcase', 'weapon',
                       'sword', 'gun', 'staff', 'wand', 'instrument', 'guitar',
                       'camera', 'phone', 'tablet', 'glasses', 'watch', 'cane',
                       'umbrella', 'book', 'journal', 'badge', 'amulet',
                       'necklace', 'ring', 'tool', 'toolbox', 'knife')
        if (any(w in appearance for w in _PROP_WORDS)
                and not visual_bible.owner_has_category(name, 'prop')
                and not visual_bible.owner_has_category(name, 'equipment')):
            prop_hints.append(c)

    location_gaps = []
    if script:
        for loc in _collect_named_locations(script):
            if not visual_bible.has_environment_for(loc):
                location_gaps.append(loc)

    if not costume_gaps and not prop_hints and not location_gaps:
        logger.info("[AD] Visual Bible coverage audit: no gaps found.")
        return 0

    logger.info(
        f"[AD] Visual Bible coverage audit: "
        f"{len(costume_gaps)} costume gap(s), {len(prop_hints)} prop gap(s), "
        f"{len(location_gaps)} location gap(s) — backfilling."
    )

    # ── Assemble the backfill request ────────────────────────────────────────
    genre = getattr(story_idea, 'genre', '') or ''
    premise = (getattr(story_idea, 'premise', '') or '')[:300]

    char_lines = []
    for c in costume_gaps:
        nm = getattr(c, 'name', '')
        ap = (getattr(c, 'appearance', '') or '')[:200]
        gd = getattr(c, 'gender', '') or ''
        rl = getattr(c, 'role', '') or ''
        char_lines.append(f'  - COSTUME for "{nm}" ({gd} {rl}): appearance — {ap}')
    for c in prop_hints:
        nm = getattr(c, 'name', '')
        ap = (getattr(c, 'appearance', '') or '')[:200]
        char_lines.append(f'  - SIGNATURE PROP for "{nm}": appearance — {ap}')
    loc_lines = [f'  - ENVIRONMENT for location: "{loc}"' for loc in location_gaps]

    request_block = "\n".join(char_lines + loc_lines)

    prompt = f"""You are the Art Director for a graphic novel. The Series Visual Bible
is MISSING canonical descriptions for the elements listed below. Write a precise,
unambiguous, image-generation description for EACH one so it is drawn identically
every time it appears.

STORY CONTEXT:
Genre: {genre}
Premise: {premise}
{('Full concept: ' + story_idea_str[:600]) if story_idea_str else ''}

MISSING ELEMENTS TO DESCRIBE:
{request_block}

For each element write a canonical visual description (≤80 words):
  - COSTUME: the character's default outfit — every garment, colour, material,
    fit, distinctive detail (head to foot). What they wear in most panels.
  - SIGNATURE PROP: the specific item — type, colour, material, shape, markings,
    how it is carried/worn.
  - ENVIRONMENT: the location — architecture, lighting, key furniture/features,
    colour palette, mood. Specific enough that the same place is recognisable
    across panels.

Return a JSON array — no markdown, no preamble:
[
  {{
    "label":     "short label, e.g. 'Maya everyday outfit' or 'the server room'",
    "canonical": "full unambiguous visual description",
    "aliases":   ["words a writer might use to refer to this"],
    "owner":     "character name for costumes/props, empty string for locations",
    "category":  "costume | prop | equipment | environment"
  }}
]
Provide exactly one entry per missing element listed above."""

    try:
        raw = _llm(prompt, temperature=0.3)
        parsed = _parse(raw)
        if not isinstance(parsed, list):
            logger.warning("[AD] Backfill returned non-list; skipping.")
            return 0
        added = 0
        for item in parsed:
            if not isinstance(item, dict) or not item.get('canonical'):
                continue
            label = str(item.get('label', '')).strip()
            # Avoid duplicating an existing label.
            if label and visual_bible.has_label(label):
                continue
            visual_bible.add_from_dict(item)
            added += 1
        if added:
            logger.info(f"[AD] Visual Bible backfill: added {added} canonical entry(ies).")
            # Re-log the (now more complete) bible for transparency.
            logger.info(visual_bible.summary())
        else:
            logger.info("[AD] Visual Bible backfill: no new entries added.")
        return added
    except Exception as e:
        logger.warning(f"[AD] Visual Bible backfill failed ({e}); leaving bible as-is.")
        return 0


# ---------------------------------------------------------------------------
# Art Director review pass — run after _validate_and_repair_panels
# ---------------------------------------------------------------------------

_BATCH_SIZE = 8   # panels per LLM call during review


def art_director_review_panels(
    script: List[Dict],
    visual_bible: SeriesVisualBible,
    story_idea,
    characters: list,
    is_adult_content: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """
    Two-stage art direction pass over every panel description.

    Stage 1 — Rule-based alias expansion (fast, free, zero API calls):
      Finds every Visual Bible alias in the description and replaces it with
      the canonical text.  Handles 80%+ of cases with no LLM cost.

    Stage 2 — LLM review of panels that Stage 1 touched OR that contain
      ambiguity-prone words not yet in the bible.  The LLM is given the
      Visual Bible as context and asked to rewrite the description so it
      fully specifies every visual element — no shortcuts, no pronouns for
      props, no context-dependent references.

    Parameters
    ----------
    script           : The full comic script (list of page dicts with panels).
    visual_bible     : The SeriesVisualBible built for this project.
    story_idea       : The StoryIdea for this project (provides genre/premise).
    characters       : The cast list.
    is_adult_content : When True, the LLM reviewer is told that this panel
                       contains adult content and should preserve mature/explicit
                       details rather than softening them.  When False (default),
                       the reviewer is instructed to keep all descriptions
                       non-explicit and suitable for a general audience, rewriting
                       any explicit content it encounters.

    Returns
    -------
    (corrected_script, change_log)
      corrected_script : same structure as input, descriptions updated in-place
      change_log       : list of human-readable correction notes for logging/debug
    """
    if not visual_bible or not visual_bible.entries:
        logger.info("[AD] Visual Bible is empty; skipping art director review.")
        return script, []

    logger.info(
        f"[AD] Art Director review: {sum(len(p.get('panels', [])) for p in script)} panels, "
        f"{len(visual_bible.entries)} visual bible entries..."
    )

    # Build a compact bible string for the LLM prompt (used in Stage 2)
    bible_lines = []
    for e in visual_bible.entries:
        owner_tag = f" [{e.owner}]" if e.owner else ""
        bible_lines.append(
            f'  "{e.label}"{owner_tag} → {e.canonical}'
        )
    bible_block = "\n".join(bible_lines)

    genre   = getattr(story_idea, 'genre', '') or ''
    premise = getattr(story_idea, 'premise', '') or ''[:300]

    change_log: List[str] = []
    panels_needing_llm: List[Tuple[dict, dict, int, int]] = []  # (panel, page, pi, pj)

    # -----------------------------------------------------------------------
    # Stage 1: rule-based expansion
    # -----------------------------------------------------------------------
    for pi, page in enumerate(script):
        for pj, panel in enumerate(page.get("panels", []) or []):
            desc = (panel.get("description") or "").strip()
            if not desc:
                continue
            expanded, changes = visual_bible.expand_description(desc)
            if changes:
                panel["description"] = expanded
                page_num = page.get("page", pi + 1)
                panel_num = panel.get("panel_index", pj + 1)
                for c in changes:
                    note = f"  p{page_num} panel {panel_num}: {c}"
                    change_log.append(note)
                    logger.info(f"[AD Stage1]{note}")
                # Flag for Stage 2 LLM refinement — Stage 1 does simple substitution,
                # the LLM will integrate the expanded text more naturally.
                panels_needing_llm.append((panel, page, pi, pj))
            else:
                # Also flag panels that contain ambiguity-prone words even if no
                # alias matched — the LLM can catch implicit context-dependence.
                if _has_ambiguous_references(desc, visual_bible):
                    panels_needing_llm.append((panel, page, pi, pj))

    logger.info(
        f"[AD] Stage 1 complete. {len(change_log)} expansions. "
        f"{len(panels_needing_llm)} panels queued for LLM review."
    )

    # -----------------------------------------------------------------------
    # Stage 2: LLM review of flagged panels in batches
    # -----------------------------------------------------------------------
    if panels_needing_llm:
        _llm_review_batches(
            panels_needing_llm, bible_block, genre, premise, change_log,
            is_adult_content=is_adult_content,
        )

    total_changes = len(change_log)
    logger.info(
        f"[AD] Art Director review complete. "
        f"{total_changes} correction(s) applied across "
        f"{sum(len(p.get('panels', [])) for p in script)} panels."
    )
    return script, change_log


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Words that are context-dependent and likely to confuse image models when
# used without a full qualifier.
_AMBIGUITY_SIGNALS = [
    r"\bthe mask\b", r"\bher mask\b", r"\bhis mask\b", r"\bthe device\b",
    r"\bthe suit\b", r"\bher suit\b", r"\bhis suit\b", r"\bthe outfit\b",
    r"\bthe gear\b", r"\bthe equipment\b", r"\bthe weapon\b",
    r"\bthe tool\b", r"\bthe unit\b", r"\bthe vehicle\b",
    r"\bthe machine\b", r"\bthe module\b", r"\bthe object\b",
    r"\bit\b",      # pronoun standing in for a prop — always risky in diffusion prompts
    # Scientific / technical shorthand that models misread without context
    r"\bthe compound\b", r"\bthe substance\b", r"\bthe element\b",
    r"\bthe specimen\b", r"\bthe sample\b", r"\bthe formula\b",
    r"\bthe reactor\b", r"\bthe chamber\b", r"\bthe container\b",
    r"\bthe crystal\b", r"\bthe structure\b", r"\bthe model\b",
    r"\bthe instrument\b", r"\bthe apparatus\b", r"\bthe mechanism\b",
    r"\bthe scanner\b", r"\bthe detector\b", r"\bthe sensor\b",
    r"\bthe probe\b", r"\bthe signal\b", r"\bthe system\b",
    r"\bthe creature\b", r"\bthe organism\b", r"\bthe entity\b",
    r"\bthe construct\b", r"\bthe artifact\b",
]

_AMBIGUITY_RE = [re.compile(p, re.IGNORECASE) for p in _AMBIGUITY_SIGNALS]


def _has_ambiguous_references(text: str, visual_bible: SeriesVisualBible) -> bool:
    """
    True if the text contains ambiguity-prone shorthand that the Visual Bible
    has entries for (so the LLM might be able to fix it even if Stage 1 missed it).
    """
    if not text:
        return False
    if not visual_bible.entries:
        return False
    for pattern in _AMBIGUITY_RE:
        if pattern.search(text):
            return True
    return False


def _llm_review_batches(
    panels_needing_llm: list,
    bible_block: str,
    genre: str,
    premise: str,
    change_log: List[str],
    is_adult_content: bool = False,
):
    """Send batches of panel descriptions to the LLM for Art Director review."""
    for batch_start in range(0, len(panels_needing_llm), _BATCH_SIZE):
        batch = panels_needing_llm[batch_start: batch_start + _BATCH_SIZE]
        _llm_review_batch(batch, bible_block, genre, premise, change_log,
                          is_adult_content=is_adult_content)


def _llm_review_batch(
    batch: list,
    bible_block: str,
    genre: str,
    premise: str,
    change_log: List[str],
    is_adult_content: bool = False,
):
    """One LLM call covering up to _BATCH_SIZE panels."""
    # Build the panel list for the prompt
    panel_items = []
    for panel, page, pi, pj in batch:
        page_num  = page.get("page", pi + 1)
        panel_num = panel.get("panel_index", pj + 1)
        desc      = (panel.get("description") or "").strip()
        setting   = (panel.get("setting") or "").strip()
        chars     = ", ".join(panel.get("characters_in_frame") or [])
        panel_items.append(
            f'{{"id": "p{page_num}_panel{panel_num}", '
            f'"description": {_json_str(desc)}, '
            f'"setting": {_json_str(setting)}, '
            f'"characters_in_frame": {_json_str(chars)}}}'
        )

    panels_block = "\n".join(panel_items)

    # Build the content-appropriate instruction for item 5 in the prompt.
    content_instruction = (
        "5. CONTENT: This panel contains NSFW adult content. Enhance all descriptions of the female figure "
        "with vivid, imaginative, creative, and beautiful detailed depictions of their naked bodies: "
        "breasts, nipples, lips, thighs, arms, legs, eyes, pose, etc. "
        "We want to enhance the emotion, passion, eroticism, sexuality, sensuality, etc. for adults to enjoy the emotion and drama.\n"
        if is_adult_content else
        "5. VISUAL QUALITY: Beyond fixing ambiguity, actively enrich any description "
        "that is sparse or generic. For each panel, consider:\n"
        "   • TEXTURE & MATERIAL: name the specific material quality of every "
        "surface — not 'rough stone wall' but 'rain-darkened limestone with mineral "
        "streaks and a lichen crust at the base'; not 'old jacket' but 'worn canvas "
        "field jacket, elbows reinforced with darker fabric, collar turned up'. "
        "Surfaces to consider: aged leather cracked at the creases, oxidised copper "
        "with patina blooms, frosted glass diffusing shapes behind it, wet asphalt "
        "reflecting light in broken shards, tarnished brass fittings, hand-stitched "
        "linen gone soft with wear, matte ceramic glaze, burnished iron.\n"
        "   • LIGHT QUALITY: name the exact quality, direction, and colour temperature "
        "of every light source — not 'dramatic lighting' but 'a single overhead "
        "fluorescent strip casting a cold downward cone, leaving the lower face in "
        "shadow'. Techniques to draw from: caustic light rippling through shallow "
        "water, volumetric shafts cutting through dust or smoke, golden-hour raking "
        "light that stretches every shadow long, sodium street-lamp orange on wet "
        "stone, ember glow from below, rim light that separates a dark figure from "
        "a dark background, the blue ghost-light of a screen in a dark room, "
        "chiaroscuro with a single candle as the only warm point in a cold space.\n"
        "   • HOW TO APPLY: do not list these as labels. Weave them into the "
        "description as if writing a camera direction — 'the fluorescent strip "
        "above bleaches the table surface white and deepens the hollows of her "
        "eyes into shadow.' One well-chosen texture and one named light source "
        "transforms a flat description into a drawable scene.\n"
        "   • NARRATIVE SPECIFICITY: every prop should belong to THIS story moment. "
        "A detective's desk has a specific cold coffee, a specific case file with "
        "tabs, a specific map with red pins — not 'a cluttered desk'. A character's "
        "clothing carries wear patterns that reveal history. Small telling props "
        "make images feel inhabited, not staged.\n"
    )

    prompt = f"""You are the Art Director for a graphic novel series.
Your task: rewrite panel descriptions so every visual element is fully specified —
no shorthand, no pronouns standing in for props, no context-dependent references.

An image generation model will receive each description in isolation, with NO other
context about the story.  It must be able to draw the panel correctly from the
description alone.

SERIES VISUAL BIBLE (authoritative descriptions — always use these, never abbreviate):
{bible_block}

STORY CONTEXT (for reference only — do NOT add story information to descriptions):
Genre: {genre}
Premise: {premise}

PANELS TO REVIEW:
{panels_block}

INSTRUCTIONS:
For each panel, check its description for:
1. Shorthand references to Visual Bible items (e.g. "the mask" when the bible defines
   a specific diving mask) — replace with the canonical description from the bible.
2. Pronouns used for props ("it", "them", "the device", "the structure") — replace
   with the specific object name + key visual details.
3. Any other term that an image model could misinterpret given the genre.
4. SCIENTIFIC / TECHNICAL OBJECTS: If the description mentions a molecule, chemical
   formula, anatomical structure, mechanical device, astronomical object, or any object
   with a well-defined real-world structure, expand it with precise visual specifications:
   - molecule → specify atom types, colours, bond arrangement, geometry
   - skeleton → specify which bones, their positions, anatomical accuracy
   - gears → specify tooth shape, meshing, size ratios
   - map → specify projection, colour coding, labeled features
   - diagram → specify what it shows, key components, layout
   Never invent scientific details that contradict the story; derive them from
   real-world accuracy for the named object.
{content_instruction}
Rules:
- Keep the description concise (1-4 sentences).  Do NOT pad with narrative.
- Do NOT add story events, emotions, or character backstory.
- Do NOT change the shot composition, action, or characters in frame.
- If a description is already unambiguous and technically accurate, return it unchanged.

Return ONLY a JSON array — no markdown, no preamble:
[
  {{"id": "p3_panel2", "description": "rewritten description here"}},
  ...
]
One entry per panel.  IDs must match exactly.
"""

    try:
        raw = _llm(prompt, temperature=0.2)
        parsed = _parse(raw)
        if not isinstance(parsed, list):
            logger.warning(f"[AD Stage2] Unexpected parse type: {type(parsed)}")
            return

        # Index corrections by id
        corrections: Dict[str, str] = {}
        for item in parsed:
            if isinstance(item, dict) and item.get("id") and item.get("description"):
                corrections[item["id"]] = item["description"].strip()

        # Apply corrections back to the panel dicts
        for panel, page, pi, pj in batch:
            page_num  = page.get("page", pi + 1)
            panel_num = panel.get("panel_index", pj + 1)
            item_id   = f"p{page_num}_panel{panel_num}"
            new_desc  = corrections.get(item_id, "")
            if new_desc and new_desc != (panel.get("description") or "").strip():
                old_preview = (panel.get("description") or "")[:60]
                panel["description"] = new_desc
                note = (
                    f"  p{page_num} panel {panel_num} [LLM rewrite]: "
                    f"'{old_preview}…' → '{new_desc[:60]}…'"
                )
                change_log.append(note)
                logger.info(f"[AD Stage2]{note}")

    except Exception as e:
        logger.warning(f"[AD Stage2] LLM review batch failed: {e}")


def _json_str(s: str) -> str:
    """Minimal JSON string serialization (avoids importing json for a single call)."""
    import json
    return json.dumps(s)


# ---------------------------------------------------------------------------
# SceneConsistencyAnchor — pinned ground truth for a scene
# ---------------------------------------------------------------------------
# Problem it solves
# -----------------
# Diffusion models generate each panel independently.  Without an explicit
# per-scene costume and environment manifest, the model re-interprets
# character descriptions on every call and can silently drift: a character's
# jacket changes colour, background furniture moves, or — most visibly — a
# character's gender changes because the pose/action text in the new panel
# description activates a different internal prior than the appearance text.
#
# SceneConsistencyAnchor captures the ground truth from the FIRST panel of a
# scene (the establishing shot) and re-injects it as a hard constraint into
# every subsequent panel prompt in that scene.
#
# Integration (comic_book_generator.py — generate_panel_images)
# --------------------------------------------------------------
# Reset at each new scene (detected by setting-string change or page-break
# when a new scene begins):
#
#   scene_anchor: Optional[SceneConsistencyAnchor] = None
#   current_scene_setting: str = ""
#
# Build from the first panel of the scene:
#
#   if scene_anchor is None:
#       scene_anchor = build_scene_consistency_anchor(
#           panel_script, characters, visual_bible
#       )
#       current_scene_setting = (panel_script.get('setting') or '').strip()
#
# Detect scene change (reset anchor):
#
#   panel_setting = (panel_script.get('setting') or '').strip()
#   if panel_setting and panel_setting != current_scene_setting:
#       scene_anchor = build_scene_consistency_anchor(
#           panel_script, characters, visual_bible
#       )
#       current_scene_setting = panel_setting
#
# Pass to compose_panel_prompt:
#
#   prompt = compose_panel_prompt(
#       panel_script, ...,
#       scene_consistency_ctx=scene_anchor,
#   )
# ---------------------------------------------------------------------------

@dataclass
class SceneConsistencyAnchor:
    """Ground-truth visual manifest for a scene, extracted from its first panel.

    Injected into every subsequent panel prompt in the same scene so that
    setting, lighting, and character costumes are held constant regardless of
    what the panel description says (or omits).

    Fields
    ------
    setting_description : Full setting/environment text from the establishing shot.
    lighting            : Lighting conditions (direction, colour, intensity).
    time_of_day         : Day / night / dusk / etc.
    atmosphere          : Mood-relevant atmospheric details (fog, rain, dust, etc.).
    character_outfits   : {character_name: detailed_outfit_description}
                          Extracted per character from the first panel description
                          and/or the character registry.
    environment_details : Any specific props, furniture, architecture details that
                          must persist across all panels in this scene.
    """
    setting_description: str = ""
    lighting: str = ""
    time_of_day: str = ""
    atmosphere: str = ""
    character_outfits: Dict[str, str] = field(default_factory=dict)
    character_genders: Dict[str, str] = field(default_factory=dict)  # name → gender token
    environment_details: str = ""
    # Tracked screen positions for the 180° rule.  Populated from the
    # establishing panel once two or more characters share the frame.
    # Values: "left" | "right" | "center".  Empty dict = not yet resolved.
    screen_positions: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def prompt_clause(self, characters_in_frame: List[str],
                       include_costume: bool = True) -> str:
        """Build a hard consistency constraint string for a panel prompt.

        The returned string should be appended to the diffusion model prompt
        AFTER the panel's own description so it acts as a correction layer on
        any ambiguities introduced by the description.

        Parameters
        ----------
        characters_in_frame : Names of characters visible in this panel.
                              Only their outfits are injected (avoids bloating
                              the prompt with off-screen characters).
        include_costume     : When False, the per-character COSTUME LOCK is
                              omitted. Use this when the AppearanceContinuity
                              tracker is authoritative for costume — the scene
                              anchor then provides ONLY the setting / lighting /
                              environment locks (which are genuinely static per
                              scene), and the tracker supplies the evolving
                              costume/condition lock separately.
        """
        if not self._has_content():
            return ""

        parts: List[str] = []

        # Environment description — must come before character details so the
        # model interprets body positions relative to the correct space.
        # Phrased as plain scene description (no meta-labels like "SCENE LOCK"),
        # because the image model renders the words it's given; instructional
        # tags add nothing visual and can be rendered as literal text.
        if self.setting_description:
            parts.append(f"Location: {self.setting_description}.")
        env_fragments = []
        if self.lighting:
            env_fragments.append(f"lighting: {self.lighting}")
        if self.time_of_day:
            env_fragments.append(f"time of day: {self.time_of_day}")
        if self.atmosphere:
            env_fragments.append(f"atmosphere: {self.atmosphere}")
        if env_fragments:
            parts.append("; ".join(env_fragments).capitalize() + ".")
        if self.environment_details:
            parts.append(f"Scene details: {self.environment_details}.")

        # Per-character costume — only for characters visible in this panel.
        # Bind each outfit to a gender-based subject phrase ("the woman", "the man")
        # rather than the proper name. Proper names risk being rendered as visible
        # text in the image. Disambiguate same-gender pairs with an ordinal.
        #
        # Skipped entirely when include_costume=False — in that mode the
        # AppearanceContinuityTracker owns the costume/condition description and
        # the scene anchor contributes only the static environment description.
        if include_costume:
            visible = [c for c in characters_in_frame if self.character_outfits.get(c)]
            # Count genders among visible characters for ordinal disambiguation.
            gender_counts: Dict[str, int] = {}
            for char in visible:
                g = self.character_genders.get(char, 'figure')
                gender_counts[g] = gender_counts.get(g, 0) + 1
            _ordinals = ['first', 'second', 'third']
            gender_seen: Dict[str, int] = {}
            outfit_parts: List[str] = []
            for char in visible:
                outfit = self.character_outfits.get(char, "")
                g = self.character_genders.get(char, 'figure')
                if gender_counts.get(g, 0) > 1:
                    idx = gender_seen.get(g, 0)
                    gender_seen[g] = idx + 1
                    ord_word = _ordinals[idx] if idx < len(_ordinals) else f"{idx+1}th"
                    subject = f"the {ord_word} {g}"
                else:
                    subject = f"the {g}"
                outfit_parts.append(f"{subject} wears {outfit}")
            if outfit_parts:
                parts.append("; ".join(outfit_parts).capitalize() + ".")

        # Screen-direction lock (180° rule) — only when positions were resolved
        # from the establishing panel and at least two characters are visible.
        # Injecting specific positional language ("Elena stands screen-left;
        # Aaron stands screen-right") is far more actionable than the generic
        # soft nudge, and prevents the most common jarring continuity break.
        if self.screen_positions and len(characters_in_frame) >= 2:
            pos_frags = []
            for char in characters_in_frame:
                pos = self.screen_positions.get(char, '')
                g = self.character_genders.get(char, 'figure')
                if pos:
                    pos_frags.append(f"the {g} stands screen-{pos}")
            if pos_frags:
                parts.append(
                    "Screen direction (hold exactly): "
                    + "; ".join(pos_frags) + "."
                )

        return " ".join(parts)

    def _has_content(self) -> bool:
        return bool(
            self.setting_description or self.character_outfits
            or self.lighting or self.environment_details
        )


# ---------------------------------------------------------------------------
# Factory: build a SceneConsistencyAnchor from the first panel of a scene
# ---------------------------------------------------------------------------

# A scene anchor is re-injected into EVERY panel of its scene, so each stored
# field must be SHORT — just the relevant ground truth, not an essay. The
# extraction prompt asks for 1-2 sentences / 30-50 words, but a misbehaving LLM
# can return a whole paragraph (or, in the worst case seen in production, a
# multi-thousand-word blob), which then multiplies across every panel and blows
# the prompt past the model's token ceiling. These caps bound each field at the
# source so only the relevant information is carried forward. They are word
# caps (not token budgets): readable, transition-preserving, and generous enough
# that a normal "blue sedan, now with a half-finished red repaint on the driver
# door" easily fits, while a runaway response is trimmed on a sentence/word
# boundary.
_ANCHOR_FIELD_WORD_CAPS = {
    "setting_description": 60,
    "lighting": 30,
    "time_of_day": 12,
    "atmosphere": 30,
    "environment_details": 50,
    "character_outfit": 100,
}


def _cap_anchor_field(text: str, field: str) -> str:
    """Trim an extracted anchor field to its word cap on a clean boundary.

    Prefers to end at a sentence break within the cap; otherwise cuts at the
    word cap. Returns the original text unchanged when it is already short.
    """
    if not text:
        return text or ""
    text = str(text).strip()
    cap = _ANCHOR_FIELD_WORD_CAPS.get(field, 60)
    words = text.split()
    if len(words) <= cap:
        return text
    head = " ".join(words[:cap])
    # Back off to the last sentence end inside the kept span, if there is one
    # reasonably close to the cap, so we don't end mid-clause.
    last_dot = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if last_dot >= len(head) * 0.5:
        head = head[: last_dot + 1]
    logger.warning(
        "[SceneAnchor] '%s' field was %d words; capped to ~%d "
        "(the extraction LLM over-ran the requested length).",
        field, len(words), cap,
    )
    return head.rstrip()


def _anchor_cache_key(panel_script: Dict, chars_in_frame: List[str],
                      registry) -> str:
    """A stable key over the TRUE determinants of a scene anchor: the
    establishing panel's description + setting + sensory anchor, the cast, and
    each cast member's registry outfit baseline. Two scenes that share all of
    these produce an identical anchor, so the LLM extraction can be reused.

    Keyed on content (not just the setting string) so returning to the same
    room with a DIFFERENT establishing action correctly re-extracts, while a
    genuine duplicate — including the same scene seen again in the image-gen
    pass after the prompt-only pass — hits the cache and skips the LLM call.
    """
    import hashlib
    desc = (panel_script.get('description') or '').strip()
    setting = (panel_script.get('setting') or '').strip()
    sensory = (panel_script.get('sensory_anchor') or '').strip()
    cast = sorted(chars_in_frame)
    outfit_sig = []
    if registry is not None and hasattr(registry, 'get_clothing'):
        for name in cast:
            outfit_sig.append(f"{name}={registry.get_clothing(name) or ''}")
    # The anchor now defers to the tracker's guarded CURRENT clothing
    # (panel['_appearance_state']), which can differ from the base outfit as the
    # story evolves. Include that tracked clothing in the key so two otherwise
    # identical establishing panels with DIFFERENT current outfits don't collide
    # on one cached anchor (which would reintroduce the very flicker we fix).
    appearance_sig = []
    _astate = panel_script.get('_appearance_state') or {}
    if isinstance(_astate, dict):
        for name in sorted(_astate.keys()):
            entry = _astate.get(name)
            if isinstance(entry, dict):
                appearance_sig.append(f"{name}={entry.get('clothing', '') or ''}")
    blob = "\u241f".join([desc, setting, sensory, "|".join(cast),
                          "|".join(outfit_sig), "|".join(appearance_sig)])
    return hashlib.sha1(blob.encode('utf-8', 'ignore')).hexdigest()


# Module-level cache: anchor-key -> SceneConsistencyAnchor. Bounded so a very
# long book cannot grow it without limit; scenes recur, so even a modest cap
# captures most of the reuse. Cleared between books via reset_scene_anchor_cache().
_SCENE_ANCHOR_CACHE: Dict[str, 'SceneConsistencyAnchor'] = {}
_SCENE_ANCHOR_CACHE_MAX = 512
_SCENE_ANCHOR_CACHE_STATS = {'hits': 0, 'misses': 0}


def reset_scene_anchor_cache() -> None:
    """Clear the scene-anchor cache and stats. Call at the start of a new book
    so anchors from a previous project cannot leak in."""
    _SCENE_ANCHOR_CACHE.clear()
    _SCENE_ANCHOR_CACHE_STATS['hits'] = 0
    _SCENE_ANCHOR_CACHE_STATS['misses'] = 0


def _store_scene_anchor(key: str, anchor: 'SceneConsistencyAnchor') -> None:
    """Store an anchor under a content key, evicting the oldest entry when the
    cache is full (simple FIFO — recurrence patterns make LRU vs FIFO a wash
    here, and FIFO avoids the per-hit bookkeeping)."""
    if key in _SCENE_ANCHOR_CACHE:
        return
    if len(_SCENE_ANCHOR_CACHE) >= _SCENE_ANCHOR_CACHE_MAX:
        try:
            oldest = next(iter(_SCENE_ANCHOR_CACHE))
            del _SCENE_ANCHOR_CACHE[oldest]
        except StopIteration:
            pass
    _SCENE_ANCHOR_CACHE[key] = anchor


def build_scene_consistency_anchor(
    panel_script: Dict,
    characters: list,
    registry=None,           # CharacterAppearanceRegistry (optional but recommended)
    visual_bible: Optional['SeriesVisualBible'] = None,
    max_retries: int = 1,
    use_cache: bool = True,
) -> SceneConsistencyAnchor:
    """Build a SceneConsistencyAnchor from a scene's establishing panel.

    Called once per scene (on the first panel).  The anchor is then passed to
    ``compose_panel_prompt`` for every subsequent panel in the scene via the
    ``scene_consistency_ctx`` parameter.

    Strategy
    --------
    1. Use the character registry's ``get_clothing()`` method (if available) to
       seed the per-character costume dict — this is the most reliable source
       because it was generated from the full character profile, not just the
       panel description.
    2. Ask the LLM to extract any outfit overrides present in THIS panel's
       description (e.g. "she puts on a rain jacket") and merge them on top.
    3. Extract setting, lighting, and environment details from the panel.

    The LLM call is optional — if it fails the anchor is still useful because
    the registry-seeded costume data is already populated.
    """
    anchor = SceneConsistencyAnchor()

    # --- Seed from the character registry (most reliable source) ---
    chars_in_frame = panel_script.get('characters_in_frame', []) or []
    # Normalise entries — same logic as compose_panel_prompt
    def _to_name(entry) -> str:
        if isinstance(entry, dict):
            return (entry.get('name') or entry.get('character') or
                    entry.get('character_name') or '')
        return str(entry) if entry is not None else ''

    chars_in_frame = [_to_name(c) for c in chars_in_frame if _to_name(c)]

    # --- Cache lookup: identical scene content reuses the extracted anchor ---
    # This is the main performance lever for the panel loop. A scene anchor is
    # deterministic given the establishing panel's content + cast + baseline
    # outfits, so once extracted it can be reused for every later occurrence —
    # crucially including the second (image-gen) pass over the same scenes and
    # any story that revisits the same location with the same framing.
    _cache_key = None
    if use_cache:
        try:
            _cache_key = _anchor_cache_key(panel_script, chars_in_frame, registry)
            cached = _SCENE_ANCHOR_CACHE.get(_cache_key)
            if cached is not None:
                _SCENE_ANCHOR_CACHE_STATS['hits'] += 1
                return cached
            _SCENE_ANCHOR_CACHE_STATS['misses'] += 1
        except Exception:
            _cache_key = None   # never let caching break the build

    if registry is not None:
        for name in chars_in_frame:
            clothing = ''
            if hasattr(registry, 'get_clothing'):
                clothing = registry.get_clothing(name) or ''
            if not clothing and hasattr(registry, 'get_appearance'):
                # Fall back to full appearance and let the LLM extract clothing
                clothing = registry.get_appearance(name) or ''
            if clothing:
                anchor.character_outfits[name] = clothing
            # Seed the gender token so the costume lock can bind by "the woman"
            # / "the man" instead of the proper name (which risks text rendering).
            if hasattr(registry, 'get_gender'):
                g = registry.get_gender(name) or ''
                if g:
                    anchor.character_genders[name] = g

    # --- Seed setting from the panel script ---
    setting = (panel_script.get('setting') or '').strip()
    if setting:
        anchor.setting_description = _cap_anchor_field(setting, 'setting_description')

    # --- LLM extraction pass (optional enrichment) ---
    description = (panel_script.get('description') or '').strip()
    if not description:
        if _cache_key is not None:
            _store_scene_anchor(_cache_key, anchor)
        return anchor   # nothing to extract; return registry-seeded anchor

    # Build character appearance block for the LLM prompt
    char_app_lines: List[str] = []
    if registry is not None:
        for name in chars_in_frame:
            locked = ''
            if hasattr(registry, 'get_locked_appearance'):
                locked = registry.get_locked_appearance(name) or ''
            elif hasattr(registry, 'get_appearance'):
                locked = registry.get_appearance(name) or ''
            if locked:
                char_app_lines.append(f"  {name}: {locked}")
    char_app_block = "\n".join(char_app_lines) if char_app_lines else "  (none)"

    # Prompt layout is deliberately STABLE-PREFIX-FIRST: the fixed instructions
    # and JSON schema (identical on every call) come first so the provider's
    # automatic prompt cache (OpenAI + Grok) can hit on the long shared prefix,
    # and only the short variable panel data changes at the end. This cuts input
    # token cost on every anchor extraction without altering the output.
    prompt = f"""You are the Continuity Supervisor for a graphic novel.
Your task: extract the precise visual ground truth from a scene's FIRST (establishing) panel.
This information will be locked and re-injected into EVERY subsequent panel in this scene
to prevent character costume drift, gender drift, and setting drift.

EXTRACT the following. If the panel description does not mention a field, use the registry data or leave blank.

Return ONLY a JSON object — no markdown, no preamble:
{{
  "setting_description": "full location description (1-2 sentences)",
  "lighting": "lighting direction, colour, intensity (1 sentence)",
  "time_of_day": "day | night | dawn | dusk | interior | etc.",
  "atmosphere": "weather, fog, dust, rain, or other atmospheric details (1 sentence or blank)",
  "environment_details": "specific persistent props, furniture, architecture that must stay consistent (1 sentence or blank)",
  "character_outfits": {{
    "CharacterName": "COMPLETE outfit: garment names + colours + materials + accessories, detailed but readable (60-90 words)",
    ...
  }},
  "screen_positions": {{
    "CharacterName": "left | right | center",
    ...
  }}
}}

For screen_positions: only populate when the description clearly places two or more
characters on opposite sides of the frame. "left" = screen-left from the viewer's
perspective. Leave the dict empty when positions are ambiguous or only one character
is present.

The CHARACTER REGISTRY is the authoritative baseline — override a character's outfit
ONLY if the panel description explicitly changes it.

=== SCENE DATA ===
FIRST PANEL DESCRIPTION:
{description}

SETTING (from script metadata):
{setting or '(not specified)'}

SENSORY ANCHOR:
{panel_script.get('sensory_anchor', '') or '(not specified)'}

CHARACTER REGISTRY (authoritative baseline):
{char_app_block}

CHARACTERS IN THIS PANEL:
{', '.join(chars_in_frame) if chars_in_frame else '(not listed)'}
"""

    for attempt in range(max_retries + 1):
        try:
            raw = _llm(prompt, temperature=0.1)
            parsed = _parse(raw)
            if not isinstance(parsed, dict):
                logger.warning(f"[SceneAnchor] Unexpected parse type on attempt {attempt+1}")
                continue

            # Merge LLM results on top of registry-seeded data (LLM overrides
            # registry only when the panel description explicitly specifies a change).
            if parsed.get('setting_description'):
                anchor.setting_description = _cap_anchor_field(
                    str(parsed['setting_description']).strip(), 'setting_description')
            if parsed.get('lighting'):
                anchor.lighting = _cap_anchor_field(
                    str(parsed['lighting']).strip(), 'lighting')
            if parsed.get('time_of_day'):
                anchor.time_of_day = _cap_anchor_field(
                    str(parsed['time_of_day']).strip(), 'time_of_day')
            if parsed.get('atmosphere'):
                anchor.atmosphere = _cap_anchor_field(
                    str(parsed['atmosphere']).strip(), 'atmosphere')
            if parsed.get('environment_details'):
                anchor.environment_details = _cap_anchor_field(
                    str(parsed['environment_details']).strip(), 'environment_details')

            outfits = parsed.get('character_outfits', {})
            if isinstance(outfits, dict):
                for char_name, outfit in outfits.items():
                    if isinstance(outfit, str) and outfit.strip():
                        # LLM result overrides registry seed (panel description wins)
                        anchor.character_outfits[char_name] = _cap_anchor_field(
                            outfit.strip(), 'character_outfit')

            # ── SINGLE SOURCE OF TRUTH RECONCILIATION ────────────────────────
            # The AppearanceContinuityTracker (run during synthesis) already
            # resolved each character's CURRENT clothing with its regression
            # guard and history, and stamped it onto the panel as
            # `_appearance_state`. That guarded value — not a fresh per-scene LLM
            # reading of the establishing description — is the authoritative
            # costume. Without this step the anchor is a parallel clothing
            # authority that can disagree with the tracker (e.g. a beach scene
            # whose establishing shots make the LLM read "shirt", then "dress",
            # then "shirt"), causing exactly the outfit flicker the tracker
            # exists to prevent. We therefore let the tracker's state OVERRIDE
            # the anchor's outfit whenever it is present, so both systems speak
            # with one voice and the regression guard actually governs the image.
            _appearance_state = panel_script.get('_appearance_state') or {}
            if isinstance(_appearance_state, dict):
                for char_name in list(anchor.character_outfits.keys()) + list(_appearance_state.keys()):
                    st_entry = _appearance_state.get(char_name)
                    if not isinstance(st_entry, dict):
                        # Try a case-insensitive / whitespace-tolerant match.
                        st_entry = next(
                            (v for k, v in _appearance_state.items()
                             if isinstance(v, dict)
                             and str(k).strip().lower() == str(char_name).strip().lower()),
                            None)
                    if isinstance(st_entry, dict):
                        tracked_clothing = str(st_entry.get('clothing', '') or '').strip()
                        if tracked_clothing:
                            anchor.character_outfits[char_name] = _cap_anchor_field(
                                tracked_clothing, 'character_outfit')

            # Screen positions for the 180° rule — only populated when the
            # LLM finds clear left/right placement of two or more characters.
            positions = parsed.get('screen_positions', {})
            _VALID_POS = {'left', 'right', 'center', 'centre'}
            if isinstance(positions, dict):
                for char_name, pos in positions.items():
                    pos_norm = str(pos or '').strip().lower().replace('centre', 'center')
                    if pos_norm in _VALID_POS and char_name:
                        anchor.screen_positions[char_name] = pos_norm

            logger.info(
                f"[SceneAnchor] Built for setting='{anchor.setting_description[:60]}', "
                f"chars={list(anchor.character_outfits.keys())}"
                + (f", screen_pos={anchor.screen_positions}" if anchor.screen_positions else "")
            )
            if _cache_key is not None:
                _store_scene_anchor(_cache_key, anchor)
            return anchor

        except Exception as e:
            logger.warning(f"[SceneAnchor] Build error (attempt {attempt+1}): {e}")

    # Return with whatever registry data was seeded, even if LLM failed.
    logger.warning("[SceneAnchor] LLM extraction failed; returning registry-seeded anchor.")
    if _cache_key is not None:
        _store_scene_anchor(_cache_key, anchor)
    return anchor


# ===========================================================================
# APPEARANCE CONTINUITY TRACKER — the evolving-state continuity engine
# ===========================================================================
# Problem it solves
# -----------------
# SceneConsistencyAnchor freezes the establishing-shot costume and re-injects
# it UNCHANGED into every panel of a scene. That is correct for the SETTING
# (a room does not rearrange itself between panels) but WRONG for character
# appearance, because stories deliberately change appearance as they flow:
#
#     panel 1: Yuki stands in her jacket
#     panel 2: she peels off the jacket, sweating
#     panel 3: she works in just a tank top
#     panel 4: she pulls a clean shirt over her head
#
# With a frozen anchor, panels 3 and 4 would force the original jacket back on,
# because the lock says "she wears <jacket>, unchanged." The result is the
# flicker the user described: shirt on, shirt off, shirt on.
#
# The fix: a STATE MACHINE that walks panels in narrative order, starts from
# the registry/visual-bible baseline, detects intentional appearance changes,
# and carries the EVOLVED state forward. Enforcement is then on the CURRENT
# state (post-change), so:
#   • unintentional drift is prevented (the jacket never reappears on its own)
#   • intentional changes PERSIST (once off, it stays off until put back on)
#   • the panel WHERE a change happens shows the transition (the act of removal)
#
# Separation of concerns
# ----------------------
#   Identity (face, body, skin, base hair colour, gender)  → NEVER changes.
#       Enforced from the registry portrait on every panel.
#   Mutable state (clothing, hair arrangement, physical
#       condition, held items)                             → EVOLVES per panel.
#       Tracked here; enforced from the current evolved state.
#   Setting / lighting / environment                       → frozen per scene.
#       Still handled by SceneConsistencyAnchor.
#
# Design: the LLM is used only for CHANGE DETECTION (a hard language task);
# the running STATE is maintained in Python (deterministic, inspectable, and
# resilient — if the LLM fails on a batch we simply carry state unchanged).
# ===========================================================================

# Mutable appearance attributes that the tracker follows. Identity attributes
# (face, body, gender, base hair colour) are deliberately excluded — they are
# enforced separately and must never change.
_TRACKED_ATTRS = ('clothing', 'hair', 'condition', 'held_items',
                  'emotion', 'markings', 'permanent_markings', 'closure')

# ---------------------------------------------------------------------------
# Naturally-unclothed entity detection
# ---------------------------------------------------------------------------
# Some characters — animals, certain supernatural/divine beings, abstract forces
# — do not wear clothes as their baseline state. For these entities clothing
# tracking behaves differently:
#
#   • Their "clothing" field holds their NATURAL COVERING (fur, feathers, scales,
#     divine radiance, bark-skin, etc.), NOT a garment.  It should be locked
#     with "has {value}" (consistent natural appearance) instead of "be wearing
#     {value}" (a garment that can be put on or taken off).
#
#   • The nudity / undress condition system is suppressed — a naked dog is not
#     "nude", and stripping a dog's fur is a story event so extreme it would be
#     described explicitly and handled as a deliberate change.
#
#   • A garment appearing on a naturally-unclothed character IS a legitimate
#     story change (a dog wearing a service-dog vest, an angel donning armour)
#     and should be tracked normally once detected.
#
# Detection: we scan the character's role, appearance, traits, and name against
# the marker sets below.  Markers are kept conservative (whole-word where
# possible) to avoid false-positives ("cat burglar", "foxy", "bear market").
# ---------------------------------------------------------------------------

# Animal / creature markers — characters that are naturally unclothed
_ANIMAL_MARKERS: Tuple[str, ...] = (
    'dog', 'puppy', 'hound', 'canine',
    'cat', 'kitten', 'feline',
    'wolf', 'wolves', 'werewolf',
    'fox', 'vixen',
    'bear', 'cub',
    'horse', 'mare', 'stallion', 'foal', 'pony',
    'lion', 'lioness', 'tiger', 'leopard', 'jaguar', 'cheetah',
    'deer', 'stag', 'doe',
    'rabbit', 'bunny', 'hare',
    'bird', 'crow', 'raven', 'hawk', 'eagle', 'owl', 'parrot',
    'snake', 'serpent', 'dragon',
    'fish', 'shark', 'dolphin', 'whale',
    'elephant', 'rhinoceros', 'hippopotamus',
    'monkey', 'ape', 'gorilla', 'chimpanzee',
    'rat', 'mouse', 'squirrel',
    'animal', 'creature', 'beast', 'familiar',
)

# Supernatural / divine beings that manifest without garments by default
_DIVINE_UNCLOTHED_MARKERS: Tuple[str, ...] = (
    'angel', 'archangel', 'seraph', 'cherub',
    'deity', 'god', 'goddess',
    'spirit', 'ghost', 'phantom', 'apparition',
    'demon', 'devil', 'fiend',
    'elemental', 'wisp', 'wraith',
    'golem', 'construct',
)


def _entity_is_naturally_unclothed(character) -> bool:
    """True when a character is naturally unclothed by their nature.

    Scans the character's role, appearance, name, and traits for animal or
    naturally-unclothed supernatural markers.  Uses whole-word matching where
    the marker is short enough to produce false-positives as a substring
    (e.g. 'bear' in 'bearable', 'cat' in 'concatenate').

    A negative idiom guard blocks the common English figures of speech that use
    an animal word to describe a HUMAN character ("cat burglar", "top dog",
    "bear market", "night owl", "old fox") so those never trip the detector.

    Returns False for humans and human-presenting characters whose baseline
    state is clothed — the vast majority of story characters.
    """
    def _g(attr: str) -> str:
        v = getattr(character, attr, None)
        if v is None and isinstance(character, dict):
            v = character.get(attr)
        if isinstance(v, (list, tuple)):
            return ' '.join(str(x) for x in v)
        return str(v or '').lower()

    blob = ' '.join(_g(a) for a in
                    ('role', 'appearance', 'name', 'backstory', 'traits',
                     'mythic_archetype', 'physical_build'))

    # Idiom guard: phrases where an animal word describes a HUMAN. If a short
    # animal marker only appears as part of one of these, it must not trigger.
    # We strip these phrases from the blob before matching so the underlying
    # animal word can't match on them, while a genuine second mention still can.
    _HUMAN_ANIMAL_IDIOMS = (
        'cat burglar', 'top dog', 'underdog', 'bear market', 'bull market',
        'night owl', 'old fox', 'sly fox', 'lone wolf', 'sea dog', 'war horse',
        'dark horse', 'stool pigeon', 'social butterfly', 'lounge lizard',
        'road hog', 'guinea pig', 'cash cow', 'scapegoat', 'copycat',
        'fat cat', 'cool cat', 'rat race', 'gym rat', 'mall rat', 'pack rat',
        'bird brain', 'jail bird', 'early bird', 'wolf pack', 'she-wolf of',
        'dog days', 'dog tag', 'watchdog', 'sheepdog',
    )
    for idiom in _HUMAN_ANIMAL_IDIOMS:
        if idiom in blob:
            blob = blob.replace(idiom, ' ')

    # Short markers that are risky as substrings get whole-word matching.
    _short = {
        'dog', 'cat', 'fox', 'owl', 'rat', 'god', 'ape', 'doe', 'cub',
        'god', 'bear', 'deer', 'hare',
    }
    for marker in _ANIMAL_MARKERS + _DIVINE_UNCLOTHED_MARKERS:
        if marker in _short:
            if re.search(rf'\b{re.escape(marker)}\b', blob):
                return True
        else:
            if marker in blob:
                return True
    return False


@dataclass
class CharacterAppearanceState:
    """The evolving, mutable appearance of one character at a story point.

    base_clothing is the registry default — the starting wardrobe. As the story
    flows, `clothing` may diverge from it (removed jacket, changed shirt, torn
    sleeve) and that divergence persists forward until another change.

    `conditions` is a SET of active persistent-condition names (see
    _PERSISTENT_CONDITIONS). This is the general mechanism that keeps any
    acquired transient physical state — nudity, mud, blood, a cut, wetness,
    dust, soot, sweat, tears, bruising — consistent across panels: once a
    condition is acquired it stays active (and is asserted in every subsequent
    panel's prompt) until an explicit resolution event or a scene change clears
    it. Tracking these as discrete flags we control is far more reliable than
    hoping a free-text field keeps restating them panel after panel.

    The single-value EVOLVING facets (clothing, hair, emotion, held_items,
    markings) are governed by the `_FACETS` table: each persists until the story
    changes it, is rendered/locked per that table, and is the sibling mechanism
    to `conditions` for "stays until changed".

    ``naturally_unclothed`` marks animals, certain supernatural beings, and any
    entity whose baseline state is not wearing garments (a dog, an angel, a
    dragon).  For these characters:
      - the ``clothing`` field holds their NATURAL COVERING (fur, feathers,
        scales, divine radiance) rather than a garment description.
      - the lock instruction is "has {value}" instead of "be wearing {value}",
        which is semantically correct and prevents the image model from trying
        to add or remove clothing on a naturally-unclothed entity.
      - the nudity / undress condition system is suppressed — a naked dog is
        not "nude" in the story sense.
      - a garment appearing on a naturally-unclothed character (service-dog
        vest, angelic armour) IS tracked as a story change like any other.
    """
    name: str = ""
    gender: str = "figure"
    base_clothing: str = ""        # registry default — the starting point
    clothing: str = ""             # CURRENT clothing/covering (evolves)
    hair: str = ""                 # CURRENT hair arrangement (loose/tied/wet/cut)
    condition: str = ""            # free-text transient state (LLM-supplied nuance)
    held_items: str = ""           # currently held/brandished objects
    emotion: str = ""              # CURRENT worn emotional register (sad, seething…)
    markings: str = ""             # applied face/body marks (paint, stamp, smear)
    # CURRENT closure / fastening sub-state of the worn garment(s): whether a
    # shirt is buttoned or open, a zip up or down, sleeves rolled or down, a
    # collar popped, a shirt tucked or untucked. This is a SUB-STATE of the same
    # garment (not a different outfit), so it is tracked separately from
    # `clothing`: a shirt that is buttoned, then unbuttoned, then buttoned again
    # is the SAME shirt throughout — only its closure changed. Empty = no
    # particular closure state established yet (as-drawn). Locked into the image
    # prompt so the closure does not silently flicker between panels; only an
    # explicit story action (buttons/unbuttons/zips/rolls/tucks) changes it.
    closure: str = ""
    conditions: set = field(default_factory=set)  # active persistent-condition names
    first_seen_page: int = 0       # for logging
    naturally_unclothed: bool = False  # animal / unclothed entity — see docstring
    # Permanent physical marks that survive scene resets and are injected into
    # the portrait lock for every subsequent appearance of this character.
    # Unlike `markings` (transient — face paint washes off, a stamp fades),
    # `permanent_markings` represent lasting changes to the character's body:
    # a significant scar acquired mid-story, a tattoo, a missing eye, a
    # prosthetic limb.  Once set they are NEVER cleared by scene changes or
    # condition resets — only an explicit story event (cosmetic surgery, a
    # miraculous healing) that the LLM detection pass reports should update them.
    permanent_markings: str = ""
    # Ordered history of every distinct clothing value this character has worn,
    # oldest first. Used by the regression guard in _detect_changes_llm so the
    # LLM can see whether a proposed new clothing value looks like a reversion
    # to a prior state rather than a genuine forward progression.
    _clothing_history: list = field(default_factory=list)
    # Same for hair — ties put up, then let down, then put back up is equally
    # detectable as a suspicious regression.
    _hair_history: list = field(default_factory=list)
    # Same for closure/fastening — buttoned → unbuttoned → buttoned is the
    # SAME shirt with its closure flip-flopping. The history trail lets the
    # regression guard flag a re-buttoning that the story never actually shows,
    # which is the classic uncontrolled-image-prompt flicker.
    _closure_history: list = field(default_factory=list)

    # --- nudity convenience (nudity is just one persistent condition) ---------
    @property
    def undressed(self) -> bool:
        return 'nude' in self.conditions

    def set_undressed(self, value: bool) -> bool:
        return self.add_condition('nude') if value else self.remove_condition('nude')

    def add_condition(self, name: str) -> bool:
        """Activate a persistent condition. Returns True if newly added."""
        name = (name or '').strip().lower()
        if not name or name in self.conditions:
            return False
        self.conditions.add(name)
        # Becoming nude removes every garment, so any fastening/closure sub-state
        # (buttoned, sleeves rolled, tucked) is now meaningless — clear it and
        # its history so a stale "keep the garment buttoned" lock can't fight the
        # undress, and so re-dressing later starts from a clean closure slate.
        if name == 'nude':
            self.closure = ""
            self._closure_history = []
        return True

    def remove_condition(self, name: str) -> bool:
        """Clear a persistent condition. Returns True if it was active."""
        name = (name or '').strip().lower()
        if name in self.conditions:
            self.conditions.discard(name)
            return True
        return False

    def clear_facet(self, name: str) -> bool:
        """Empty an evolving facet (held_items, emotion, markings). Returns True
        if it actually held a value."""
        if name not in _FACETS_BY_NAME:
            return False
        if getattr(self, name, ''):
            setattr(self, name, "")
            return True
        return False

    def clear_held_items(self) -> bool:
        """Back-compat shim: empty the character's hands."""
        return self.clear_facet('held_items')

    def snapshot(self) -> Dict[str, object]:
        """Return a plain-dict snapshot of the current mutable state."""
        return {
            'name': self.name,
            'gender': self.gender,
            'clothing': self.clothing,
            'hair': self.hair,
            'condition': self.condition,
            'held_items': self.held_items,
            'emotion': self.emotion,
            'markings': self.markings,
            'closure': self.closure,
            'permanent_markings': self.permanent_markings,
            'conditions': sorted(self.conditions),   # JSON-serialisable list
            'undressed': self.undressed,             # back-compat convenience
            'naturally_unclothed': self.naturally_unclothed,
        }

    def apply_change(self, attribute: str, new_value: str) -> bool:
        """Apply a detected change. Returns True if the state actually changed.

        Evolving facets (clothing, hair, emotion, held_items, markings) are
        handled uniformly through the `_FACETS` table: a facet that allows
        emptiness can be CLEARED by a sentinel value ("nothing", "neutral",
        "no markings"), while a facet that does not (clothing, hair) treats an
        empty value as a no-op so a costume is never silently blanked. `condition`
        remains the legacy free-text nuance field.

        Clothing and hair changes are appended to ``_clothing_history`` /
        ``_hair_history`` so the regression guard in ``_detect_changes_llm``
        can show the LLM the full progression and flag reversions.
        """
        attribute = (attribute or '').strip().lower()
        new_value = (new_value or '').strip()
        if attribute not in _TRACKED_ATTRS:
            return False
        old = getattr(self, attribute, '')

        facet = _FACETS_BY_NAME.get(attribute)
        if facet is not None:
            if facet.allow_empty and facet.is_empty(new_value):
                return self.clear_facet(attribute)
            # Strip a filler lead-in ("she is wearing …", "dressed in …") from
            # clothing/hair values at STORAGE time so the stored canonical value
            # — used by both the hard lock and the soft descriptor — starts on
            # the real content. This removes only grammatical filler; no visual
            # detail is lost. (The heavier comma-compression happens later, at
            # image-prompt build time, and only on the soft line.)
            if attribute in ('clothing', 'hair'):
                _stripped = _DESC_LEAD_FILLER_RE.sub('', new_value).strip()
                if _stripped:
                    new_value = _stripped
            if not new_value or new_value == old:
                return False
            setattr(self, attribute, new_value)
            if attribute == 'clothing':
                self._sync_conditions_from_text(new_value)
                # Record every distinct clothing value for the regression guard.
                if not self._clothing_history or self._clothing_history[-1] != new_value:
                    self._clothing_history.append(new_value)
                # A genuinely new outfit starts with a clean closure slate — the
                # old shirt's buttoned/unbuttoned history must not false-flag the
                # new garment's closure. Clear both the current closure and its
                # history so closure tracking restarts for the new outfit.
                self.closure = ""
                self._closure_history = []
            elif attribute == 'hair':
                if not self._hair_history or self._hair_history[-1] != new_value:
                    self._hair_history.append(new_value)
            elif attribute == 'closure':
                if not self._closure_history or self._closure_history[-1] != new_value:
                    self._closure_history.append(new_value)
            return True

        # Legacy free-text 'condition' field.
        if not new_value or new_value == old:
            return False
        # permanent_markings are append-only: a new scar or tattoo is added to
        # whatever marks already exist rather than replacing them, and they can
        # never be cleared by a scene change or a regression guard reset.
        if attribute == 'permanent_markings':
            if old:
                new_value = f"{old}; {new_value}"
            setattr(self, attribute, new_value)
            return True
        setattr(self, attribute, new_value)
        self._sync_conditions_from_text(new_value)
        return True

    def _sync_conditions_from_text(self, text: str) -> None:
        """Add/remove structured conditions implied by a free-text value."""
        for cond in _PERSISTENT_CONDITIONS:
            if cond.text_indicates_active(text):
                self.add_condition(cond.name)
            elif cond.text_indicates_resolved(text):
                self.remove_condition(cond.name)


# ===========================================================================
# GENERAL PERSISTENT-CONDITION SYSTEM
# ---------------------------------------------------------------------------
# A character's transient physical state is more than just clothing. Mud from a
# splash, a bleeding cut, a bloody nose, being soaked from water, dust, soot,
# sweat, tears, bruises — each is ACQUIRED at some panel and then PERSISTS in
# every following panel of the scene until it is explicitly resolved (washed
# off, bandaged, dried, wiped away) or the scene changes. The reported nudity
# flicker is one instance of a general failure: the artwork dropping an acquired
# condition because a later panel's description simply doesn't re-mention it.
#
# This data-driven table lets the art director enforce ALL such conditions with
# one mechanism. Each entry carries:
#   - acquire/resolve phrase cues (conservative, word-boundary matched)
#   - a prompt descriptor injected verbatim so the model renders it
#   - resets_on_scene_change: whether a hard cut to a new scene clears it
#       (nudity & wetness: yes — they'd dress/dry; a fresh cut: no — it doesn't
#        heal between two adjacent scenes)
# ===========================================================================

@dataclass
class PersistentCondition:
    name: str                       # canonical id, e.g. 'nude', 'muddy', 'bleeding'
    descriptor: str                 # text injected into the image prompt
    acquire: Tuple[str, ...]        # phrases that ACQUIRE the condition
    resolve: Tuple[str, ...] = ()   # phrases that RESOLVE/clear it
    category: str = 'condition'     # 'dress' | 'substance' | 'injury' | 'wetness'
    resets_on_scene_change: bool = False
    # Optional phrases that, if present, BLOCK acquisition (guard against the
    # single-garment false positive: "peels off her jacket" must not = nude).
    block: Tuple[str, ...] = ()

    def _has(self, text: str, phrases: Tuple[str, ...]) -> bool:
        for p in phrases:
            if re.search(r'\b' + re.escape(p) + r'\b', text):
                return True
        return False

    def text_indicates_active(self, text: str) -> bool:
        t = (text or '').lower()
        if not t.strip():
            return False
        if self.block and self._has(t, self.block):
            return False
        return self._has(t, self.acquire)

    def text_indicates_resolved(self, text: str) -> bool:
        t = (text or '').lower()
        if not t.strip() or not self.resolve:
            return False
        return self._has(t, self.resolve)


# The table. Conservative acquisition cues; specific resolution cues.
_PERSISTENT_CONDITIONS: List[PersistentCondition] = [
    PersistentCondition(
        name='nude', category='dress',
        descriptor='completely nude, no clothing of any kind',
        acquire=(
            'nude', 'naked', 'fully nude', 'completely nude', 'stark naked',
            'unclothed', 'no clothes', 'nothing on', 'in the nude', 'bare body',
            'bare naked', 'undress', 'undresses', 'undressing', 'undressed',
            'strip down', 'strips down', 'stripping down', 'stripped down',
            'strips naked', 'stripped naked', 'strip naked',
            'strips off her clothes', 'strips off his clothes',
            'peels off her clothes', 'peels off his clothes',
            'takes off her clothes', 'takes off his clothes',
            'removes her clothes', 'removes his clothes',
            'sheds her clothes', 'sheds his clothes', 'sheds their clothes',
            'disrobes', 'disrobing', 'pulls off her clothes', 'pulls off his clothes',
            'tears off her clothes', 'tears off his clothes',
            'clothes fall away', 'clothing falls away',
        ),
        resolve=(
            'gets dressed', 'dresses', 'dressing', 'puts on', 'putting on',
            'pulls on', 'pulling on', 'slips on', 'slips into', 'pulls back on',
            'back on', 'buttons up', 'buttons his', 'buttons her', 'zips up',
            'wraps a robe', 'pulls a fresh', 'redresses', 'covers herself',
            'covers himself', 'pulls the sheet', 'wraps the sheet', 'dons',
            'now dressed', 'fully clothed again', 'clothed again', 'dressed again',
            'pulls her dress back', 'pulls his shirt back',
        ),
        resets_on_scene_change=True,
    ),
    PersistentCondition(
        name='muddy', category='substance',
        descriptor='spattered with mud, dirt streaking the skin and clothing',
        acquire=(
            'splashed with mud', 'mud splash', 'splashes mud', 'sprayed with mud',
            'covered in mud', 'caked in mud', 'mud-spattered', 'muddy',
            'falls in the mud', 'falls into the mud', 'lands in the mud',
            'mud splatters', 'spattered with dirt', 'covered in dirt',
            'caked with dirt', 'streaked with mud',
        ),
        resolve=(
            'wipes off the mud', 'wipes the mud', 'washes off the mud', 'washes',
            'washed', 'cleans up', 'cleaned up', 'showers', 'showered', 'hoses off',
            'wipes herself clean', 'wipes himself clean', 'scrubs clean', 'now clean',
            'cleans herself', 'cleans himself',
        ),
        resets_on_scene_change=False,
    ),
    PersistentCondition(
        name='bloody_nose', category='injury',
        descriptor='blood running from the nose, smeared across the upper lip',
        acquire=(
            'bloody nose', 'nose bleeds', 'nosebleed', 'blood from his nose',
            'blood from her nose', 'blood trickles from the nose',
            'blood running from the nose', 'punched in the nose', 'broken nose',
        ),
        resolve=(
            'wipes the blood', 'wipes his nose', 'wipes her nose', 'staunches',
            'cleans the blood', 'blood stops', 'nose stops bleeding', 'tissue to the nose',
        ),
        resets_on_scene_change=False,
    ),
    PersistentCondition(
        name='bleeding_cut', category='injury',
        descriptor='a fresh bleeding cut, blood welling and trickling from the wound',
        acquire=(
            'bleeding cut', 'gash', 'slashed', 'open wound', 'blood wells',
            'blood trickles', 'cut on her', 'cut on his', 'cut across',
            'blade bites', 'knife catches', 'wound on', 'laceration', 'deep cut',
            'blood seeps', 'blood streaks down', 'cut opens',
        ),
        resolve=(
            'bandage', 'bandaged', 'bandages the', 'stitches', 'stitched',
            'wound dressed', 'dresses the wound', 'staunches the', 'cleans the wound',
            'healed', 'wound closed', 'patched up', 'gauze',
        ),
        resets_on_scene_change=False,
    ),
    PersistentCondition(
        name='bruised', category='injury',
        descriptor='a darkening bruise, swollen and discoloured',
        acquire=(
            'bruise', 'bruised', 'black eye', 'swollen cheek', 'welt',
            'split lip', 'busted lip', 'contusion',
        ),
        resolve=('bruise fades', 'healed', 'ice on the', 'bruise gone'),
        resets_on_scene_change=False,
    ),
    PersistentCondition(
        name='wet', category='wetness',
        descriptor='soaking wet, water dripping, hair plastered down and clothing clinging',
        acquire=(
            'soaked', 'soaking wet', 'drenched', 'dripping wet', 'sopping',
            'falls into the water', 'falls in the river', 'plunges into',
            'emerges from the river', 'emerges from the water', 'climbs out of the water',
            'rain pours', 'caught in the rain', 'doused', 'splashed with water',
            'soaked to the skin', 'waterlogged',
        ),
        resolve=(
            'dries off', 'dried off', 'towels off', 'towelled off', 'dry clothes',
            'now dry', 'wrung out', 'wrings out', 'changes into dry', 'pats herself dry',
            'pats himself dry',
        ),
        resets_on_scene_change=True,
    ),
    PersistentCondition(
        name='dusty', category='substance',
        descriptor='covered in dust, a film of grey grime over skin and clothes',
        acquire=(
            'covered in dust', 'dusty', 'coated in dust', 'caked in dust',
            'cloud of dust', 'dust settles over', 'grimy with dust', 'ash-covered',
            'covered in soot', 'sooty', 'streaked with soot', 'blackened with soot',
        ),
        resolve=(
            'dusts off', 'dusts herself', 'dusts himself', 'brushes off the dust',
            'washes', 'washed', 'cleans up', 'cleaned up', 'now clean', 'wipes clean',
        ),
        resets_on_scene_change=True,
    ),
    PersistentCondition(
        name='sweaty', category='condition',
        descriptor='glistening with sweat, skin sheened and damp',
        acquire=(
            'drenched in sweat', 'covered in sweat', 'sweat-soaked', 'pouring sweat',
            'sweat beads', 'sweat drips', 'glistening with sweat', 'sheen of sweat',
        ),
        resolve=('wipes the sweat', 'mops his brow', 'mops her brow', 'cooled down', 'now dry'),
        resets_on_scene_change=True,
    ),
    PersistentCondition(
        name='tear-streaked', category='condition',
        descriptor='tear-streaked face, eyes red and cheeks wet',
        acquire=(
            'tears stream', 'tears streaming', 'tear-streaked', 'crying',
            'weeping', 'tears roll', 'tears spill', 'sobbing', 'eyes well with tears',
        ),
        resolve=('wipes her eyes', 'wipes his eyes', 'dries her tears', 'dries his tears',
                 'composes herself', 'composes himself', 'stops crying', 'tears dry'),
        resets_on_scene_change=True,
    ),
    # ── Additions: high-frequency story states the table previously missed ──
    PersistentCondition(
        name='torn_clothing', category='dress',
        descriptor='clothing visibly torn and ripped, ragged tears in the fabric',
        acquire=(
            'clothes torn', 'clothing torn', 'torn clothing', 'torn clothes',
            'shirt rips', 'shirt ripped', 'jacket rips', 'jacket ripped',
            'sleeve tears', 'sleeve torn', 'fabric tears', 'fabric rips',
            'rips her dress', 'rips his shirt', 'tattered clothes',
            'clothes in tatters', 'clothing shredded', 'ragged clothes',
            'tears his shirt', 'tears her dress', 'clothes ripped',
        ),
        resolve=(
            'changes clothes', 'changes into', 'fresh clothes', 'new clothes',
            'mends her', 'mends his', 'sews up', 'patched up clothing',
            'changed into clean',
        ),
        # Torn fabric does not repair itself between scenes — unlike wetness
        # the damage persists until the story changes the clothing.
        resets_on_scene_change=False,
        # Guard: 'torn between' is an emotion, not fabric damage.
        block=('torn between', 'torn apart inside', 'heart torn'),
    ),
    PersistentCondition(
        name='blood_spattered', category='substance',
        descriptor="spattered with blood that is not their own, dark flecks across clothing and skin",
        acquire=(
            'spattered with blood', 'splattered with blood', 'blood spatter',
            'blood splatter', 'sprayed with blood', 'covered in blood',
            'blood-soaked', 'blood soaked', 'flecked with blood',
            "someone else's blood", 'blood across her face', 'blood across his face',
        ),
        resolve=(
            'washes off the blood', 'washes the blood', 'wipes off the blood',
            'wipes the blood away', 'cleans the blood', 'scrubs the blood',
            'washes', 'washed', 'cleans up', 'cleaned up', 'now clean',
        ),
        # Blood doesn't vanish on a scene cut — it stays until washed.
        resets_on_scene_change=False,
        # Guard: bleeding from one's own wound is the bleeding_cut condition.
        block=('her own blood', 'his own blood', 'bleeding from'),
    ),
    PersistentCondition(
        name='snow_covered', category='wetness',
        descriptor='dusted with snow, white flakes settled on shoulders and hair',
        acquire=(
            'covered in snow', 'dusted with snow', 'snow settles on',
            'snow in her hair', 'snow in his hair', 'caked with snow',
            'trudges through the snow', 'blizzard', 'snowstorm',
            'snow falls on', 'snow-covered shoulders',
        ),
        resolve=(
            'brushes off the snow', 'shakes off the snow', 'snow melts',
            'steps inside', 'comes in from the cold', 'by the fire',
            'warms up', 'thaws',
        ),
        # Like wetness: indoors next scene, the snow would have been shed.
        resets_on_scene_change=True,
    ),
    PersistentCondition(
        name='bandaged', category='injury',
        descriptor='visible bandage wrapping the wound, white gauze secured in place',
        acquire=(
            'bandaged', 'bandages the wound', 'bandages her', 'bandages his',
            'wraps the wound', 'wrapped in gauze', 'gauze wrapped', 'dresses the wound',
            'applies a bandage', 'field dressing', 'wound dressed',
        ),
        resolve=(
            'removes the bandage', 'bandage comes off', 'unwraps the bandage',
            'fully healed', 'wound has healed', 'scar has faded',
        ),
        # A dressing stays on across scenes and days until removed/healed.
        # Chains naturally with bleeding_cut: the same bandaging phrases that
        # RESOLVE the bleeding condition ACQUIRE this one, so the artwork
        # transitions from open wound to dressed wound instead of the injury
        # simply vanishing.
        resets_on_scene_change=False,
    ),
]

# Quick lookup by name.
_CONDITIONS_BY_NAME: Dict[str, PersistentCondition] = {
    c.name: c for c in _PERSISTENT_CONDITIONS
}


def _scan_conditions(description: str) -> Tuple[set, set]:
    """Scan a panel description for condition acquisition/resolution cues.

    Returns (acquired, resolved) sets of condition names. Conservative: only
    fires on the explicit cues in the table, so ordinary prose never spuriously
    flags a condition. Resolution wins over acquisition for the same condition
    in one panel (a panel that both mentions and clears a state resolves it).
    """
    acquired: set = set()
    resolved: set = set()
    if not (description or '').strip():
        return acquired, resolved
    for cond in _PERSISTENT_CONDITIONS:
        active = cond.text_indicates_active(description)
        done = cond.text_indicates_resolved(description)
        if done:
            resolved.add(cond.name)
        elif active:
            acquired.add(cond.name)
    return acquired, resolved


# ===========================================================================
# GENERAL PERSISTENT-FACET CONTINUITY
# ---------------------------------------------------------------------------
# A character is more than clothing + physical conditions. Across a scene they
# may acquire ANY visual state that the reader then expects to persist until the
# story changes it: an object in hand (a ball, a sword), an emotional register
# worn on the face (sad, seething, hollow with grief), a marking applied to the
# body (a heart painted on the cheek, war paint, a stamped hand, smudged
# lipstick), a changed hairstyle, and so on. The failure mode is always the
# same: the LLM notes the change once, then a later panel that simply doesn't
# re-mention it lets the artwork silently revert — the ball vanishes, the grief
# resets to a neutral face, the painted heart disappears.
#
# Rather than special-case each kind of dynamic, we model them all as PERSISTENT
# FACETS: a single, ordered, data-driven table where each entry declares how its
# current value is rendered into the prompt, how it is authoritatively locked so
# the diffusion model cannot drop it, whether it can be cleared (and by what
# textual cues), and whether a hard scene cut resets it. Adding a new kind of
# character dynamic is then ONE table row, not bespoke logic. This is the
# evolving-single-value sibling of the multi-flag `_PERSISTENT_CONDITIONS`
# table above (which handles simultaneous physical states like mud + blood +
# wet); together they cover the full surface of "stays until the story changes
# it."
# ===========================================================================

@dataclass
class TrackedFacet:
    """Declarative spec for one evolving, persistent character facet."""
    name: str                          # 'clothing','hair','emotion','held_items','markings'
    descriptor_template: str           # soft state line, e.g. "holding {value}"
    # Authoritative lock predicate, beginning with a bare verb so fragments
    # join cleanly after "must still …": e.g. "be holding the {value}",
    # "remain {value}", "bear {value}". Empty -> facet is SOFT-ONLY (it appears
    # in the state descriptor but is never hard-locked — right for clothing/hair,
    # which legitimately shift often and shouldn't be frozen).
    lock_template: str = ""
    allow_empty: bool = False          # can this facet be cleared to "nothing"?
    resets_on_scene_change: bool = False
    release_cues: Tuple[str, ...] = ()       # phrase cues that CLEAR the facet
    release_patterns: Tuple[str, ...] = ()   # regex cues (verb…object…particle)
    empty_sentinels: Tuple[str, ...] = ()    # values meaning "cleared"
    empty_regex: str = ""              # regex on the normalised value meaning cleared
    strip_article: bool = True         # strip leading a/an/the before {value} in lock

    def is_empty(self, value: str) -> bool:
        v = (value or '').strip()
        if not self.allow_empty:
            return not v
        if not v:
            return True
        low = v.lower().rstrip('.')
        if low in self.empty_sentinels:
            return True
        core = re.sub(r'\(.*?\)', '', low).strip(' .,-')
        if core in self.empty_sentinels:
            return True
        if self.empty_regex and re.match(self.empty_regex, core):
            return True
        return False

    def scan_release(self, description: str) -> bool:
        """True if the description explicitly CLEARS this facet (set the ball
        down, wipe the paint off …). Conservative, word-boundary matched."""
        t = (description or '').lower()
        if not t.strip():
            return False
        for cue in self.release_cues:
            if re.search(r'\b' + re.escape(cue) + r'\b', t):
                return True
        for pat in self.release_patterns:
            if re.search(pat, t):
                return True
        return False

    def lock_fragment(self, value: str) -> str:
        """Render the authoritative 'must still …' predicate, or '' if soft."""
        if not self.lock_template:
            return ''
        v = (value or '').strip()
        if self.strip_article:
            v = re.sub(r'^(a|an|the)\s+', '', v, flags=re.I)
        return self.lock_template.format(value=v)

    def descriptor(self, value: str) -> str:
        return self.descriptor_template.format(value=(value or '').strip())


# Verb/phrase cues that RELEASE whatever a character is holding. Kept explicit
# and word-boundary matched so ordinary prose never spuriously empties a hand.
_HELD_RELEASE_CUES: Tuple[str, ...] = (
    'sets it down', 'sets them down', 'sets it aside', 'sets them aside',
    'puts it down', 'puts them down', 'puts it away', 'puts them away',
    'lays it down', 'lays them down', 'lays it aside', 'sets down the',
    'puts down the', 'lays down the', 'puts away the', 'sets aside the',
    'drops it', 'drops them', 'drops the', 'lets it fall', 'lets it drop',
    'lets go', 'lets go of', 'releases', 'releasing', 'released',
    'hands it over', 'hands them over', 'hands it to', 'hands over the',
    'hands the', 'passes it', 'passes the', 'passes them', 'gives it to',
    'gives the', 'hands off', 'tosses it', 'tosses the', 'tosses them',
    'throws it', 'throws the', 'throws them', 'hurls it', 'hurls the',
    'flings it', 'flings the', 'casts it aside', 'tosses it aside',
    'sheathes', 'sheaths', 'sheathing', 'sheathed', 're-sheathes',
    'holsters', 'holstering', 'holstered', 'pockets', 'pocketing',
    'stows', 'stowing', 'stowed', 'tucks it away', 'tucks away the',
    'slips it into', 'slides it into', 'puts it back', 'sets it back',
    'returns it', 'hangs it up', 'hangs up the', 'places it on',
    'places it down', 'rests it on', 'leaves it on', 'abandons the',
    'empty-handed', 'empty handed', 'hands now empty', 'hands are empty',
    'no longer holding', 'no longer carrying', 'no longer carries',
    'lowers the', 'shoves it into', 'puts it in his pocket',
    'puts it in her pocket', 'stuffs it into', 'hands now free',
)
_HELD_RELEASE_PATTERNS: Tuple[str, ...] = (
    r'\b(?:sets?|puts?|lays?) (?:it|them|the\b[\w\s]{0,30}?|her|his) down\b',
    r'\b(?:sets?|puts?|lays?) [\w\s]{0,30}? (?:down|aside|away)\b',
    r'\bputs? [\w\s]{0,30}? (?:back|away)\b',
    r'\bhands? (?:it|them|him|her|over|off) (?:to|over|off|back)?\b',
    r'\bhands? (?:him|her|them|\w+) (?:the|a|an|it|them)\b',
    r'\bgives? (?:him|her|them|\w+) (?:the|a|an|it|them)\b',
    r'\b(?:tosses?|throws?|hurls?|flings?) (?:it|them|the|a|an|aside|away)\b',
    r'\bslips? (?:it|them) (?:into|in)\b',
    r'\b(?:tucks?|stuffs?|shoves?) (?:it|them|[\w\s]{0,20}?) (?:into|in|away)\b',
)
_HELD_EMPTY_SENTINELS: Tuple[str, ...] = (
    '', 'none', 'nothing', 'no items', 'no item', 'empty', 'empty-handed',
    'empty handed', 'nothing held', 'hands empty', 'hands free', 'n/a',
    'nil', '-', 'no longer holding anything', 'nothing in hand',
    'nothing in their hands', 'no held items',
)

# Cues that REMOVE a facial/body marking (wash the paint off, wipe it away).
_MARKING_REMOVE_CUES: Tuple[str, ...] = (
    'wipes off the paint', 'wipes the paint', 'wipes away the paint',
    'washes off the paint', 'washes the paint', 'scrubs off the paint',
    'rubs off the paint', 'wipes her face clean', 'wipes his face clean',
    'washes her face', 'washes his face', 'cleans the paint', 'paint is gone',
    'paint washed off', 'removes the paint', 'marking fades', 'mark fades',
    'wipes it off', 'rubs it off', 'washes it off', 'cleans it off',
    'paint smears away', 'no longer painted', 'paint gone', 'face now clean',
)
_MARKING_EMPTY_SENTINELS: Tuple[str, ...] = (
    '', 'none', 'no markings', 'no marking', 'unmarked', 'clean face',
    'nothing', 'no longer marked', 'wiped clean', 'face clean', 'bare face',
)


# The facet table. ORDER controls how facets read in the prompt.
_FACETS: List[TrackedFacet] = [
    TrackedFacet(
        name='clothing',
        descriptor_template='{value}',
        # Hard-lock clothing exactly like emotion/markings/held_items. The tracker
        # already excludes a facet from the lock on the panel where it is actively
        # changing, so this never freezes a legitimate costume change mid-transition.
        # Without this lock, the image model receives only a soft descriptor and can
        # silently revert a character from a suit back to a shirt simply because the
        # panel description does not re-state the outfit explicitly.
        lock_template='be wearing {value}',
        allow_empty=False,
        strip_article=False,             # "a red jacket" → lock says "be wearing a red jacket"
    ),
    TrackedFacet(
        name='hair',
        descriptor_template='hair {value}',
        # Hard-lock hair arrangement for the same reason as clothing: a character
        # whose hair was put up should not have it loose again two panels later
        # unless the story explicitly shows the change.
        lock_template='have hair {value}',
        allow_empty=False,
        strip_article=False,
    ),
    TrackedFacet(
        name='emotion',
        descriptor_template='a {value} expression',
        # Gentle but authoritative: the established mood persists on the face
        # unless this very panel shifts it. The tracker excludes a facet from
        # the lock on the panel where it changes, so this never freezes a beat.
        lock_template='look {value}',
        allow_empty=True,
        empty_sentinels=('neutral', 'blank', 'expressionless', 'composed',
                         'calm', 'no particular emotion'),
        strip_article=True,
    ),
    TrackedFacet(
        name='markings',
        descriptor_template='{value}',
        lock_template='bear {value}',
        allow_empty=True,
        release_cues=_MARKING_REMOVE_CUES,
        empty_sentinels=_MARKING_EMPTY_SENTINELS,
        empty_regex=r'^(no|none|unmarked|clean|bare)\b',
        strip_article=False,
    ),
    TrackedFacet(
        name='held_items',
        descriptor_template='holding {value}',
        lock_template='be holding the {value}',
        allow_empty=True,
        release_cues=_HELD_RELEASE_CUES,
        release_patterns=_HELD_RELEASE_PATTERNS,
        empty_sentinels=_HELD_EMPTY_SENTINELS,
        empty_regex=r'^(nothing|empty|no items?|no held|hands? (free|empty))\b',
        strip_article=True,
    ),
    TrackedFacet(
        name='closure',
        # Soft descriptor reads naturally: "shirt unbuttoned to the chest".
        descriptor_template='{value}',
        # Hard-lock the closure/fastening state so the image model keeps a shirt
        # buttoned (or open, or sleeves rolled) across panels instead of letting
        # it flicker. The tracker excludes a facet from the lock on the panel
        # where it actively changes, so a genuine story button/unbutton is never
        # frozen mid-action.
        lock_template='keep the garment {value}',
        allow_empty=True,
        # Closure is a property OF the current outfit, so a hard cut to a brand
        # new scene (often a new outfit) should not carry a stale closure state.
        resets_on_scene_change=True,
        empty_sentinels=('none', 'default', 'as drawn', 'as-drawn', 'unspecified',
                         'no particular closure', 'normal', 'neutral'),
        empty_regex=r'^(no|none|default|unspecified|as[\s-]?drawn)\b',
        strip_article=True,
    ),
]
_FACETS_BY_NAME: Dict[str, TrackedFacet] = {f.name: f for f in _FACETS}


def _facet_is_empty(name: str, value: str) -> bool:
    f = _FACETS_BY_NAME.get(name)
    return f.is_empty(value) if f else (not (value or '').strip())


# Back-compat thin wrappers (held-item helpers used to be standalone).
def _held_item_is_empty(value: str) -> bool:
    return _FACETS_BY_NAME['held_items'].is_empty(value)


def _scan_held_item_release(description: str) -> bool:
    return _FACETS_BY_NAME['held_items'].scan_release(description)


# --- Backward-compatible nudity helpers (now thin wrappers over the table) ----
def _CLOTHING_IS_NUDE(text: str) -> bool:
    return _CONDITIONS_BY_NAME['nude'].text_indicates_active(text) \
        or 'fully nude, no clothing' in (text or '').lower()


def _CLOTHING_IS_DRESSED(text: str) -> bool:
    """True only for affirmative clothing text that is NOT a nudity phrase."""
    t = (text or '').lower()
    if not t.strip() or _CLOTHING_IS_NUDE(t):
        return False
    garment_words = (
        'wearing', 'dress', 'shirt', 'suit', 'jacket', 'trousers', 'pants',
        'skirt', 'blouse', 'coat', 'gown', 'robe', 'sweater', 'jeans',
        'uniform', 'clothed', 'clothing', 'outfit', 'attire', 'lingerie',
        'underwear', 'bra', 'shorts', 't-shirt', 'tank top', 'sheet wrapped',
    )
    return any(w in t for w in garment_words)


def _detect_undress_dress(description: str) -> Optional[bool]:
    """Back-compat nudity detector, now derived from the general scan.

    Returns True if the panel acquires nudity, False if it resolves it
    (dressing), or None otherwise.
    """
    acquired, resolved = _scan_conditions(description)
    if 'nude' in resolved:
        return False
    if 'nude' in acquired:
        return True
    return None


# Function words that are too common to safely use as a standalone first-name
# match in _reconcile_frame. A titled character name like "The Witch" has
# "the" as its first token; matching on "the" alone would fire on nearly every
# panel description in the English language regardless of whether the
# character actually appears, silently flooding characters_in_frame with a
# character who isn't in the scene. Any character whose first name token is
# one of these words requires the FULL name to match instead of just the first
# token. This also guards ordinary names that happen to coincide with a common
# word (e.g. a character literally named "Will" or "May").
_NAME_TOKEN_STOPWORDS = frozenset({
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'and', 'or', 'but', 'nor', 'for', 'yet', 'so',
    'in', 'on', 'at', 'by', 'to', 'of', 'as', 'is', 'are', 'was', 'were',
    'with', 'from', 'into', 'onto', 'over', 'under', 'near', 'between',
    'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their',
    'who', 'what', 'when', 'where', 'why', 'how',
    'one', 'all', 'any', 'each', 'every', 'some', 'no', 'not',
})


class AppearanceContinuityTracker:
    """Walks the script in narrative order, evolving each character's mutable
    appearance and annotating every panel with the resolved current state.

    After ``process_script`` runs, every panel dict carries:
      panel['_appearance_state']   = {char_name: {clothing, hair, condition, ...}}
      panel['_appearance_changes'] = [{character, attribute, transition}, ...]
                                     (changes HAPPENING in this panel — used to
                                      show the transition action in the image)
    """

    def __init__(self, registry=None, visual_bible: Optional[SeriesVisualBible] = None,
                 character_names: Optional[List[str]] = None,
                 noncorporeal_names: Optional[List[str]] = None,
                 naturally_unclothed_names: Optional[List[str]] = None,
                 signature_markings: Optional[Dict[str, str]] = None):
        self.registry = registry
        self.visual_bible = visual_bible
        # {character_name: "tangible identity marker text"} — the always-visible
        # scar/tattoo/distinctive feature from the character's signature. Seeded
        # into permanent_markings on first appearance so the identity marker is a
        # tracked, hard-locked fact from panel one, not dependent on the LLM
        # re-describing it each scene. Matched case-insensitively.
        self.signature_markings: Dict[str, str] = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (signature_markings or {}).items() if str(v).strip()
        }
        # Full roster of known character names, used to reconcile
        # characters_in_frame against what each description actually depicts.
        self.character_names: List[str] = [
            str(n).strip() for n in (character_names or []) if str(n).strip()
        ]
        # Names of NON-CORPOREAL entities (a chorus of gods, a cosmic presence,
        # a disembodied spirit). Physical persistent conditions — nudity, sweat,
        # mud, blood, wetness — must never be applied to these: a watching
        # pantheon does not get "sweaty" or "nude" because the panel's human
        # protagonist is. Compared case-insensitively.
        self.noncorporeal_names = {
            str(n).strip().lower() for n in (noncorporeal_names or []) if str(n).strip()
        }
        # Names of characters that are NATURALLY UNCLOTHED by their nature —
        # animals, certain supernatural beings, etc. Their "clothing" field
        # stores their natural covering (fur, feathers, scales…), their lock
        # instruction uses "has" not "be wearing", and the nudity/undress
        # condition system is suppressed for them. Compared case-insensitively.
        self.naturally_unclothed_names = {
            str(n).strip().lower() for n in (naturally_unclothed_names or []) if str(n).strip()
        }
        self.states: Dict[str, CharacterAppearanceState] = {}
        # Setting string of the scene we're currently in. Used to reset transient
        # state (notably nudity) when the story moves to a new scene/location —
        # a character nude on a couch should not stay nude after a hard cut to a
        # different place/time unless that new scene also depicts nudity.
        self._current_setting: str = ""
        # Exposure tier of the current scene (public / private / intimate / …),
        # classified from the setting. Used to guard against acquiring undress in
        # a public place where it would be implausible. Recomputed on each scene
        # change alongside _current_setting.
        self._current_exposure: str = _EXPOSURE_UNKNOWN
        # Per-character setting-appropriate clothing for the current scene. When
        # a scene resolves a different appropriate baseline than the registry
        # default (e.g. street clothes for a public scene when the registry
        # baseline is intimate), that scene baseline is the reset target so
        # transient-condition resets land on setting-appropriate attire.
        self._scene_wardrobe: Dict[str, str] = {}
        # Rolling buffer of the last few (panel_id, description) pairs from the
        # PRECEDING batch, so the change-detector can see the story events that
        # produced the current incoming state. Without this, the regression
        # guard over-fires: a panel that references "the jacket he put on" reads
        # as a suspicious reversion because the batch can't see the panel where
        # he put it on. Kept short (3) so it never bloats the prompt.
        self._recent_descriptions: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------
    def _seed_state(self, name: str, page_num: int) -> CharacterAppearanceState:
        """Create the baseline state for a character on first appearance.

        For naturally-unclothed characters (animals, certain supernatural
        beings) the registry's clothing field may have been populated with a
        fallback garment description ("casual everyday clothes") by the
        CharacterAppearanceRegistry normaliser.  We detect this and strip it,
        leaving the clothing field empty so the image-prompt lock uses the
        character's natural-appearance description from the portrait instead.
        """
        clothing = ''
        gender = 'figure'
        if self.registry is not None:
            if hasattr(self.registry, 'get_clothing'):
                clothing = self.registry.get_clothing(name) or ''
            if not clothing and hasattr(self.registry, 'get_appearance'):
                clothing = self.registry.get_appearance(name) or ''
            if hasattr(self.registry, 'get_gender'):
                gender = self.registry.get_gender(name) or 'figure'

        is_unclothed = name.strip().lower() in self.naturally_unclothed_names

        # Strip the registry's garment fallback for naturally-unclothed entities.
        # The normaliser in CharacterAppearanceRegistry._build_registry writes
        # "casual everyday clothes, fully covered" when a character has no outfit,
        # which is wrong for a dog or angel. We clear it here so their clothing
        # field starts empty (no garment) and the portrait carries their look.
        if is_unclothed and clothing:
            _GARMENT_FALLBACKS = (
                'casual everyday clothes',
                'fully covered',
                'ordinary setting-appropriate',
                'setting-appropriate street clothes',
            )
            cl = clothing.lower()
            if any(f in cl for f in _GARMENT_FALLBACKS):
                clothing = ''

        st = CharacterAppearanceState(
            name=name,
            gender=gender or 'figure',
            base_clothing=clothing,
            clothing=clothing,
            first_seen_page=page_num,
            naturally_unclothed=is_unclothed,
            # Seed history with the baseline so the regression guard can
            # detect a reversion all the way back to the starting outfit.
            _clothing_history=[clothing] if clothing else [],
        )
        # Seed the identity marker (scar/tattoo/distinctive feature) so it is
        # locked from the character's very first panel and survives every scene
        # change (permanent_markings are append-only and never cleared).
        sig = self.signature_markings.get(name.strip().lower(), '')
        if sig:
            st.permanent_markings = sig
        self.states[name] = st
        return st

    def _resolve_scene_wardrobe(self, panel: Dict, page_num: int,
                                exposure: str) -> None:
        """Resolve setting-appropriate clothing for every character in a scene.

        Called on each scene change. Populates ``self._scene_wardrobe`` with
        ``{name: clothing}`` for the characters in this scene, where the clothing
        is what each character would plausibly be wearing given the scene's
        location and exposure tier.

        Resolution strategy (cheap → rich):
          1. If the registry baseline is already setting-appropriate (not an
             exposed/intimate descriptor, or the scene is private/intimate where
             anything goes), use it unchanged — no work, no LLM call.
          2. Otherwise (a public/semi-public scene whose characters carry an
             exposed or intimate baseline), derive a neutral, setting-appropriate
             outfit. Prefer an LLM call that dresses each character for the scene
             while preserving their identity/style; fall back to a deterministic
             neutral outfit so this never blocks synthesis.

        The resolved wardrobe is the *reset target* for transient-condition
        resets and the public-scene re-clothe guard — it does NOT overwrite the
        character's current clothing mid-scene (continuity still owns that).
        """
        self._scene_wardrobe = {}
        names = [n for n in _chars_in_frame(panel) if self._is_corporeal(n)]
        if not names:
            return

        setting = (panel.get('setting') or '').strip()

        # In private/intimate scenes, the registry/intimate baseline is fine —
        # nothing to resolve.
        if _exposure_allows_undress(exposure):
            for name in names:
                st = self.states.get(name)
                if st:
                    self._scene_wardrobe[name] = st.clothing or st.base_clothing
            return

        # Public / semi-public / unknown scene: ensure each character has a
        # setting-appropriate baseline. Only the characters whose baseline is
        # exposed/intimate need active re-dressing; the rest keep their baseline.
        need_redress: List[str] = []
        for name in names:
            st = self.states.get(name)
            if not st:
                continue
            baseline = st.base_clothing or ''
            if _clothing_is_exposed(baseline) or not baseline:
                need_redress.append(name)
            else:
                self._scene_wardrobe[name] = baseline

        if not need_redress:
            return

        # Try an LLM pass to dress the flagged characters for the scene. This
        # keeps each character's personal style while making the outfit
        # plausible for the location. Falls back to a neutral outfit on failure.
        resolved = self._llm_scene_wardrobe(need_redress, setting, exposure)
        for name in need_redress:
            outfit = (resolved.get(name) or '').strip()
            if not outfit:
                # Deterministic, identity-neutral fallback.
                outfit = "ordinary setting-appropriate street clothes, fully covered"
            self._scene_wardrobe[name] = outfit
            logger.info(
                f"[Wardrobe] p{page_num}: {name} dressed for "
                f"'{setting[:40]}' ({exposure}) → {outfit[:50]}"
            )

    def _llm_scene_wardrobe(self, names: List[str], setting: str,
                            exposure: str) -> Dict[str, str]:
        """Ask the LLM for setting-appropriate outfits for the named characters.

        Returns ``{name: outfit}``. On any failure returns {} so the caller uses
        its deterministic fallback. Identity is preserved by passing each
        character's locked portrait so the LLM dresses THEM, not a generic body.
        """
        try:
            char_lines = []
            for name in names:
                portrait = ''
                if self.registry is not None:
                    if hasattr(self.registry, 'get_portrait'):
                        portrait = self.registry.get_portrait(name) or ''
                    if not portrait and hasattr(self.registry, 'get_appearance'):
                        portrait = self.registry.get_appearance(name) or ''
                char_lines.append(f"  {name}: {portrait[:200]}")
            char_block = "\n".join(char_lines) if char_lines else "  (none)"

            prompt = (
                "You are a graphic-novel costume designer. Dress each character "
                "in clothing that is PLAUSIBLE and APPROPRIATE for the scene's "
                "location, while keeping their personal style and identity. "
                "People wear setting-appropriate clothing in public — a "
                "character who is undressed in a private scene is fully clothed "
                "in a store, on the street, at work, or anywhere public.\n\n"
                f"SCENE LOCATION: {setting or '(unspecified public place)'}\n"
                f"EXPOSURE CONTEXT: {exposure} (public/semi-public — fully "
                "clothed, ordinary attire for this place)\n\n"
                "CHARACTERS (identity — keep their look, just dress them):\n"
                f"{char_block}\n\n"
                "For EACH character, write a complete, fully-covered outfit "
                "appropriate to the location (garments + colours + materials, "
                "20-40 words). No nudity, no lingerie, no intimate/undressed "
                "states — this is a public scene.\n\n"
                "Return ONLY a JSON object mapping each name to its outfit:\n"
                '{ "Name": "outfit description", ... }'
            )
            raw = _llm(prompt, temperature=0.2)
            parsed = _parse(raw)
            if isinstance(parsed, dict):
                out = {}
                for k, v in parsed.items():
                    if isinstance(v, str) and v.strip():
                        # Resolve the LLM's key back to a known frame name.
                        match = _match_state_name(k, {n: None for n in names}) or k
                        out[match] = v.strip()
                return out
        except Exception as e:
            logger.warning(f"[Wardrobe] LLM dressing failed ({e}); using fallback.")
        return {}

    def _is_corporeal(self, name: str) -> bool:
        """True unless `name` is a known non-corporeal entity (a watching
        pantheon, a cosmic chorus, a disembodied spirit). Physical persistent
        conditions are never applied to non-corporeal entities."""
        return str(name).strip().lower() not in self.noncorporeal_names

    # ------------------------------------------------------------------
    def process_script(self, script: List[Dict],
                       batch_size: int = 8) -> List[Dict]:
        """Main entry point. Annotates every panel with evolved appearance state.

        Walks panels in (page, panel_index) order. For each batch it asks the
        LLM which appearance attributes change in each panel, applies those
        changes to the running per-character state, then snapshots the resolved
        state onto the panel.
        """
        # Flatten panels in narrative order, skipping act-break pages.
        ordered: List[Tuple[Dict, Dict, int, int]] = []  # (panel, page, page_num, panel_num)
        for pi, page in enumerate(script):
            if page.get('_act_break'):
                continue
            page_num = page.get('page', pi + 1)
            panels = page.get('panels', []) or []
            # Sort by panel_index so narrative order is correct even if the LLM
            # emitted panels out of order. Fall back to original order on ties.
            indexed = list(enumerate(panels))
            indexed.sort(key=lambda t: _safe_int(t[1].get('panel_index', t[0] + 1)))
            for orig_pos, panel in indexed:
                panel_num = _safe_int(panel.get('panel_index', orig_pos + 1))
                ordered.append((panel, page, page_num, panel_num))

        if not ordered:
            return script

        logger.info(
            f"[Continuity] Tracking appearance flow across {len(ordered)} panels..."
        )

        total_changes = 0
        for batch_start in range(0, len(ordered), batch_size):
            batch = ordered[batch_start: batch_start + batch_size]
            total_changes += self._process_batch(batch)

        logger.info(
            f"[Continuity] Done. {total_changes} appearance change(s) detected and "
            f"propagated forward across the sequence."
        )
        return script

    # ------------------------------------------------------------------
    def _reconcile_frame(self, panel: Dict, page_num: int) -> None:
        """Make characters_in_frame agree with who the description depicts.

        The script LLM frequently lists only one character in
        characters_in_frame while the description clearly stages two or more
        (e.g. "Sarah arches as Marcus guides her hips" but frame=[Marcus]).
        That mismatch is the root cause of impossible prompts like
        "One figure: one man" attached to a two-person description: the figure
        head-count is built from characters_in_frame, while the appearance
        clauses are built from the description — so they contradict.

        This reconciler scans the description for any KNOWN character name
        (full name or first name) and ensures every such name present in the
        description is also in characters_in_frame. It never removes names the
        script explicitly listed (the script may stage an off-page speaker),
        but it does add the missing ones so the head-count cannot undercount.
        """
        if not self.character_names:
            return
        desc = (panel.get('description') or '')
        if not desc:
            return
        desc_low = desc.lower()

        current = _chars_in_frame(panel)
        # Canonicalize existing entries so short forms ("Marcus") and full
        # names ("Marcus Vane") collapse to one, preventing duplicate figures.
        canon_current: List[str] = []
        canon_seen: set = set()
        for c in current:
            cc = self._canonical_name(c)
            if cc.lower() not in canon_seen:
                canon_current.append(cc)
                canon_seen.add(cc.lower())
        current = canon_current
        current_low = canon_seen

        added: List[str] = []
        for full_name in self.character_names:
            if not full_name:
                continue
            fl = full_name.lower()
            canonical = self._canonical_name(full_name)
            if canonical.lower() in current_low:
                continue
            # Match on full name OR first-name token, as a whole word so we
            # don't match "Sam" inside "Samuel" or "same".
            #
            # GUARD: when the "first name" token is a common English function
            # word (article/preposition/pronoun/etc.) rather than a distinctive
            # name, the first-name fallback is unsafe — e.g. "The Witch"'s
            # first token is "the", which appears in nearly every sentence in
            # the English language. Matching on it would inject The Witch into
            # almost every panel in the book regardless of whether she's
            # actually in the scene (this was happening in practice). Titled
            # names ("The Witch", "The Oracle") and any name whose first token
            # is a stopword fall back to requiring the FULL name to match.
            first = fl.split()[0] if fl.split() else fl
            if len(first) < 3 or first in _NAME_TOKEN_STOPWORDS:
                candidates = [fl]
            else:
                candidates = [fl, first]
            for cand in candidates:
                if re.search(r'\b' + re.escape(cand) + r'\b', desc_low):
                    if canonical.lower() not in current_low:
                        current.append(canonical)
                        current_low.add(canonical.lower())
                        added.append(canonical)
                    break

        # Always write back the canonicalized list (dedupes even if nothing added).
        panel['characters_in_frame'] = current
        if added:
            pid = f"p{page_num}_panel{panel.get('panel_index', '?')}"
            logger.info(
                f"[Continuity] {pid}: frame reconciled — added "
                f"{', '.join(added)} (named in description but missing from "
                f"characters_in_frame)."
            )

    def _canonical_name(self, name: str) -> str:
        """Return the registry's canonical spelling of a name if resolvable.

        Requires an UNAMBIGUOUS match. If `name`'s first token matches more
        than one registry character (e.g. two siblings sharing a surname, or
        a shared title), this returns the name unchanged rather than
        collapsing two distinct characters into one canonical key — which
        would merge their tracked appearance states and make one character's
        wardrobe/condition changes bleed onto the other.
        """
        if self.registry is not None and hasattr(self.registry, 'gender_map'):
            keys = list(getattr(self.registry, 'gender_map', {}))
            name_low = name.lower()
            # Exact case-insensitive match — always safe, no ambiguity check needed.
            for k in keys:
                if k.lower() == name_low:
                    return k
            # First-name match — require exactly one candidate, and never
            # match on a generic title/stopword token.
            name_first = name_low.split()[0] if name_low.split() else name_low
            if name_first not in _NAME_TOKEN_STOPWORDS and len(name_first) >= 2:
                matches = []
                for k in keys:
                    k_first = k.lower().split()[0] if k.lower().split() else k.lower()
                    if k_first in _NAME_TOKEN_STOPWORDS:
                        continue
                    if k_first == name_first:
                        matches.append(k)
                if len(matches) == 1:
                    return matches[0]
                # 0 or >1 matches: fall through to returning name unchanged
                # rather than guessing among multiple candidates.
        return name

    # ------------------------------------------------------------------
    def _process_batch(self, batch: List[Tuple[Dict, Dict, int, int]]) -> int:
        """Detect + apply changes for one batch of panels, in order."""
        # Reconcile each panel's frame list with its description BEFORE seeding,
        # so newly-added characters get seeded and tracked like any other.
        for panel, page, page_num, panel_num in batch:
            self._reconcile_frame(panel, page_num)

        # First, make sure every character appearing in the batch is seeded.
        for panel, page, page_num, panel_num in batch:
            for name in _chars_in_frame(panel):
                if name not in self.states:
                    self._seed_state(name, page_num)

        # Build the LLM detection prompt with the CURRENT incoming state.
        detections = self._detect_changes_llm(batch)

        applied = 0
        for panel, page, page_num, panel_num in batch:
            pid = f"p{page_num}_panel{panel_num}"
            changes_here = detections.get(pid, [])
            recorded_changes: List[Dict[str, str]] = []

            for ch in changes_here:
                cname = _match_state_name(ch.get('character', ''), self.states)
                if not cname:
                    continue
                attr = ch.get('attribute', '')
                newval = ch.get('new_value', '')
                transition = (ch.get('transition', '') or '').strip()
                st = self.states.get(cname)
                if not st:
                    continue

                # ── physical_condition — the new unified condition attribute ──
                # The LLM now detects all transient physical states (mud, blood,
                # wetness, etc.) in one pass as 'physical_condition' changes.
                # We map these back to the structured _CONDITIONS table so
                # persistence tracking, scene-reset logic, and prompt injection
                # all work identically to before.
                if attr == 'physical_condition':
                    _cond_text = newval.lower()
                    _resolved = _cond_text.startswith('resolved:')
                    if _resolved:
                        # Detect WHICH condition was resolved from the text
                        for cond in _PERSISTENT_CONDITIONS:
                            if any(re.search(r'\b' + re.escape(kw) + r'\b', _cond_text)
                                   for kw in list(cond.resolve[:4]) + [cond.name]):
                                if st.remove_condition(cond.name):
                                    if cond.category == 'dress':
                                        st.clothing = (
                                            self._scene_wardrobe.get(cname)
                                            or st.base_clothing or 'fully clothed'
                                        )
                                    applied += 1
                                    recorded_changes.append({
                                        'character': cname, 'attribute': 'condition',
                                        'transition': f'{cond.name} resolved',
                                        'new_value': f'{cond.name} cleared',
                                    })
                                    logger.info(
                                        f"[Continuity] {pid}: {cname} → "
                                        f"{cond.name} resolved (LLM-detected)."
                                    )
                    else:
                        # Acquire: match against condition names + acquire cues
                        _matched_cond = None
                        for cond in _PERSISTENT_CONDITIONS:
                            if cond.text_indicates_active(_cond_text):
                                _matched_cond = cond
                                break
                        if _matched_cond:
                            if not _exposure_allows_undress(self._current_exposure) \
                                    and _matched_cond.category == 'dress' \
                                    and not _public_undress_is_explicit(
                                        panel.get('description', '')):
                                logger.info(
                                    f"[Continuity] {pid}: suppressed "
                                    f"'{_matched_cond.name}' for {cname} "
                                    f"(public setting guard)."
                                )
                            elif st.add_condition(_matched_cond.name):
                                if _matched_cond.category == 'dress':
                                    st.clothing = 'fully nude, no clothing'
                                applied += 1
                                recorded_changes.append({
                                    'character': cname, 'attribute': 'condition',
                                    'transition': transition or f'acquiring {_matched_cond.name}',
                                    'new_value': _matched_cond.descriptor,
                                })
                                logger.info(
                                    f"[Continuity] {pid}: {cname} → "
                                    f"{_matched_cond.name} (LLM-detected)."
                                )
                        else:
                            # Store as free-text condition note for nuance
                            if st.apply_change('condition', newval):
                                applied += 1
                                recorded_changes.append({
                                    'character': cname, 'attribute': 'condition',
                                    'transition': transition,
                                    'new_value': newval,
                                })
                    continue  # physical_condition handled above; skip normal apply

                if st and st.apply_change(attr, newval):
                    applied += 1
                    recorded_changes.append({
                        'character': cname,
                        'attribute': attr,
                        'transition': transition,
                        'new_value': newval,
                    })
                    logger.info(
                        f"[Continuity] {pid}: {cname}.{attr} → '{newval[:50]}'"
                        + (f"  (transition: {transition[:40]})" if transition else "")
                    )

            # ── Scene-change reset (general) ──────────────────────────────────
            # On a hard cut to a new scene, conditions whose
            # resets_on_scene_change=True are cleared (nudity, wetness, dust,
            # sweat, tears) unless this new panel itself re-establishes them —
            # a character dries/dresses between scenes. Injuries (cuts, bruises,
            # bloody nose) do NOT reset: a wound doesn't heal across a cut.
            panel_setting = (panel.get('setting') or '').strip()
            scene_changed = bool(panel_setting) and _settings_differ(
                panel_setting, self._current_setting
            )
            desc = panel.get('description') or ''
            acquired_now, resolved_now = _scan_conditions(desc)

            # Classify exposure for THIS panel's setting. Done every panel (not
            # only on scene change) so the public-undress guard below always has
            # the correct tier even within a scene the change-detector merged.
            _panel_exposure = classify_setting_exposure(panel_setting, desc)

            if scene_changed:
                # Recompute the scene exposure and resolve setting-appropriate
                # clothing for everyone entering the new scene. The resolved
                # wardrobe becomes the reset target so a transient state (nudity,
                # wetness) clears to scene-appropriate attire, not a possibly
                # inappropriate global baseline.
                self._current_exposure = _panel_exposure
                self._resolve_scene_wardrobe(panel, page_num, _panel_exposure)
                for name in _chars_in_frame(panel):
                    st = self.states.get(name)
                    if not st:
                        continue
                    # Scene-appropriate reset target for this character: prefer
                    # the resolved scene wardrobe, then the registry baseline.
                    _scene_clothes = (
                        self._scene_wardrobe.get(name)
                        or st.base_clothing or 'fully clothed'
                    )
                    for cond_name in list(st.conditions):
                        cond = _CONDITIONS_BY_NAME.get(cond_name)
                        if (cond and cond.resets_on_scene_change
                                and cond_name not in acquired_now):
                            st.remove_condition(cond_name)
                            if cond.category == 'dress':
                                st.clothing = _scene_clothes
                            logger.info(
                                f"[Continuity] {pid}: scene change → {name} "
                                f"'{cond_name}' cleared."
                            )
                    # Even when no transient condition is active, a public-scene
                    # entry should re-clothe a character whose current clothing
                    # is an intimate/undress descriptor carried over from a soft
                    # cut the reset above didn't catch. This is the belt-and-
                    # suspenders fix for the "nude in the grocery store" case.
                    if (not _exposure_allows_undress(self._current_exposure)
                            and _clothing_is_exposed(st.clothing)
                            and not _public_undress_is_explicit(desc)):
                        st.clothing = _scene_clothes
                        st.remove_condition('nude')
                        logger.info(
                            f"[Continuity] {pid}: public scene → {name} "
                            f"re-clothed to scene-appropriate attire."
                        )
                    # General facet scene-reset: any evolving facet whose table
                    # entry sets resets_on_scene_change=True is cleared on a hard
                    # cut UNLESS this new panel re-establishes it. By default no
                    # facet resets — the governing principle is "persists until
                    # the story changes it", so a held object, a worn emotion, or
                    # a painted marking travels with the character across a cut
                    # and is only cleared by an explicit in-story change. (The
                    # field is here so a future facet can opt into resetting.)
                    for facet in _FACETS:
                        if not facet.resets_on_scene_change:
                            continue
                        cur = getattr(st, facet.name, '')
                        if cur and not facet.is_empty(cur):
                            if st.clear_facet(facet.name):
                                logger.info(
                                    f"[Continuity] {pid}: scene change → {name} "
                                    f"'{facet.name}' reset."
                                )
                self._current_setting = panel_setting
            elif panel_setting and not self._current_setting:
                self._current_setting = panel_setting
                self._current_exposure = _panel_exposure
                self._resolve_scene_wardrobe(panel, page_num, _panel_exposure)
            elif not self._current_setting:
                # No setting yet established and none on this panel: still derive
                # exposure from the description so the guard can act.
                self._current_exposure = _panel_exposure

            # ── Condition safety net (LLM-failure failsafe only) ─────────────
            # _scan_conditions is kept ONLY as a backstop for the case where
            # the entire LLM batch failed and returned {} — meaning the LLM
            # had no chance to detect conditions in this panel at all. When the
            # LLM did run (detections is non-empty for at least one panel in
            # the batch), we trust it completely: re-running the regex net on
            # top of a successful LLM pass would double-apply conditions and
            # produce spurious acquisitions on prose that the LLM correctly
            # judged as not depicting a condition change.
            _llm_ran = bool(detections)  # True when the LLM returned ANY result
            if not _llm_ran:
                # LLM batch completely failed — fall back to regex safety net
                # to avoid losing all condition tracking for this panel.
                for name in _chars_in_frame(panel):
                    st = self.states.get(name)
                    if not st:
                        continue
                    _corporeal = self._is_corporeal(name)
                    for cond_name in resolved_now:
                        if st.remove_condition(cond_name):
                            cond = _CONDITIONS_BY_NAME.get(cond_name)
                            if cond and cond.category == 'dress':
                                st.clothing = (
                                    self._scene_wardrobe.get(name)
                                    or st.base_clothing or 'fully clothed'
                                )
                            applied += 1
                            recorded_changes.append({
                                'character': name, 'attribute': 'condition',
                                'transition': f'{cond_name} resolved',
                                'new_value': f'{cond_name} cleared',
                            })
                            logger.info(
                                f"[Continuity] {pid}: {name} → {cond_name} "
                                f"resolved (regex failsafe — LLM batch failed)."
                            )
                    for cond_name in acquired_now:
                        if cond_name in resolved_now:
                            continue
                        if not _corporeal:
                            continue
                        cond = _CONDITIONS_BY_NAME.get(cond_name)
                        if (cond and cond.category == 'dress'
                                and not _exposure_allows_undress(self._current_exposure)
                                and not _public_undress_is_explicit(desc)):
                            continue
                        if st.add_condition(cond_name):
                            if cond and cond.category == 'dress':
                                st.clothing = 'fully nude, no clothing'
                            applied += 1
                            recorded_changes.append({
                                'character': name, 'attribute': 'condition',
                                'transition': f'acquiring {cond_name}',
                                'new_value': (cond.descriptor if cond else cond_name),
                            })
                            logger.info(
                                f"[Continuity] {pid}: {name} → {cond_name} "
                                f"(regex failsafe — LLM batch failed)."
                            )

            # ── Facet-release safety net (LLM-failure failsafe only) ──────────
            # Only runs when the LLM batch completely failed. When the LLM ran,
            # it is responsible for detecting facet releases (held item set down,
            # marking wiped off) via direct attribute changes in its output.
            if not _llm_ran:
                for facet in _FACETS:
                    if not (facet.release_cues or facet.release_patterns):
                        continue
                    if not facet.scan_release(desc):
                        continue
                    for name in _chars_in_frame(panel):
                        st = self.states.get(name)
                        if not st:
                            continue
                        cleared = getattr(st, facet.name, '')
                        if cleared and st.clear_facet(facet.name):
                            applied += 1
                            recorded_changes.append({
                                'character': name, 'attribute': facet.name,
                                'transition': f'removing / setting aside the {cleared[:40]}',
                                'new_value': 'none (cleared)',
                            })
                            logger.info(
                                f"[Continuity] {pid}: {name}.{facet.name} cleared "
                                f"'{cleared[:40]}' (regex failsafe — LLM batch failed)."
                            )

            # Snapshot the resolved CURRENT state for every character in frame.
            state_snapshot: Dict[str, Dict[str, object]] = {}
            for name in _chars_in_frame(panel):
                st = self.states.get(name)
                if st:
                    state_snapshot[name] = st.snapshot()

            panel['_appearance_state'] = state_snapshot
            panel['_appearance_changes'] = recorded_changes

        # Refresh the inter-batch context buffer with the tail of THIS batch so
        # the next batch can see the events that led into its incoming state.
        tail = []
        for panel, page, page_num, panel_num in batch:
            desc = (panel.get('description') or '').strip()
            if desc:
                tail.append((f"p{page_num}_panel{panel_num}", desc))
        if tail:
            self._recent_descriptions = tail[-3:]

        return applied

    # ------------------------------------------------------------------
    def _detect_changes_llm(
        self, batch: List[Tuple[Dict, Dict, int, int]]
    ) -> Dict[str, List[Dict[str, str]]]:
        """Ask the LLM to detect ALL appearance state changes in each panel.

        This is the single source of truth for appearance state — it replaces
        both the previous LLM change-detection pass AND the deterministic
        _scan_conditions regex safety-net. By unifying them into one prompt
        we give the LLM full context to reason about:
          - Clothing / costume changes (including undress / re-dress)
          - Hair arrangement changes
          - Transient physical conditions (mud, blood, wetness, dust, sweat,
            tears — their acquisition AND resolution)
          - Facet releases (held item set down, marking wiped off)
          - Emotion register shifts
          - Persistent-condition resolution (bandaging a cut, drying off)

        The LLM reads the full incoming state so it never drops a condition
        that was established in an earlier panel but not re-mentioned here.

        Returns {panel_id: [ {character, attribute, new_value, transition}, ... ]}.
        On any failure returns {} (state carries forward unchanged — safe).
        """
        # Compose the incoming-state block + the panel descriptions.
        # Build a richly annotated state block for the LLM so it has the full
        # current picture of each character — clothing, conditions, facets —
        # and can detect both new acquisitions AND explicit resolutions.
        state_lines: List[str] = []
        seen_in_batch: set = set()
        for panel, page, page_num, panel_num in batch:
            for name in _chars_in_frame(panel):
                if name in seen_in_batch:
                    continue
                seen_in_batch.add(name)
                st = self.states.get(name)
                if not st:
                    continue
                cur = []
                # Active persistent conditions — must persist unless explicitly
                # resolved. Show the human-readable descriptor so the LLM knows
                # what "wet" or "muddy" actually looks like in the art.
                active = set(st.conditions)
                if active:
                    labelled = []
                    for cn in sorted(active):
                        cobj = _CONDITIONS_BY_NAME.get(cn)
                        labelled.append(
                            f"{cn} [{cobj.descriptor}]" if cobj else cn
                        )
                    cur.append("ACTIVE CONDITIONS (persist until resolved): "
                                + "; ".join(labelled))
                # Mutable facets — show each with its current value
                if st.naturally_unclothed:
                    # Tell the LLM explicitly so it never reports this entity as
                    # "nude" and applies the right rules for clothing changes.
                    cur.append("NATURALLY UNCLOTHED (animal/entity — their natural "
                               "appearance, not a garment; nudity condition does not apply)")
                if st.clothing and 'nude' not in active:
                    label = "natural_covering" if st.naturally_unclothed else "clothing"
                    cur.append(f"{label}: {st.clothing}")
                    # Expose prior clothing values so the LLM regression guard
                    # can detect shirt→suit→shirt reversions.
                    if len(st._clothing_history) > 1:
                        hist = " → ".join(f'"{v}"' for v in st._clothing_history[:-1])
                        cur.append(f"clothing_history (oldest→previous): {hist}")
                elif 'nude' in active and not st.naturally_unclothed:
                    cur.append("clothing: NONE (character is fully nude)")
                if st.hair:
                    cur.append(f"hair: {st.hair}")
                    if len(st._hair_history) > 1:
                        hist = " → ".join(f'"{v}"' for v in st._hair_history[:-1])
                        cur.append(f"hair_history (oldest→previous): {hist}")
                if st.closure:
                    cur.append(f"closure: {st.closure}")
                    # Expose prior closure values so the LLM regression guard can
                    # detect buttoned→unbuttoned→buttoned flip-flops that the
                    # story never actually depicts.
                    if len(st._closure_history) > 1:
                        hist = " → ".join(f'"{v}"' for v in st._closure_history[:-1])
                        cur.append(f"closure_history (oldest→previous): {hist}")
                if st.emotion:
                    cur.append(f"emotion: {st.emotion}")
                if st.markings:
                    cur.append(f"markings: {st.markings}")
                if st.held_items:
                    cur.append(f"held_items: {st.held_items}")
                if st.condition:
                    cur.append(f"condition_note: {st.condition}")
                if st.permanent_markings:
                    cur.append(
                        f"PERMANENT MARKS (always present, never clear): "
                        f"{st.permanent_markings}"
                    )
                state_lines.append(
                    f"  {name}: " + ("; ".join(cur) if cur else "(fresh baseline)")
                )
        state_block = "\n".join(state_lines) if state_lines else "  (no characters yet)"

        panel_items: List[str] = []
        for panel, page, page_num, panel_num in batch:
            desc = (panel.get('description') or '').strip()
            chars = ", ".join(_chars_in_frame(panel))
            panel_items.append(
                f'{{"id": "p{page_num}_panel{panel_num}", '
                f'"characters": {_json_str(chars)}, '
                f'"description": {_json_str(desc)}}}'
            )
        panels_block = "\n".join(panel_items)

        # Inter-batch context — the tail descriptions of the PREVIOUS batch, so
        # the LLM can recognise when this batch's prose references an event that
        # already happened (e.g. "the jacket he put on earlier") rather than
        # flagging it as a suspicious reversion.
        recent_block = ""
        if self._recent_descriptions:
            recent_lines = "\n".join(
                f"  {pid}: {desc[:200]}"
                for pid, desc in self._recent_descriptions
            )
            recent_block = (
                "\nRECENT PRECEDING PANELS (context only — already processed, "
                "do NOT emit changes for these; use them to understand references "
                "in the current batch):\n" + recent_lines + "\n"
            )

        # Build the structured condition reference so the LLM knows exactly what
        # each condition means in visual terms — it uses these to decide when a
        # description is acquiring or resolving a physical state.
        conditions_ref = "\n".join(
            f"  {c.name}: ACQUIRED by [{', '.join(c.acquire[:4])}...]; "
            f"RESOLVED by [{', '.join(c.resolve[:3])}...]; "
            f"scene-reset={'yes' if c.resets_on_scene_change else 'no'}"
            for c in _PERSISTENT_CONDITIONS
        )
        facets_ref = "\n".join(
            f"  {f.name}: release cues [{', '.join(list(f.release_cues)[:3])}...]"
            for f in _FACETS if f.release_cues
        )

        prompt = f"""You are the Continuity Supervisor for a graphic novel.
Your job: in ONE pass, detect ALL appearance state changes in each panel — clothing,
physical conditions (mud, blood, wetness, etc.), held items, emotion, hair, and markings.
You are the SOLE source of truth for all appearance state; there is no separate safety net.

This means you MUST catch:
  1. New conditions acquired (blood, wetness, mud, bruises, tears, sweat, dust)
  2. Conditions resolved (wound bandaged, washed off, dried)
  3. Clothing changes — including full undress AND re-dressing
  4. Held items picked up OR set down / handed off / holstered
  5. Emotion register established or shifted
  6. Markings applied OR removed
  7. Hair changes

GOVERNING PRINCIPLE — EVERYTHING IS STICKY:
Once a character acquires a state it PERSISTS in every following panel until explicitly
changed or resolved. Never drop a state just because a later panel omits it. Silence = unchanged.

CURRENT STATE entering this batch (carry everything forward):
{state_block}
{recent_block}
PANELS IN NARRATIVE ORDER (process top to bottom; state flows forward):
{panels_block}

=== PHYSICAL CONDITIONS REFERENCE ===
These are the trackable physical conditions. Detect acquisition AND resolution:
{conditions_ref}

CONDITION RULES:
- Acquisition: when the description clearly depicts the event (fell in mud, got punched, emerged from water)
- Resolution: when the description shows it being explicitly fixed (wiped off, bandaged, dried)
- Partial undress is NOT full nudity: "shirtless" / "bare-chested" / "removes shirt" → partial state.
  Only report full nudity when unambiguously nude with NOTHING on.
- Public-setting guard: undress/nudity in clearly public places (stores, streets, offices) is almost
  always a mis-read. If uncertain, report the more clothed state.
- If an active condition already in the CURRENT STATE is not resolved in this panel, do NOT re-report it.

=== NATURALLY UNCLOTHED ENTITIES ===
Some characters in the CURRENT STATE block are marked NATURALLY UNCLOTHED (animals, angels,
divine beings, creatures).  For these characters:
- Their "natural_covering" field (fur, feathers, scales, radiance…) is their permanent natural
  appearance — it is NOT a garment and should NEVER be reported as a clothing change.
- Do NOT apply the nudity condition to them — a dog without a coat is not "nude".
- A garment ADDED to a naturally-unclothed character (a service-dog vest, angelic armour) IS
  a genuine clothing change — report it normally.
- If a garment previously added is later removed (the vest is taken off), report that change.
- Their natural covering changes only if the story explicitly depicts it changing (the dragon
  sheds its scales, the wolf moults) — an extremely rare event that will be unambiguous.

=== CONTROLLED WARDROBE TRANSITIONS (ALWAYS REPORT THESE) ===
The locks that keep a character's look consistent must YIELD to any real
in-story change. When a panel actually depicts clothing being added, removed, or
altered — for ANY reason — REPORT it as a "clothing" change so the transition
renders and the new state carries forward. This includes:

- SELF-INITIATED changes: a character takes off a coat on entering a warm room,
  removes a jacket, kicks off shoes, undresses for a sauna or bath, pulls on a
  sweater, changes outfits, puts up a hood.
- PASSIVE / ENVIRONMENTAL changes: the wind blows a hat off, a gust tears away a
  scarf, a wave soaks and strips a garment, a branch snags and rips off a cloak,
  someone ELSE removes a character's coat or hat for them (a host taking a guest's
  coat, a nurse cutting away a sleeve). No deliberate self-action is required — if
  the garment leaves or changes, it changed.
- PARTIAL removals: report the SPECIFIC garment that left. new_value = the
  resulting full outfit MINUS that item (e.g. current "wool coat over a blue
  jumper" + "she hangs her coat by the door" → new_value "a blue jumper"; current
  "denim jacket and a red cap" + "the wind snatches his cap away" → new_value
  "a denim jacket"). If EVERYTHING comes off (sauna, skinny-dip), use the nudity
  condition instead.

These are the changes we WANT. Reporting them is how the art shows the moment and
how the character stays consistent afterward. The regression guards below only
reject a RE-APPEARANCE of a garment/state the character already left WITHOUT the
story showing them put it back — they never suppress a genuine forward removal or
addition like the ones above.

=== CLOTHING AND HAIR REGRESSION GUARD ===
The CURRENT STATE block above shows each character's current clothing and hair,
plus a clothing_history / hair_history trail of every prior value (oldest → previous).
Use this to detect and REJECT false reversions:

- A "clothing" change whose new_value closely resembles a value from clothing_history
  (same garments, same colours) is a REGRESSION — the character would be reverting to
  an earlier outfit without any story justification. DO NOT report it as a change.
  The description most likely re-mentions clothing incidentally (e.g. "he straightened
  his collar") without depicting an actual change back to the old outfit.

- The same rule applies to "hair": if a proposed new hair value matches a prior value
  in hair_history, it is a regression. Only report it if the description explicitly
  shows the character re-doing their hair back to that earlier state.

- A reversion IS legitimate when the panel description explicitly shows the change
  happening (e.g. "she changes back into her shirt", "he removes the jacket he put on
  earlier"). In that case, report it normally with a clear transition value.

- When in doubt between "genuine change" and "incidental re-mention / regression",
  default to NO CHANGE. Silence = unchanged.

=== CLOSURE / FASTENING STATE GUARD ===
"closure" tracks the fastening SUB-STATE of the garment a character is already
wearing — buttoned vs unbuttoned/open, zipped vs unzipped, sleeves rolled up vs
down, collar popped vs flat, shirt tucked vs untucked. This is the SAME garment
throughout; only how it is fastened changes.

- Report a "closure" change ONLY when the panel description EXPLICITLY depicts the
  action: "unbuttons his shirt", "buttons her coat", "rolls up his sleeves",
  "loosens his tie", "zips up the jacket", "tucks in his shirt", "pops the
  collar". new_value = the resulting closure state (e.g. "shirt unbuttoned to the
  chest", "sleeves rolled to the elbow", "coat buttoned to the collar",
  "shirt untucked"). To clear it back to as-drawn, new_value = "none".

- Do NOT invent a closure change from an incidental re-mention. "He adjusted his
  collar", "she smoothed her shirt", "his jacket hung open" (already established)
  are NOT changes — they merely restate the current state.

- REGRESSION GUARD: the CURRENT STATE block shows each character's current closure
  plus a closure_history trail. A proposed closure value that matches an earlier
  value in closure_history (e.g. buttoned → unbuttoned → **buttoned again**) is a
  REGRESSION and must NOT be reported UNLESS this very panel explicitly shows the
  character re-doing that action (re-buttoning, re-zipping, rolling the sleeves
  back down). A shirt does not silently re-fasten itself between panels — that is
  exactly the uncontrolled flicker we are preventing. When in doubt, NO CHANGE.

=== FACET RELEASE REFERENCE ===
These facets have release cues — detect when a character explicitly sets an item down etc:
{facets_ref}

=== WHAT TO REPORT ===
Report a change entry ONLY when the panel description actually depicts the change.
Attributes: clothing | hair | condition | held_items | emotion | markings | physical_condition | permanent_markings | closure
  - Use "physical_condition" for transient physical states (mud, blood, wetness, etc.).
    new_value = full description of the resulting condition state.
    For resolution: new_value = "resolved: <what happened>" (e.g. "resolved: wound bandaged")
  - For held_items release: new_value = "nothing (hands free)"
  - For emotion: new_value = the resulting register (or "neutral" to clear)
  - For markings removal: new_value = "none"
  - Use "closure" for the fastening sub-state of the CURRENT garment (buttoned/
    unbuttoned, zipped/unzipped, sleeves rolled/down, tucked/untucked, collar
    popped). new_value = the resulting state (e.g. "shirt unbuttoned to the
    chest", "sleeves rolled to the elbow"); "none" to clear back to as-drawn.
    Report ONLY when the panel explicitly shows the buttoning/unbuttoning/etc.
  - Use "permanent_markings" for lasting, irreversible body changes: a significant scar
    from a wound that healed, a tattoo, a missing limb, a prosthetic, a brand. Only
    report when the panel EXPLICITLY shows the character ACQUIRING this mark (not merely
    having it — it persists automatically once set). new_value = concise visual description
    of the mark and its location (e.g. "jagged scar across the left cheek, pale and raised").

Give the FULL resulting state for each attribute (not just the delta).
'transition' = the visible action happening in this panel ("in the act of peeling off the jacket");
              leave "" if the panel shows an already-established state.

Return ONLY a JSON array — no markdown, no preamble:
[
  {{"id": "p3_panel2", "changes": [
      {{"character": "Yuki", "attribute": "clothing",
        "new_value": "wearing only the grey tank top, jacket removed",
        "transition": "in the act of peeling the jacket off her shoulders"}},
      {{"character": "Marcus", "attribute": "physical_condition",
        "new_value": "blood trickling from a cut above the left eyebrow",
        "transition": "recoiling from the punch"}}
  ]}},
  ...
]
Only include panels that have changes. IDs must match exactly.
"""

        try:
            raw = _llm(prompt, temperature=0.15)
            parsed = _parse(raw)
            if not isinstance(parsed, list):
                logger.warning("[Continuity] LLM returned non-list; carrying state forward.")
                return {}
            result: Dict[str, List[Dict[str, str]]] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                pid = item.get('id', '')
                changes = item.get('changes', [])
                if pid and isinstance(changes, list):
                    clean = [c for c in changes if isinstance(c, dict) and c.get('attribute')]
                    if clean:
                        result[pid] = clean
            return result
        except Exception as e:
            logger.warning(f"[Continuity] Change-detection batch failed: {e}; carrying state forward.")
            return {}


# ---------------------------------------------------------------------------
# Module-level helpers for the tracker
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LLM-backed scene-boundary detector
# ---------------------------------------------------------------------------
# The old Jaccard-overlap heuristic misses soft cuts (bedroom → kitchen of the
# same apartment shares "the/room/apartment" vocabulary → false negative) and
# misfires on varied re-descriptions of the same place. The LLM reads the two
# setting strings as a human editor would and gives a definitive yes/no.
#
# A fast Jaccard pre-check is kept as a tier-0 short-circuit:
#   • Identical strings         → same scene (True negative)
#   • Very high overlap (≥0.65) → same scene (high-confidence skip)
#   • Very low overlap (≤0.12)  → different scene (high-confidence fire)
#   • 0.12 < overlap < 0.65    → defer to LLM
# Results are cached per (a_normalised, b_normalised) pair so each unique pair
# is evaluated only once per synthesis run, even when the tracker visits the
# same setting boundary many times.
# ---------------------------------------------------------------------------

_SCENE_BOUNDARY_CACHE: Dict[Tuple[str, str], bool] = {}


def _settings_differ(a: str, b: str) -> bool:
    """Return True when two setting strings denote clearly different scenes.

    Uses a fast Jaccard pre-screen for high-confidence cases and delegates
    ambiguous cases to the LLM. Cached per pair so re-evaluation is free.
    """
    a_n = (a or '').strip().lower()
    b_n = (b or '').strip().lower()
    if not a_n or not b_n:
        return False
    if a_n == b_n:
        return False

    cache_key = (a_n, b_n) if a_n < b_n else (b_n, a_n)
    if cache_key in _SCENE_BOUNDARY_CACHE:
        return _SCENE_BOUNDARY_CACHE[cache_key]

    # Tier-0: Jaccard pre-screen
    _stop = {
        'the', 'a', 'an', 'in', 'on', 'at', 'of', 'and', 'with', 'to', 'into',
        'near', 'by', 'inside', 'outside', 'over', 'under', 'is', 'are', 'as',
        'scene', 'shot', 'view', 'interior', 'exterior', 'int', 'ext',
    }
    def _toks(s: str) -> set:
        return {w for w in re.findall(r'[a-z0-9]+', s) if w not in _stop and len(w) > 2}
    ta, tb = _toks(a_n), _toks(b_n)
    if ta and tb:
        overlap = len(ta & tb) / max(1, len(ta | tb))
        if overlap >= 0.65:   # clearly same place, different description
            _SCENE_BOUNDARY_CACHE[cache_key] = False
            return False
        if overlap <= 0.12:   # clearly different place
            _SCENE_BOUNDARY_CACHE[cache_key] = True
            return True

    # Tier-1: LLM judgment for ambiguous cases
    result = _llm_scene_boundary(a, b)
    _SCENE_BOUNDARY_CACHE[cache_key] = result
    return result


def _llm_scene_boundary(a: str, b: str) -> bool:
    """Ask the LLM whether two setting descriptions denote different scenes.

    Returns True (different scene → reset transient states) or False (same
    scene — no reset). Falls back to True (conservative) on failure so we
    never fail to reset when we should.
    """
    try:
        prompt = (
            "You are a comic-book continuity supervisor determining whether two "
            "setting descriptions refer to DIFFERENT scenes or to the SAME scene "
            "described in different ways.\n\n"
            f"Setting A: {a}\n"
            f"Setting B: {b}\n\n"
            "Answer with ONLY a JSON object, no prose:\n"
            "{\"different_scene\": true} if clearly different locations/times.\n"
            "{\"different_scene\": false} if plausibly the same place.\n\n"
            "DIFFERENT signals: distinct named locations, clear time jump, "
            "indoor to outdoor, different building or room type.\n"
            "SAME signals: close-up or re-angle of the same room, same "
            "vocabulary, continuous scene from a new angle.\n"
            "When uncertain, lean toward false to avoid spurious resets."
        )
        raw = _llm(prompt, temperature=0.0, large=False)
        parsed = _parse(raw)
        if isinstance(parsed, dict) and 'different_scene' in parsed:
            return bool(parsed['different_scene'])
    except Exception as e:
        logger.warning(f"[SceneBoundary] LLM call failed ({e}); defaulting to True.")
    return True


# ===========================================================================
# SCENE WARDROBE RESOLUTION — setting-appropriate clothing per scene
# ---------------------------------------------------------------------------
# Problem it solves
# -----------------
# The AppearanceContinuityTracker keeps clothing *consistent* (no flicker) and
# resets transient states (nudity, wetness) on a hard cut — but it resets to a
# single fixed registry baseline, and it has no notion of whether a given state
# is APPROPRIATE for the current location. Two failure modes follow:
#
#   1. A character is nude/undressed in an intimate scene (a bedroom), then the
#      story hard-cuts to a PUBLIC place (a grocery store, a street, an office).
#      If the scene-change reset misfires (settings share vocabulary, or the
#      cut is "soft"), the nudity persists into the public scene — a character
#      drawn nude in a grocery store.
#   2. The registry baseline outfit itself may be scene-inappropriate. In a
#      "steamy drama" a character's baseline might be lingerie or a robe; when
#      nudity resets, it falls back to THAT baseline even in a public setting.
#
# The general principle the user articulated: a character's depicted clothing
# must be a function of the SCENE (its location + social context), not just a
# fixed baseline plus continuity. A character intimate at home is clothed at
# the store. This is true across every genre — it is about plausibility, not
# censorship: people wear setting-appropriate clothing in public.
#
# The fix: a per-scene wardrobe resolver. On each scene change it (a) classifies
# the setting's exposure context (public / semi-public / private / intimate),
# and (b) computes a setting-appropriate clothing baseline for each character in
# the scene. That scene baseline becomes the reset target, and a guard prevents
# undress/nudity from being acquired in a clearly PUBLIC setting unless the
# description gives strong, explicit, plausible justification (a locker room, a
# medical exam, a nude beach — places where it is contextually normal).
#
# Design: deterministic keyword classification first (fast, no tokens), with an
# optional LLM enrichment per UNIQUE setting (cached) for ambiguous cases. If
# the LLM is unavailable the keyword classifier alone is safe and useful.
# ===========================================================================

# Exposure tiers, ordered from most to least public. A character's undress is
# only plausible without explicit justification at 'private'/'intimate'.
_EXPOSURE_PUBLIC      = 'public'        # store, street, office, school, restaurant
_EXPOSURE_SEMIPUBLIC  = 'semi_public'   # lobby, hallway, party, gym floor, bar
_EXPOSURE_PRIVATE     = 'private'       # home, apartment, hotel room, car
_EXPOSURE_INTIMATE    = 'intimate'      # bedroom, bathroom, bath, shower, bed
_EXPOSURE_UNKNOWN     = 'unknown'

# Settings where undress/nudity IS contextually normal even though they are not
# someone's private home — explicit, plausible exceptions to the public guard.
_EXPOSURE_NUDE_OK_CUES = (
    'locker room', 'changing room', 'change room', 'dressing room', 'fitting room',
    'shower', 'showers', 'bath', 'bathhouse', 'bathroom', 'sauna', 'steam room',
    'spa', 'hot spring', 'onsen', 'nude beach', 'naturist', 'skinny dip',
    'examination room', 'exam room', 'operating room', 'morgue', 'autopsy',
    'life drawing', 'art studio', 'massage', 'pool', 'swimming',
)

# Keyword cues for each exposure tier. First match (most public wins on ties via
# ordering of the checks below) determines the tier.
_EXPOSURE_PUBLIC_CUES = (
    'grocery', 'supermarket', 'market', 'store', 'shop', 'mall', 'street',
    'sidewalk', 'road', 'avenue', 'plaza', 'square', 'park', 'office',
    'workplace', 'school', 'classroom', 'university', 'campus', 'library',
    'restaurant', 'cafe', 'café', 'diner', 'courtroom', 'court', 'hospital',
    'clinic', 'station', 'airport', 'train', 'bus', 'subway', 'platform',
    'church', 'temple', 'mosque', 'museum', 'gallery', 'stadium', 'arena',
    'parking lot', 'crosswalk', 'bank', 'pharmacy', 'checkout', 'aisle',
    'marketplace', 'bazaar', 'festival', 'parade', 'public',
)
_EXPOSURE_SEMIPUBLIC_CUES = (
    'lobby', 'hallway', 'corridor', 'elevator', 'stairwell', 'foyer',
    'reception', 'waiting room', 'bar', 'pub', 'club', 'lounge', 'party',
    'gym', 'rooftop', 'balcony', 'porch', 'garden', 'backyard', 'courtyard',
    'hotel lobby', 'office kitchen', 'break room',
)
_EXPOSURE_PRIVATE_CUES = (
    'apartment', 'flat', 'house', 'home', 'living room', 'kitchen', 'study',
    'den', 'office at home', 'hotel room', 'motel', 'cabin', 'car', 'vehicle',
    'studio apartment', 'loft', 'basement', 'attic', 'garage', 'trailer',
    "someone's home", 'her place', 'his place', 'their place',
)
_EXPOSURE_INTIMATE_CUES = (
    'bedroom', 'bed', 'master suite', 'boudoir', 'bathtub',
    'en suite', 'ensuite', 'hotel suite', 'honeymoon',
)


def classify_setting_exposure(setting: str, description: str = "") -> str:
    """Classify a scene's exposure context from its setting (+ description).

    Returns one of _EXPOSURE_PUBLIC / _SEMIPUBLIC / _PRIVATE / _INTIMATE /
    _UNKNOWN. Deterministic keyword match — fast and token-free. The intimate
    and nude-OK cues are checked first (most specific), then private, then the
    public/semi-public cues; this ordering means a "hotel room bathroom" reads
    as intimate, not public, while a "grocery store" reads as public.

    `description` is scanned in addition to `setting` so a scene whose setting
    string is vague ("a large room") but whose description names a checkout
    counter still classifies correctly.
    """
    blob = f"{setting or ''} {description or ''}".lower()
    if not blob.strip():
        return _EXPOSURE_UNKNOWN

    def _has(cues) -> bool:
        return any(cue in blob for cue in cues)

    # Nude-OK contexts and intimate spaces are the most specific — check first.
    if _has(_EXPOSURE_NUDE_OK_CUES):
        # These are places undress is normal; treat as intimate for wardrobe
        # purposes (undress allowed) even if technically semi-public.
        return _EXPOSURE_INTIMATE
    if _has(_EXPOSURE_INTIMATE_CUES):
        return _EXPOSURE_INTIMATE
    # Public cues win over private when both somehow appear (a "store" reference
    # dominates), because the cost of mistakenly allowing undress in public is
    # far higher than the reverse.
    if _has(_EXPOSURE_PUBLIC_CUES):
        return _EXPOSURE_PUBLIC
    if _has(_EXPOSURE_SEMIPUBLIC_CUES):
        return _EXPOSURE_SEMIPUBLIC
    if _has(_EXPOSURE_PRIVATE_CUES):
        return _EXPOSURE_PRIVATE
    return _EXPOSURE_UNKNOWN


def _exposure_allows_undress(exposure: str) -> bool:
    """True if undress/nudity is contextually plausible in this exposure tier
    WITHOUT explicit strong justification. Private and intimate spaces qualify;
    public, semi-public, and unknown do not (unknown errs on the safe side)."""
    return exposure in (_EXPOSURE_PRIVATE, _EXPOSURE_INTIMATE)


# Strong, explicit undress cues that may override the public-setting guard when
# the story genuinely depicts public undress as a deliberate beat (a streaker, a
# public protest, a nightmare). These are intentionally narrow: a bare verb of
# undressing in a public place is NOT enough; the description must make the
# public exposure itself the explicit subject.
_PUBLIC_UNDRESS_OVERRIDE_CUES = (
    'streak', 'streaker', 'streaking', 'public nudity', 'naked in public',
    'nude in public', 'strips in front of everyone', 'strips in front of the crowd',
    'flashes the crowd', 'bares herself to the crowd', 'bares himself to the crowd',
    'naked protest', 'nude protest',
)


def _public_undress_is_explicit(description: str) -> bool:
    """True only when a public scene's description makes public undress its
    explicit, deliberate subject — the narrow exception to the public guard."""
    d = (description or '').lower()
    return any(cue in d for cue in _PUBLIC_UNDRESS_OVERRIDE_CUES)


# Clothing descriptors that denote an exposed / undressed / intimate state. Used
# to detect when a character's CURRENT clothing string (carried across a soft
# cut) is inappropriate for a public scene and must be replaced. Kept narrow and
# explicit so ordinary attire ("a low-cut dress", "shorts") is never flagged —
# only genuinely undressed/intimate states.
_EXPOSED_CLOTHING_CUES = (
    'nude', 'naked', 'no clothing', 'no clothes', 'unclothed', 'undressed',
    'bare body', 'fully nude', 'completely nude', 'topless', 'bottomless',
    'in nothing but', 'wearing nothing', 'just a towel', 'only a towel',
    'just underwear', 'only underwear', 'in lingerie', 'lingerie only',
    'in only a robe', 'open robe', 'just a bra', 'just panties',
    'in only her underwear', 'in only his underwear',
)


def _clothing_is_exposed(clothing: str) -> bool:
    """True if a clothing descriptor denotes an undressed / intimate state that
    would be inappropriate in a public setting."""
    c = (clothing or '').lower()
    return any(cue in c for cue in _EXPOSED_CLOTHING_CUES)





def _safe_int(v, default: int = 1) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        digits = ''.join(c for c in str(v) if c.isdigit())
        return int(digits) if digits else default


def _chars_in_frame(panel: Dict) -> List[str]:
    """Normalise characters_in_frame to a list of plain names."""
    raw = panel.get('characters_in_frame', []) or []
    out: List[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            nm = (entry.get('name') or entry.get('character')
                  or entry.get('character_name') or '')
        else:
            nm = str(entry) if entry is not None else ''
        nm = nm.strip()
        if nm:
            out.append(nm)
    return out


def _match_state_name(candidate: str, states: Dict[str, 'CharacterAppearanceState']) -> str:
    """Fuzzily match an LLM-returned character name to a tracked state key.

    Requires the match to be UNAMBIGUOUS. If `candidate` could plausibly refer
    to more than one tracked character (e.g. two characters share a surname —
    "Lopez" matching both "Maria Lopez" and "Diego Lopez" — or a shared
    title/rank like "Captain"), this returns '' (no match) rather than
    guessing. Guessing wrong here means one character's clothing/condition
    change gets silently applied to a DIFFERENT character — exactly the bug
    behind characters appearing to swap identities mid-scene even though the
    scene never changed.
    """
    candidate = (candidate or '').strip()
    if not candidate:
        return ''
    if candidate in states:
        return candidate
    cl = candidate.lower()
    # Exact case-insensitive — require a single match.
    ci_matches = [k for k in states if k.lower() == cl]
    if len(ci_matches) == 1:
        return ci_matches[0]
    elif len(ci_matches) > 1:
        return ''

    # First-name / title-token match — never match on a generic stopword/title
    # token, and require exactly one candidate.
    cl_first = cl.split()[0] if cl.split() else cl
    first_token_matches = []
    if cl_first not in _NAME_TOKEN_STOPWORDS and len(cl_first) >= 2:
        for k in states:
            kl = k.lower()
            k_first = kl.split()[0] if kl.split() else kl
            if k_first in _NAME_TOKEN_STOPWORDS:
                continue
            if k_first == cl_first or kl == cl_first or cl == k_first:
                first_token_matches.append(k)
    if len(first_token_matches) == 1:
        return first_token_matches[0]
    elif len(first_token_matches) > 1:
        return ''

    # Substring containment — require exactly one candidate, and skip
    # trivially short candidates (matches almost anything).
    if len(cl) >= 3:
        substring_matches = [k for k in states if cl in k.lower() or k.lower() in cl]
        if len(substring_matches) == 1:
            return substring_matches[0]
        elif len(substring_matches) > 1:
            return ''

    return ''


def track_appearance_continuity(
    script: List[Dict],
    characters: list,
    registry=None,
    visual_bible: Optional[SeriesVisualBible] = None,
    graph=None,
) -> List[Dict]:
    """Entry point: annotate every panel with evolved per-character appearance.

    Run this in generate_comic_script AFTER art_director_review_panels (so it
    sees the disambiguated descriptions) and BEFORE image generation. It mutates
    the script in place and also returns it.

    When ``graph`` (the EI CharacterGraph) is supplied, each character's tangible
    identity signature (scar/tattoo/distinctive feature) is seeded into the
    tracker so it is locked as a permanent marking from panel one.

    Each story panel gains:
      panel['_appearance_state']   - resolved CURRENT appearance per visible char
      panel['_appearance_changes'] - changes happening in THIS panel (transitions)
    """
    try:
        names = []
        noncorporeal = []
        naturally_unclothed = []
        _NONCORP_MARKERS = (
            'mythic', 'cosmic', 'celestial', 'divine', 'deity', 'deities',
            'god', 'goddess', 'pantheon', 'chorus', 'observer', 'spirit',
            'eternal', 'primordial', 'incorporeal', 'ethereal', 'phantom',
            'apparition', 'presence', 'angel', 'demon', 'ghost', 'entity',
            'abstract', 'force',
        )
        for c in (characters or []):
            nm = getattr(c, 'name', None) or (c.get('name') if isinstance(c, dict) else None)
            if not nm:
                continue
            nm = str(nm).strip()
            names.append(nm)
            # Pool the descriptive fields and check for non-corporeal markers.
            def _g(attr):
                v = getattr(c, attr, None)
                if v is None and isinstance(c, dict):
                    v = c.get(attr)
                if isinstance(v, (list, tuple)):
                    return ' '.join(str(x) for x in v)
                return str(v or '')
            blob = ' '.join(_g(a) for a in
                            ('role', 'mythic_archetype', 'age', 'backstory')).lower()
            blob += ' ' + ' '.join(_g(a) for a in ('traits',)).lower()
            blob += ' ' + nm.lower()
            # Word-boundary match so substrings don't false-trigger
            # (e.g. "evangelist" must not match "angel").
            if any(re.search(rf'\b{re.escape(m)}\b', blob) for m in _NONCORP_MARKERS):
                noncorporeal.append(nm)
            # Detect naturally-unclothed entities (animals, certain supernatural
            # beings) using the module-level helper which scans role, appearance,
            # name, and traits. These get special clothing-lock handling.
            if _entity_is_naturally_unclothed(c):
                naturally_unclothed.append(nm)
        if noncorporeal:
            logger.info(
                f"[Continuity] {len(noncorporeal)} non-corporeal entity(ies) "
                f"exempt from physical conditions: {', '.join(noncorporeal)}"
            )
        if naturally_unclothed:
            logger.info(
                f"[Continuity] {len(naturally_unclothed)} naturally-unclothed "
                f"entity(ies) — clothing lock uses natural-covering framing: "
                f"{', '.join(naturally_unclothed)}"
            )
        # Pull each character's tangible identity signature from the EI graph
        # node so it seeds a permanent marking (locked from panel one).
        signature_markings: Dict[str, str] = {}
        if graph is not None and getattr(graph, 'nodes', None):
            for nm in names:
                node = None
                key = nm.strip().lower()
                for gk, gn in graph.nodes.items():
                    if key == gk.lower() or key in gk.lower() or gk.lower() in key:
                        node = gn
                        break
                if node is not None and hasattr(node, 'visual_signature_lock'):
                    sig = node.visual_signature_lock(permanent_only=True)
                    if sig:
                        signature_markings[nm] = sig
        if signature_markings:
            logger.info(
                f"[Continuity] Seeded {len(signature_markings)} identity "
                f"signature(s) as permanent markings from panel one."
            )
        tracker = AppearanceContinuityTracker(
            registry=registry, visual_bible=visual_bible, character_names=names,
            noncorporeal_names=noncorporeal,
            naturally_unclothed_names=naturally_unclothed,
            signature_markings=signature_markings,
        )
        return tracker.process_script(script)
    except Exception as e:
        logger.warning(f"[Continuity] Tracking failed ({e}); panels left un-annotated.")
        return script


def appearance_continuity_clause(panel_script: Dict, characters_in_frame: List[str]) -> str:
    """Build the per-panel appearance lock from the tracked evolved state.

    This REPLACES the static costume lock for panels the tracker has annotated.
    It locks the CURRENT (post-change) value of every persistent facet —
    clothing, hair, conditions, held items, worn emotion, applied markings — for
    each visible character, and, when a facet is changing in this panel,
    instructs the model to depict the transition action instead of locking it.

    Returns "" when the panel has no tracked state (falls back to the scene
    anchor's static costume lock in that case).
    """
    state = panel_script.get('_appearance_state') or {}
    changes = panel_script.get('_appearance_changes') or []
    if not state:
        return ''

    # Index this-panel transitions by character for quick lookup.
    transitions_by_char: Dict[str, List[Dict[str, str]]] = {}
    for ch in changes:
        cname = ch.get('character', '')
        if cname:
            transitions_by_char.setdefault(cname, []).append(ch)

    # Build per-character locks, binding to gender subject phrases (not names,
    # which risk being rendered as visible text) with ordinal disambiguation.
    visible = [c for c in characters_in_frame if c in state]
    if not visible:
        # Fall back to whatever the tracker knows about, if names didn't line up
        visible = list(state.keys())
    if not visible:
        return ''

    gender_counts: Dict[str, int] = {}
    for c in visible:
        g = state[c].get('gender', 'figure') or 'figure'
        gender_counts[g] = gender_counts.get(g, 0) + 1
    _ordinals = ['first', 'second', 'third', 'fourth']
    gender_seen: Dict[str, int] = {}

    lock_parts: List[str] = []
    transition_parts: List[str] = []
    # Subjects -> authoritative "must still …" fragments. EVERY persistent facet
    # (a condition like mud/blood, a held prop, a worn emotion, a painted
    # marking) contributes a fragment here so the few-step diffusion model is
    # told, in one strong instruction, exactly what to carry over from the
    # preceding panels. A facet is excluded only when it is the very thing
    # changing in this panel — then the transition wording owns that beat.
    must_still_by_subject: List[Tuple[str, List[str]]] = []
    permanent_parts: List[str] = []  # scar / tattoo / prosthetic identity locks

    for c in visible:
        st = state[c]
        g = st.get('gender', 'figure') or 'figure'
        if gender_counts.get(g, 0) > 1:
            idx = gender_seen.get(g, 0)
            gender_seen[g] = idx + 1
            ordw = _ordinals[idx] if idx < len(_ordinals) else f"{idx+1}th"
            subject = f"the {ordw} {g}"
        else:
            subject = f"the {g}"

        # Resolve the active persistent conditions for this character. Prefer the
        # structured `conditions` list (new), falling back to legacy fields.
        active = set(st.get('conditions') or [])
        is_unclothed = bool(st.get('naturally_unclothed'))
        # Suppress nudity condition for naturally-unclothed entities — a dog or
        # angel without garments is simply their natural state, not "nude".
        if is_unclothed:
            active.discard('nude')
        elif st.get('undressed') or _CLOTHING_IS_NUDE(st.get('clothing', '')):
            active.add('nude')

        # Persistent-condition descriptors (authoritative, must-not-drop).
        cond_descriptors: List[str] = []
        # Stable, readable order: dress first, then injuries, then substances/etc.
        _order = {'dress': 0, 'injury': 1, 'wetness': 2, 'substance': 3, 'condition': 4}
        for cond_name in sorted(active, key=lambda n: _order.get(
                _CONDITIONS_BY_NAME.get(n).category if n in _CONDITIONS_BY_NAME else 'condition', 9)):
            cond = _CONDITIONS_BY_NAME.get(cond_name)
            if cond:
                cond_descriptors.append(cond.descriptor)

        # Compose the soft current-state descriptor AND collect authoritative
        # "must still …" fragments, both driven by the facet table so every
        # persistent dynamic is handled by the same code path.
        is_nude = 'nude' in active
        must_still: List[str] = []
        # Conditions (mud, blood, wet, nude…) are authoritative as before.
        if cond_descriptors:
            must_still.append("show " + ", ".join(cond_descriptors))

        clothing_desc: Optional[str] = None
        other_descs: List[str] = []
        for facet in _FACETS:
            val = (st.get(facet.name) or '').strip()
            if not val or facet.is_empty(val):
                continue
            # Clothing is suppressed while nude — the nude condition descriptor
            # already asserts no clothing and would otherwise contradict it.
            # Exception: naturally-unclothed entities have no nudity condition,
            # so their natural covering always shows.
            if facet.name == 'clothing' and is_nude and not is_unclothed:
                continue
            d = facet.descriptor(val)
            if facet.name == 'clothing':
                clothing_desc = d
            else:
                # Compress the soft descriptor for non-clothing facets too, but
                # keep the template prefix ("hair …", "holding …") intact by
                # compressing only the value and re-rendering. Articles are kept
                # here (they read naturally after a prefix like "holding the …")
                # to avoid odd phrasing; only filler/tails/connectors are cut.
                d = facet.descriptor(_compress_descriptor(val, drop_articles=False))
                other_descs.append(d)
            # Hard-lock a facet only when it has a lock template AND it is not
            # the attribute changing in this very panel.
            facet_changing = any(
                ch.get('attribute', '') == facet.name
                for ch in transitions_by_char.get(c, [])
            )
            # Closure is a sub-property of the garment: if the CLOTHING itself is
            # changing this panel (a coat removed, an outfit swapped) or the
            # character is going nude, the old closure lock ("keep the garment
            # buttoned") must yield too — otherwise it would fight the wardrobe
            # change. Treat closure as changing whenever clothing changes.
            if facet.name == 'closure':
                clothing_changing = any(
                    ch.get('attribute', '') == 'clothing'
                    for ch in transitions_by_char.get(c, [])
                )
                if clothing_changing or is_nude:
                    facet_changing = True
            # For naturally-unclothed entities, override the clothing lock
            # template to use "have" / natural-covering framing instead of
            # "be wearing", which implies a removable garment.
            if facet.name == 'clothing' and is_unclothed and val:
                frag = f"have {val}" if not facet_changing else ''
            else:
                frag = facet.lock_fragment(val)
            if frag and not facet_changing:
                must_still.append(frag)

        # Soft state line, in reading order: clothing, conditions, then the rest.
        # This line is what the DIFFUSION MODEL reads, so we compress it into
        # concise comma-separated phrasing (dropping "is wearing", redundant
        # possessive tails, articles) to maximise visual detail per token. The
        # hard-lock "must still …" predicates above are left in natural grammar
        # untouched — they are read by the LLM refinement pass, not the image
        # model, and rely on that grammar to be understood as constraints.
        descriptors: List[str] = []
        if clothing_desc:
            descriptors.append(_compress_descriptor(clothing_desc))
        descriptors.extend(cond_descriptors)
        # Free-text condition nuance only if not already covered structurally.
        ft = (st.get('condition') or '').strip()
        if ft and not any(ft.lower() in d.lower() or d.lower() in ft.lower()
                          for d in cond_descriptors):
            descriptors.append(ft)
        descriptors.extend(other_descs)

        if descriptors:
            lock_parts.append(f"{subject}: {', '.join(descriptors)}")
        if must_still:
            must_still_by_subject.append((subject, must_still))

        # Permanent markings (scars, tattoos, prosthetics) are injected as a
        # separate hard lock — they survive scene changes and are never cleared
        # by the regression guard. They go into their own sentence so the image
        # model treats them as identity-level constraints, not transient state.
        pm = (st.get('permanent_markings') or '').strip()
        if pm:
            permanent_parts.append(f"{subject}: {pm}")

        # If a change is happening in this panel, describe the transition action.
        for ch in transitions_by_char.get(c, []):
            tr = ch.get('transition', '')
            if tr:
                transition_parts.append(f"{subject} is {tr}")

    if not lock_parts and not transition_parts and not permanent_parts:
        return ''

    out: List[str] = []
    if lock_parts:
        out.append("; ".join(lock_parts).capitalize() + ".")
    if permanent_parts:
        # Permanent marks are identity — they pre-date this scene and survive
        # every scene change. Phrased as identity fact, not continuity nudge.
        out.append(
            "Permanent identity marks (always visible, never absent): "
            + "; ".join(permanent_parts) + "."
        )
    if must_still_by_subject:
        # One strong, unambiguous instruction that wins over any contradicting
        # wording in the panel description — this is what stops the flicker of
        # ANY persistent facet (a condition, a held prop, a worn emotion, a
        # painted marking) between consecutive panels of a scene.
        phrases = [f"{subject} must still {', '.join(frags)}"
                   for subject, frags in must_still_by_subject]
        out.append(
            "Continuity is critical: "
            + "; ".join(phrases)
            + " — exactly as established in the preceding panels of this scene; "
            "do not omit, drop, swap, clean up, heal, or otherwise revert these "
            "unless this very panel shows the change."
        )
    if transition_parts:
        out.append("; ".join(transition_parts).capitalize() + ".")
    return " ".join(out)


# ---------------------------------------------------------------------------
# compose_panel_prompt patch helper
# ---------------------------------------------------------------------------

def visual_bible_clause(visual_bible: Optional[SeriesVisualBible], description: str) -> str:
    """
    Drop-in helper for compose_panel_prompt().

    Returns a string to append to the prompt parts list (or "" if nothing matches).
    Typically called like:

        vb_clause = visual_bible_clause(visual_bible, description)
        if vb_clause:
            parts.append(vb_clause)
    """
    if not visual_bible:
        return ""
    return visual_bible.prompt_clause_for(description)


# =============================================================================
# AMBIGUOUS NAME RESOLUTION — closing the loop on [NameResolve] warnings
# =============================================================================
# CharacterAppearanceRegistry._resolve_name() correctly REFUSES to guess when a
# script name ambiguously matches several cast members (e.g. "Elena Morales"
# sharing a first name with both registered "Elena Petrova" and "Elena
# Vasquez") or matches none of them outright. That refusal is the right default
# in the hot path — silently picking the wrong character is worse than losing
# one panel's appearance lock — but it shouldn't be the END of the story for a
# whole production run: an unresolved name then drops that character's lock on
# EVERY panel it appears in, for the entire book.
#
# This pass runs ONCE, after the full script exists (so every occurrence's
# surrounding panel context is available) and BEFORE image generation, and
# closes the loop the safe way: for each name the registry has already flagged
# as unresolved/ambiguous, it gathers the panel(s) where that name appears,
# shows the LLM the candidate characters' own descriptions, and asks for an
# explicit, auditable decision — "this is candidate X" or "this is genuinely a
# new/different character, not in the cast." Only on a CONFIDENT decision does
# it call registry.register_alias(...), permanently and correctly resolving
# every future lookup of that name. Anything the LLM itself is unsure about is
# left exactly as unresolved as before — this pass only ever ADDS resolutions,
# it never relaxes the registry's refuse-to-guess safety net.
# =============================================================================

def _gather_name_occurrence_context(script: List[Dict], name: str,
                                    max_examples: int = 3) -> List[str]:
    """Short context snippets (setting + description) for panels where ``name``
    appears in characters_in_frame or as a dialogue speaker — what the LLM
    disambiguator reads to judge which real character is meant."""
    snippets: List[str] = []
    name_low = name.lower()
    for page in script:
        if not isinstance(page, dict) or page.get('_act_break'):
            continue
        for panel in (page.get('panels') or []):
            present = any(
                str(c).strip().lower() == name_low
                for c in (panel.get('characters_in_frame') or [])
            )
            spoken = any(
                str(d.get('speaker', '')).strip().lower() == name_low
                for d in (panel.get('dialogue') or [])
            )
            if not (present or spoken):
                continue
            setting = (panel.get('setting') or '').strip()
            desc = (panel.get('description') or '').strip()
            snippets.append(f"[{setting}] {desc}"[:220])
            if len(snippets) >= max_examples:
                return snippets
    return snippets


def resolve_ambiguous_character_names(
    script: List[Dict],
    registry: 'CharacterAppearanceRegistry',
    characters: Optional[list] = None,
) -> Dict[str, str]:
    """One-time, context-informed pass that resolves names the registry has
    flagged as ambiguous or unmatched, via registry.register_alias().

    Call this AFTER the full script is generated and BEFORE image generation —
    it needs ``registry.get_unresolved_names()`` to be populated, which only
    happens once something has actually tried to resolve those names. If the
    registry hasn't seen any lookups yet, this proactively probes every name
    that appears in the script's characters_in_frame / dialogue speakers first,
    so ambiguities are caught and fixed BEFORE the first panel renders rather
    than discovered panel-by-panel during generation.

    Returns {original_name: resolved_registry_key} for every alias registered.
    Never raises — any failure leaves the registry's existing safe behaviour
    (refuse to guess) untouched.
    """
    try:
        # Proactively probe every distinct name in the script so ambiguities
        # surface now, in one batch, rather than trickling out during image
        # generation one warning at a time.
        all_names: set = set()
        for page in script:
            if not isinstance(page, dict) or page.get('_act_break'):
                continue
            for panel in (page.get('panels') or []):
                for c in (panel.get('characters_in_frame') or []):
                    nm = str(c).strip()
                    if nm:
                        all_names.add(nm)
                for d in (panel.get('dialogue') or []):
                    nm = str(d.get('speaker', '')).strip()
                    if nm and nm.upper() != 'NARRATOR':
                        all_names.add(nm)
        for nm in all_names:
            registry._resolve_name(nm)   # populates get_unresolved_names() via cache

        unresolved = registry.get_unresolved_names()
        if not unresolved:
            return {}

        char_by_name = {}
        for c in (characters or getattr(registry, '_characters', None) or []):
            cname = getattr(c, 'name', '') or (c.get('name') if isinstance(c, dict) else '')
            if cname:
                char_by_name[cname] = c

        resolved: Dict[str, str] = {}
        items = list(unresolved.items())
        batch_size = 6
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            resolved.update(_disambiguate_name_batch(batch, script, registry, char_by_name))

        if resolved:
            logger.info(
                f"[NameResolve] Disambiguation pass resolved {len(resolved)} "
                f"ambiguous name(s): {resolved}"
            )
        return resolved
    except Exception as e:
        logger.warning(f"[NameResolve] Disambiguation pass skipped ({e}).")
        return {}


def _disambiguate_name_batch(batch, script, registry, char_by_name) -> Dict[str, str]:
    """Ask the LLM to judge one batch of ambiguous names against their
    candidates, using each occurrence's panel context. Applies register_alias
    for confident decisions only."""
    entries = []
    for name, info in batch:
        candidates = info.get('candidates', [])
        cand_descs = []
        for cand in candidates:
            ch = char_by_name.get(cand)
            role = getattr(ch, 'role', '') if ch is not None else ''
            appearance = registry.registry.get(cand, '')
            cand_descs.append(f"    - {cand} (role: {role}): {appearance[:100]}")
        context = _gather_name_occurrence_context(script, name)
        ctx_text = "\n".join(f"    \"{c}\"" for c in context) or "    (no panel context found)"
        entries.append(
            f"- name as written: \"{name}\"\n"
            f"  candidates it might refer to:\n" + "\n".join(cand_descs) + "\n"
            f"  panels where it appears:\n{ctx_text}"
        )
    entries_text = "\n".join(entries)

    prompt = f"""A graphic-novel script uses some character names that don't exactly
match the registered cast — likely a different surname, a nickname, or a
typo from an earlier writing pass. For each one, decide which registered
candidate it actually refers to, using the panel context as evidence.

{entries_text}

For EACH name, decide:
  - "match": the registry key of the candidate it refers to, IF you are
    CONFIDENT (the context clearly points to one candidate — e.g. matching
    role, scene partner, or established relationship). Use the EXACT candidate
    string given above.
  - OR "new_character": true, if the context suggests this is actually a
    genuinely different person not in the candidate list (e.g. a new minor
    character who happens to share a first name) — in this case do NOT force a
    match.
  - OR leave both empty if you are not confident either way.

Be conservative: only return a "match" when the evidence clearly supports it.
A wrong guess is worse than no guess.

Return ONLY a JSON array:
[{{"name": "as written", "match": "exact candidate string or empty",
   "new_character": false}}]
"""
    try:
        raw = _llm(prompt, temperature=0.1)
        parsed = _parse(raw)
        if not isinstance(parsed, list):
            return {}
    except Exception as e:
        logger.warning(f"[NameResolve] Disambiguation batch LLM call failed: {e}")
        return {}

    resolved: Dict[str, str] = {}
    valid_names = {n for n, _ in batch}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name', '')).strip()
        if name not in valid_names:
            continue
        if item.get('new_character'):
            logger.info(
                f"[NameResolve] '{name}' judged a genuinely new/different "
                f"character — left unresolved (no alias registered)."
            )
            continue
        match = str(item.get('match', '')).strip()
        if not match:
            continue
        if registry.register_alias(name, match):
            resolved[name] = match
    return resolved

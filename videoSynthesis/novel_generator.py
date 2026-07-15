"""
=============================================================================
NOVEL GENERATOR — "Bestseller Edition" (Importable Module Version)
=============================================================================
This is the novel-generation pipeline, packaged as an importable module so
the comic generator can reuse its building blocks (StoryIdea, Character,
CharacterGraph, image generation, LLM helpers, etc.).

The original code was a single executable script. Three changes were made
to make it import-safe:

  1. `from __future__ import annotations` moved to line 1 (was illegal mid-file)
  2. The entire main pipeline (story synthesis + image gen + docx assembly)
     is now under `if __name__ == "__main__":` so importing this module
     does NOT trigger a 30-minute book generation.
  3. Optional heavy dependencies (aura_sr, diffusers, openai client) are
     imported defensively — the comic generator only needs a subset.

EVERYTHING ELSE IS IDENTICAL TO THE ORIGINAL CODE.
=============================================================================
"""
from __future__ import annotations

import os
os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"

import gc
import logging
import pickle
import json
import random
import re
import sys
import time
import textwrap

# Standard library
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Union, Any, Set

# Token-budget guard. Self-contained, stdlib-only; imported defensively so a
# missing file degrades to a no-op instead of breaking the whole pipeline.
try:
    import comic_book_token_budget as _tb
except Exception:  # pragma: no cover
    _tb = None

# Third party — these are required
import numpy as np
import json_repair
import requests

# Image processing
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

# Document creation
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, Mm, RGBColor, Emu
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH

# tqdm
from tqdm import tqdm

# LLM client
from openai import OpenAI

# Heavy ML libs — imported defensively. The comic generator only needs the
# image pipeline functions; if torch/diffusers/aura_sr aren't installed,
# we can still use the LLM-based functions (story/character generation).
try:
    import torch
    import torch.multiprocessing as mp
    _HAS_TORCH = True
except ImportError:
    torch = None
    mp = None
    _HAS_TORCH = False

try:
    from diffusers import ZImagePipeline
    from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel
    try:
        from diffusers import ChromaPipeline
    except ImportError:
        ChromaPipeline = None
    _HAS_DIFFUSERS = True
except ImportError:
    ZImagePipeline = None
    ErnieImagePipeline = None
    ErnieImageTransformer2DModel = None
    ChromaPipeline = None
    _HAS_DIFFUSERS = False

# LensPipeline ships in Microsoft's `lens` package (github.com/microsoft/Lens),
# NOT in diffusers. Importing `lens` is what registers its custom
# LensGptOssEncoder / LensTransformer2DModel components with the transformers
# and diffusers namespaces that the model's model_index.json references, so the
# import must succeed before from_pretrained("microsoft/Lens") will work.
# Independent of _HAS_DIFFUSERS so a missing `lens` package doesn't disable the
# other models (and vice versa).
try:
    from lens import LensPipeline
    _HAS_LENS = True
except ImportError:
    LensPipeline = None
    _HAS_LENS = False

# KLEIN2 = FLUX.2 Klein 9B (black-forest-labs/FLUX.2-klein-9B) loaded via the
# generic diffusers DiffusionPipeline, with an uncensored drop-in text encoder
# (ponpoke/flux2-klein-9b-uncensored-text-encoder) swapped in at load time.
# Both come through diffusers/transformers with trust_remote_code, so we only
# need DiffusionPipeline + AutoModel/AutoTokenizer (imported below). The flag is
# independent so a diffusers build without FLUX.2 support doesn't disable the
# other models.
try:
    from diffusers import DiffusionPipeline as _Flux2DiffusionPipeline
    _HAS_KLEIN2 = True
except ImportError:
    _Flux2DiffusionPipeline = None
    _HAS_KLEIN2 = False

try:
    from aura_sr import AuraSR
    _HAS_AURA = True
except ImportError:
    AuraSR = None
    _HAS_AURA = False

# transformers — used by some image flows
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoModel = None
    _HAS_TRANSFORMERS = False


logger = logging.getLogger(__name__)


def _host(name, default=None):
    """Read a global by name with fallback default."""
    return globals().get(name, default)


# =============================================================================
# Descriptor compression — concise, image-model-friendly appearance text.
# =============================================================================
# The canonical model-sheet clause ("<portrait>. Wearing a short skirt and denim
# blue jeans with black leather gloves over her hands.") is injected into the
# image prompt for EVERY panel a character appears in, so its length is paid
# repeatedly. Diffusion models parse concise comma-separated noun phrases
# ("short skirt, blue jeans, black leather gloves") more reliably than full
# prose, so compressing the CLOTHING portion is a double win: fewer tokens and
# often better adherence.
#
# This mirrors the conservative compressor in comic_book_art_director, kept
# self-contained here so novel_generator has no cross-module dependency (the
# art director already imports novel_generator, so importing back would risk a
# cycle). It only strips a fixed set of known-redundant lead-ins, possessive
# tails, connectors, and leading articles — anything it does not recognise is
# passed through unchanged, so material/colour/count/placement detail always
# survives. Idempotent. Applied ONLY to the clothing text, never the portrait
# (identity description) and never any hard constraint text.

_ND_LEAD_FILLER_RE = re.compile(
    r'^(?:she|he|they|it)?\s*(?:is|are|was|were)?\s*'
    r'(?:currently\s+)?(?:wearing|dressed in|clad in|adorned (?:with|in)|'
    r'attired in|garbed in|outfitted in|sporting|donning)\s+',
    re.IGNORECASE)
_ND_REDUNDANT_TAIL_RE = re.compile(
    r'\s+(?:over|on|upon)\s+(?:her|his|their|its)\s+'
    r'(?:hands?|feet|head|face|body|frame|form|shoulders?|legs?|arms?|torso)\b'
    r'(?=\s*(?:,|;|and\b|$))',
    re.IGNORECASE)
_ND_AND_CONNECTOR_RE = re.compile(r'\s+(?:along )?with\s+|\s+and\s+', re.IGNORECASE)
_ND_LEADING_ARTICLE_RE = re.compile(r'(^|,\s*|;\s*)(?:a|an|the)\s+', re.IGNORECASE)
_ND_MULTISPACE_RE = re.compile(r'\s{2,}')
_ND_MULTICOMMA_RE = re.compile(r'\s*,\s*(?:,\s*)+')


def _compress_clothing_text(text: str, drop_articles: bool = True) -> str:
    """Compress a clothing description into concise, comma-separated,
    image-model-friendly phrasing WITHOUT losing visual detail.

    "she is wearing a short skirt and denim blue jeans with black leather gloves
    over her hands" -> "short skirt, denim blue jeans, black leather gloves".

    Conservative and idempotent: only strips known filler; preserves meaningful
    placement ("gloves tucked into her belt", "scarf around the neck").
    """
    if not text:
        return text
    t = text.strip()
    t = _ND_LEAD_FILLER_RE.sub('', t)
    t = ' ' + t + ' '
    prev = None
    while prev != t:
        prev = t
        t = _ND_REDUNDANT_TAIL_RE.sub('', t)
    t = _ND_AND_CONNECTOR_RE.sub(', ', t)
    t = _ND_MULTICOMMA_RE.sub(', ', t)
    t = _ND_MULTISPACE_RE.sub(' ', t)
    t = t.strip(' ,;')
    if drop_articles:
        prev = None
        while prev != t:
            prev = t
            t = _ND_LEADING_ARTICLE_RE.sub(r'\1', t)
        t = t.strip(' ,;')
    return t


# =============================================================================
# CONFIGURATION
# =============================================================================
# IMPORTANT: Set these to your real keys via environment variables in production.
# The defaults are placeholders matching the original code structure.
openai_api_key = ''
hg_token = ''
grok_api_key = ''

openai_model = 'gpt-5.4-nano'
openai_model_large = 'gpt-5.4-mini'
openai_model_small_reasoning = 'gpt-5.4-nano'
grok_fast_reasoning_model = 'grok-4.3'
grok_fast_nonreasoning_model = 'grok-4.3'
# Vision-capable model for image-in, text-out calls (panel consistency checks,
# etc.). Both providers' flagship chat models here are multimodal, so this
# simply reuses the "large" tier rather than naming a separate model.
openai_vision_model = openai_model_large
grok_vision_model = grok_fast_nonreasoning_model
retry_limit = 5

# --- Prompt-size ceiling ---------------------------------------------------
# Hard upper bound (in tokens) for ANY single LLM call, enforced inside
# get_openai_prompt_response(). Sits far below the 1M model limit so that even
# a prompt-builder that fails to bound its own context can never trigger the
# 400 "maximum prompt length" / 413 "Payload Too Large" errors again — the
# guard trims the prompt and logs loudly instead. Tune via the
# LLM_GLOBAL_MAX_PROMPT_TOKENS env var, or lower it here to push closer to the
# "minimal context per subtask" ideal.
MAX_PROMPT_TOKENS = int(os.getenv("LLM_GLOBAL_MAX_PROMPT_TOKENS", "200000"))

# Output / runtime
docSaveDir = "./SynthesizedBooks"
LOAD_PREVIOUS_BOOK = False
USE_GROK = True
IMAGE_MODEL = 'ZIMAGE'  # 'ZIMAGE' | 'ERNIE' | 'CHROMA' | 'LENS' | 'KLEIN2'

# Negative-prompt guidance scale.
# The default few-step distilled models (ZImage @ CFG 0, ERNIE @ CFG 1) do NOT
# use classifier-free guidance, so negative prompts have no effect and are not
# passed to the pipeline. If you switch to a CFG-capable model, set this to a
# real scale (e.g. 4.0) and gen_ImageZ_image will pass the negative prompt and
# use this guidance scale automatically. Leave at 0.0 for the distilled models.
#
# NOTE: CHROMA (lodestones/Chroma1-HD) is a full CFG model — it ALWAYS uses
# its negative prompt and its own guidance scale (CHROMA_GUIDANCE below),
# independent of NEGATIVE_PROMPT_GUIDANCE. The negative prompt genuinely
# improves CHROMA output, so it is always passed for that model.
NEGATIVE_PROMPT_GUIDANCE = 0.0

# CHROMA (lodestones/Chroma1-HD) parameters — a full classifier-free-guidance
# model (NOT distilled), so it needs many more steps and a real guidance scale,
# and it benefits from negative prompts. These follow the model's reference usage.
CHROMA_MODEL_ID = "lodestones/Chroma1-HD"
CHROMA_N_STEPS = 40       # reference uses 40; quality model, not a few-step one
CHROMA_GUIDANCE = 3.0     # reference guidance_scale
CHROMA_COVER_N_STEPS = 40 # covers use the same quality settings

# LENS (microsoft/Lens) parameters — a 3.8B full classifier-free-guidance
# text-to-image model (github.com/microsoft/Lens). Like CHROMA it uses a real
# guidance scale and benefits from a negative prompt. It natively supports
# arbitrary aspect ratios up to 1440x1440; we drive it with explicit
# height/width (rounded to its VAE factor of 16) for pixel-precise panel fit.
# Use "microsoft/Lens-Turbo" with LENS_N_STEPS=4 for the fast distilled variant.
LENS_MODEL_ID = "microsoft/Lens-Turbo"
LENS_N_STEPS = 4         # reference uses 20 (Lens-Turbo: 4; Lens-Base: 50)
LENS_GUIDANCE = 1.0       # reference guidance_scale (HF card default is 5.0)
LENS_COVER_N_STEPS = 20
LENS_VAE_FACTOR = 16      # height/width must be divisible by this
LENS_MAX_DIM = 1440       # native max resolution per side
LENS_CPU_OFFLOAD = True   # True = enable_model_cpu_offload(); False = .to("cuda")

# KLEIN2 (FLUX.2 Klein 9B) parameters — a 9B FLUX.2 model loaded via the generic
# diffusers DiffusionPipeline, with an uncensored drop-in text encoder swapped
# in at load time. The Flux2KleinPipeline does NOT accept a negative_prompt
# argument, so generation is driven by the positive prompt + guidance scale
# only. Native resolution 1536x1536; we drive it with explicit height/width
# (rounded to its VAE factor of 16) for pixel-precise panel fit.
KLEIN2_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
KLEIN2_ENCODER_ID = "ponpoke/flux2-klein-9b-uncensored-text-encoder"
KLEIN2_N_STEPS = 4        # reference uses 4
KLEIN2_GUIDANCE = 1.0     # reference guidance_scale
KLEIN2_COVER_N_STEPS = 4
KLEIN2_VAE_FACTOR = 16     # height/width must be divisible by this
KLEIN2_MAX_DIM = 1536      # native resolution per side
KLEIN2_CPU_OFFLOAD = True  # reference uses enable_model_cpu_offload()

# ---------------------------------------------------------------------------
# Per-model native generation geometry.
# ---------------------------------------------------------------------------
# Each text-to-image model has a native training resolution and a dimension
# constraint (sizes must be divisible by the model's VAE/patch factor). To get
# the best quality we anchor the LONGER side of a panel at the model's native
# base resolution, then scale the SHORTER side down to match the panel's aspect
# ratio — staying as close to native as the aspect ratio allows, rounded to the
# model's factor and clamped to its max side.
#
#   base   : native square training resolution (the anchor / "full" side length)
#   factor : required pixel multiple for H and W
#   max_dim: hard cap per side
#   min_dim: smallest sensible side (avoids degenerate slivers)
#
# LENS is 1440-native; KLEIN2 is 1536-native; ZIMAGE / ERNIE / CHROMA are 1024-native.
# NOTE: all current diffusion backends (Z-Image, ERNIE/Flux, Chroma/Flux,
# Lens, FLUX.2 Klein) require height & width divisible by 16. An earlier ÷8
# setting for ZIMAGE/ERNIE produced sizes like 888 (=8x111) that the Z-Image
# pipeline rejects ("Height must be divisible by 16"). Keep every factor at 16.
_MODEL_GEN_GEOMETRY = {
    'ZIMAGE': {'base': 1024, 'factor': 16, 'max_dim': 1024, 'min_dim': 384},
    'ERNIE':  {'base': 1024, 'factor': 16, 'max_dim': 1024, 'min_dim': 384},
    'CHROMA': {'base': 1024, 'factor': 16, 'max_dim': 1024, 'min_dim': 384},
    'LENS':   {'base': 1440, 'factor': LENS_VAE_FACTOR, 'max_dim': LENS_MAX_DIM, 'min_dim': 512},
    'KLEIN2': {'base': 1536, 'factor': KLEIN2_VAE_FACTOR, 'max_dim': KLEIN2_MAX_DIM, 'min_dim': 512},
}
_DEFAULT_GEN_GEOMETRY = {'base': 1024, 'factor': 16, 'max_dim': 1024, 'min_dim': 384}


def get_model_gen_geometry(image_model: str = None) -> dict:
    """Return the native generation geometry dict for a model name.

    Defaults to the module-level IMAGE_MODEL when none is given, and to a safe
    1024/×8 profile for any unrecognised model.
    """
    name = (image_model or globals().get('IMAGE_MODEL', 'ZIMAGE') or 'ZIMAGE').upper()
    return dict(_MODEL_GEN_GEOMETRY.get(name, _DEFAULT_GEN_GEOMETRY))

# Image generation params
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
dtype = torch.bfloat16 if _HAS_TORCH else None

# Interior illustration parameters
width = 768
height = 1024
guidance_scale = 4
n_steps = 8

# Cover art parameters
COVER_WIDTH = 640
COVER_HEIGHT = 1024
COVER_N_STEPS = 8
COVER_GUIDANCE = 4

lora_triggers = ''
cover_images_to_generate = 16
images_to_generate = 4
TRANSPARENT_BACKGROUND = False
IMAGE_JPEG_QUALITY = 90
IMAGE_MAX_DIMENSION = 3200

# Story parameters
number_of_plot_points = 24
number_of_chapters = 36

# Quality gates
MIN_CHAPTER_WORDS = 600
TARGET_CHAPTER_WORDS = 2500
MAX_DIALOGUE_RATIO = 0.65
MIN_DIALOGUE_RATIO = 0.10
REPETITION_THRESHOLD = 3

# Bestseller-quality gates
REQUIRED_SHOWSTOPPER_SCENES = 3
REQUIRED_QUOTABLE_LINES = 8
MIN_HOOK_STRENGTH_SCORE = 7

# Interludes
ENABLE_INTERLUDES = True
INTERLUDE_FREQUENCY = 6

# Audience override
MANUAL_AUDIENCE_OVERRIDE = None

# Ornament characters
ORNAMENT_CHARS = {'\u2766', '\u2042', '\u2767', '\u2756'}
ORNAMENT_PATTERN = re.compile('[' + ''.join(re.escape(c) for c in ORNAMENT_CHARS) + ']')

# Style baseline
IMAGE_STYLE = (
    '3D Animation, CGI, special effects, cinematic lighting, '
    'intimate and emotional closeup scene'
)


# =============================================================================
# DATA CLASSES
# =============================================================================

class Character:
    """Rich character profile used throughout the pipeline.
    
    Hardened against malformed LLM output: list-typed fields (traits,
    knowledge_domains, signature_habits, intelligence_markers) are coerced
    to clean lists of strings at construction time. This means downstream
    code can safely do `', '.join(char.traits)` without TypeError.
    """
    
    def __init__(self, name, age, role, traits, backstory, appearance="",
                 speech_pattern="", arc="", personality_type="",
                 cognitive_style="", humor_style="", knowledge_domains=None,
                 coping_mechanism="", social_energy="", inner_world="",
                 signature_habits=None, relationship_style="",
                 voice_guide="", romantic_energy="", creative_expression="",
                 sensory_orientation="",
                 dialect="", mythic_archetype="", wit_level="",
                 creative_medium="", intelligence_markers=None,
                 gender="", physical_build="",
                 cadence="", vocabulary_level="", verbal_tics=None,
                 catchphrases=None, voice_profile="", origin=""):
        # Scalar string fields — coerce None to "" and dicts/lists to str
        self.name = self._coerce_to_str(name, default="Unknown")
        self.age = age  # leave as-is; could be int or "Unknown"
        self.role = self._coerce_to_str(role)
        self.backstory = self._coerce_to_str(backstory)
        self.appearance = self._coerce_to_str(appearance)
        # Visual-consistency fields. gender is an explicit token used as the
        # FIRST word in every image prompt for this character so diffusion
        # models cannot drift to the opposite sex between panels.
        # physical_build holds the canonical body description (height, frame,
        # skin tone) — captured at creation so it never has to be re-invented.
        self.gender = self._coerce_to_str(gender)
        self.physical_build = self._coerce_to_str(physical_build)
        self.speech_pattern = self._coerce_to_str(speech_pattern)
        self.arc = self._coerce_to_str(arc)
        self.personality_type = self._coerce_to_str(personality_type)
        self.cognitive_style = self._coerce_to_str(cognitive_style)
        self.humor_style = self._coerce_to_str(humor_style)
        self.coping_mechanism = self._coerce_to_str(coping_mechanism)
        self.social_energy = self._coerce_to_str(social_energy)
        self.inner_world = self._coerce_to_str(inner_world)
        self.relationship_style = self._coerce_to_str(relationship_style)
        self.romantic_energy = self._coerce_to_str(romantic_energy)
        self.creative_expression = self._coerce_to_str(creative_expression)
        self.sensory_orientation = self._coerce_to_str(sensory_orientation)
        self.voice_guide = self._coerce_to_str(voice_guide)
        self.dialect = self._coerce_to_str(dialect)
        # origin — place/culture this character is from (e.g. "Guadalajara,
        # Mexico"; "rural Appalachia, USA"; "Mumbai, India"). This is the seed
        # fact that everything culturally-flavoured about the voice hangs off:
        # dialect, code-switching, cultural/religious references. Captured
        # here (rather than inferred later from a maybe-silent backstory) so
        # the voice-profile enrichment pass in comic_book_dialogue.py has
        # real grounding instead of guessing.
        self.origin = self._coerce_to_str(origin)
        self.mythic_archetype = self._coerce_to_str(mythic_archetype)
        self.wit_level = self._coerce_to_str(wit_level)
        self.creative_medium = self._coerce_to_str(creative_medium)

        # --- Voice profile: how this character SOUNDS on the page. ---
        # These power the dialogue generation + review passes so each
        # character has a distinct rhythm, dialect, vocabulary register, and
        # set of verbal fingerprints. Captured at creation so the writer LLM
        # never has to re-invent a voice mid-story (the main cause of every
        # character sounding the same).
        #   cadence          — the music of their speech: do they alternate
        #                       short punchy lines with long winding clauses?
        #                       clipped? lyrical? halting? breathless?
        #   vocabulary_level  — register and word choice: gutter slang, plain
        #                       everyday, erudite/Dickinson-precise, jargon-heavy.
        #   verbal_tics       — involuntary speech fingerprints (filler words,
        #                       sentence-enders, grammatical quirks of dialect).
        #   catchphrases      — signature recurring phrases the reader learns.
        #   voice_profile     — a flowing paragraph synthesising all of the
        #                       above for prompt injection.
        self.cadence = self._coerce_to_str(cadence)
        self.vocabulary_level = self._coerce_to_str(vocabulary_level)
        self.voice_profile = self._coerce_to_str(voice_profile)
        
        # List fields — coerce to flat list of clean strings.
        # This is the fix for the TypeError. The LLM sometimes returns
        # traits as a mix of strings and dicts, or pure dicts, or strings
        # with commas — _coerce_to_string_list normalizes all of it.
        self.traits = _coerce_to_string_list(traits)
        self.knowledge_domains = _coerce_to_string_list(knowledge_domains)
        self.signature_habits = _coerce_to_string_list(signature_habits)
        self.intelligence_markers = _coerce_to_string_list(intelligence_markers)
        self.verbal_tics = _coerce_to_string_list(verbal_tics)
        self.catchphrases = _coerce_to_string_list(catchphrases)
    
    @staticmethod
    def _coerce_to_str(value, default: str = "") -> str:
        """Coerce a value to a string. Handles None, dicts, lists."""
        if value is None:
            return default
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            # Pull the most likely descriptive field
            extracted = (value.get('name') or value.get('text')
                         or value.get('description') or value.get('value'))
            if extracted:
                return str(extracted)
            # Otherwise stringify first scalar value
            for v in value.values():
                if v and isinstance(v, (str, int, float)):
                    return str(v)
            return default
        if isinstance(value, list):
            # Join non-empty string-coerced items
            parts = [Character._coerce_to_str(v) for v in value]
            parts = [p for p in parts if p]
            return ', '.join(parts) if parts else default
        return str(value)
    
    def __str__(self):
        traits_list = ', '.join(self.traits) if self.traits else ''
        return (
            f"Character:\n  Name: {self.name}\n  Age: {self.age}\n  Role: {self.role}\n"
            f"  Traits: {traits_list}\n  Backstory: {self.backstory}\n"
            f"  Appearance: {self.appearance}\n"
        )
    
    def build_voice_guide(self) -> str:
        """Build a compact voice guide string for prompt injection.

        Synthesises the full voice profile — personality, thinking, humour,
        dialect, cadence, vocabulary register, verbal tics, and catchphrases —
        into one block the dialogue writer and reviewer can lean on so every
        character sounds like a distinct, believable person on the page.
        """
        parts = [f"VOICE GUIDE FOR {self.name.upper()}:"]
        if self.personality_type:
            parts.append(f"Personality: {self.personality_type}.")
        if self.cognitive_style:
            parts.append(f"Thinking: {self.cognitive_style}.")
        if self.humor_style:
            parts.append(f"Humor: {self.humor_style}.")
        if self.origin:
            parts.append(f"From: {self.origin}.")
        if self.dialect:
            parts.append(f"Dialect/region: {self.dialect}.")
        if self.cadence:
            parts.append(f"Cadence: {self.cadence}.")
        if self.vocabulary_level:
            parts.append(f"Vocabulary register: {self.vocabulary_level}.")
        if self.speech_pattern:
            parts.append(f"Speech: {self.speech_pattern}.")
        if self.verbal_tics:
            parts.append(f"Verbal tics: {'; '.join(self.verbal_tics)}.")
        if self.catchphrases:
            parts.append(f"Catchphrases: {'; '.join(self.catchphrases)}.")
        if self.knowledge_domains:
            parts.append(f"Expert in: {', '.join(self.knowledge_domains)}.")
        if self.signature_habits:
            parts.append(f"Habits: {'; '.join(self.signature_habits)}.")
        # If the LLM authored a synthesised paragraph, lead with it — it's the
        # richest single source and reads naturally for the writer model.
        if self.voice_profile:
            parts.append(f"Voice in full: {self.voice_profile}")
        self.voice_guide = ' '.join(parts)
        return self.voice_guide


class StoryIdea:
    """Compact representation of a story concept."""
    
    def __init__(self, genre, themes, mood, premise):
        self.genre = genre
        self.themes = themes if isinstance(themes, list) else [themes] if themes else []
        self.mood = mood
        self.premise = premise
    
    def __str__(self):
        themes_list = ', '.join(self.themes)
        return (f"Story Idea:\n  Genre: {self.genre}\n  Themes: {themes_list}\n"
                f"  Mood: {self.mood}\n  Premise: {self.premise}\n")


class CharacterAppearanceRegistry:
    """Builds and queries a registry of character visual descriptions.

    Upgraded for character consistency across graphic novel panels.

    The central idea: diffusion models reproduce a character reliably only when
    the SAME descriptive tokens appear in the SAME ORDER in every prompt.  So
    this registry builds, for each character, a CANONICAL PORTRAIT — a strictly
    ordered descriptor (gender → age → skin → build → hair → eyes → face) that
    is injected byte-for-byte identically into every panel prompt the character
    appears in.

    Maps
    ----
      - ``gender_map``       : explicit gender token ("woman" / "man" / etc.),
                               the FIRST token in every prompt so the model
                               cannot drift to the opposite sex.
      - ``portrait_map``     : the canonical fixed-order physical descriptor
                               (immutable identity — never changes per scene).
      - ``clothing_map``     : the default outfit (can be overridden per scene
                               by the SceneConsistencyAnchor).
      - ``locked_appearance``: portrait + outfit combined — the string to inject
                               via ``get_locked_appearance()``.
    """

    # Valid gender tokens.  Order: binary terms first so substring checks are simple.
    _GENDER_TOKENS = {"woman", "man", "girl", "boy", "person", "nonbinary person"}

    def __init__(self, characters: List[Character]):
        self.registry: Dict[str, str] = {}           # name → plain visual desc (legacy)
        self.locked_appearance: Dict[str, str] = {}  # name → portrait + outfit clause
        self.portrait_map: Dict[str, str] = {}       # name → canonical fixed-order portrait
        self.color_palettes: Dict[str, str] = {}
        self.gaze_map: Dict[str, str] = {}          # name → eye/gaze "tell" (expressiveness)
        self.gender_map: Dict[str, str] = {}         # name → "woman" | "man" | …
        self.clothing_map: Dict[str, str] = {}       # name → base outfit description
        # Explicit, externally-registered name → registry-key resolutions. Checked
        # BEFORE any matching tier, so once an ambiguous name is resolved (by
        # register_alias, typically from an LLM disambiguation pass that read the
        # surrounding scene context) every future lookup is instant and correct —
        # no re-matching, no repeated log warnings.
        self._alias_map: Dict[str, str] = {}
        # Memoizes _resolve_name() per raw query string. This is what stops the
        # SAME ambiguity warning firing repeatedly: get_appearance, get_gender,
        # get_clothing, get_portrait, and get_locked_appearance all resolve the
        # same panel name independently, so without caching one ambiguous name in
        # one panel logs the same warning up to 5 times. Cache key is the raw
        # input string (case-sensitive — different surface forms are tracked
        # separately on purpose, since "Elena" and "Elena Morales" may resolve
        # differently).
        self._resolution_cache: Dict[str, str] = {}
        # Every name that reached "ambiguous" or "no match" at least once, with a
        # count and the candidates seen — for end-of-run visibility (see
        # get_unresolved_names()) and as the input to an optional one-time LLM
        # disambiguation pass (resolve_ambiguous_character_names in
        # comic_book_art_director.py) rather than a silent in-the-loop guess.
        self._unresolved_log: Dict[str, Dict[str, Any]] = {}
        self._build_registry(characters)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_registry(self, characters: List[Character]):
        """Build the visual description registry for all characters.

        Source of truth precedence for gender and physique:
          1. The Character object's own ``gender`` / ``physical_build`` fields
             (captured at creation — most reliable).
          2. LLM enrichment that turns the raw fields into a polished, strictly
             ordered canonical portrait + outfit.
          3. Heuristic inference (only when both above are unavailable).

        Capturing gender at creation means the LLM here never has to *guess* it,
        which is the single biggest fix for the male→female drift problem.
        """
        def _safe_join(items, sep=", ") -> str:
            if not items:
                return ""
            cleaned = []
            for item in items:
                if isinstance(item, str):
                    if item.strip():
                        cleaned.append(item.strip())
                elif isinstance(item, dict):
                    extracted = (item.get('name') or item.get('text')
                                 or item.get('description') or item.get('value'))
                    if extracted:
                        cleaned.append(str(extracted).strip())
                elif item is not None:
                    cleaned.append(str(item).strip())
            return sep.join(cleaned)

        # Build the LLM input. Pass the authoritative gender + physical_build so
        # the model polishes rather than invents.
        char_descriptions = []
        # Track the creation-time gender so we can enforce it over the LLM output.
        creation_gender: Dict[str, str] = {}
        for char in characters:
            traits_str = _safe_join(char.traits)
            cg = self._normalize_gender(getattr(char, 'gender', ''))
            if cg:
                creation_gender[char.name] = cg
            char_descriptions.append(
                f"Name: {char.name}, Age: {char.age}, Role: {char.role}, "
                f"Gender: {getattr(char, 'gender', '') or '(unspecified)'}, "
                f"PhysicalBuild: {getattr(char, 'physical_build', '') or '(unspecified)'}, "
                f"SignatureOutfit: {char.appearance or '(unspecified)'}, "
                f"Traits: {traits_str}"
            )

        prompt = (
            "You are a character model-sheet artist for a graphic novel. For each "
            "character below, produce a CANONICAL visual specification that will be "
            "injected IDENTICALLY into every illustrated panel the character appears "
            "in, so they look like the same person every time.\n\n"
            "Return FIVE fields per character:\n\n"
            "1. gender — ONE word: woman, man, girl, boy, nonbinary person. "
            "Use the character's stated Gender if given; never contradict it.\n\n"
            "2. portrait — the character's PERMANENT physical identity, written as a "
            "comma-separated descriptor in this EXACT ORDER (this order is critical "
            "for image-model consistency):\n"
            "   <gender>, <apparent age>, <skin tone>, <body frame/height>, "
            "<hair color + length + style>, <eye color>, <1-2 distinctive facial features>.\n"
            "   Example: 'woman, late 30s, warm brown skin, tall and lean, "
            "short black curls graying at the temples, dark brown eyes, "
            "thin scar across the left eyebrow'.\n"
            "   Use the stated PhysicalBuild when provided. Do NOT include clothing, "
            "emotions, poses, or backstory here — identity only.\n\n"
            "3. clothing — the character's DEFAULT outfit: garment names, colors, "
            "materials, fit, accessories (30-50 words). Use the stated SignatureOutfit.\n\n"
            "4. visual — a single flowing sentence combining portrait + clothing, "
            "for general use (50-70 words).\n\n"
            "5. palette — 3-4 hex color codes that define this character's look.\n\n"
            "6. gaze — how THIS character's eyes carry feeling: their default "
            "expressive 'tell' (one short phrase). The eyes are the gateway to the "
            "soul, and this makes close-ups read as THIS person. Examples: 'quick "
            "bright eyes that give everything away', 'eyes that go flat and "
            "unreadable when guarded', 'a steady, searching gaze that rarely blinks', "
            "'warm crinkling eyes always a half-second from a smile'. Identity-level "
            "and permanent — NOT a momentary emotion.\n\n"
            "Return ONLY a JSON object — no markdown, no preamble:\n"
            '{"CharName": {"gender": "woman", "portrait": "woman, late 30s, …", '
            '"clothing": "…", "visual": "…", "palette": ["#hex1", …], '
            '"gaze": "…"}}\n\n'
            "Characters:\n" + "\n".join(char_descriptions)
        )

        response = get_openai_prompt_response(
            prompt, temperature=0.2, openai_model=openai_model, use_grok=USE_GROK
        )
        parsed = parse_json_response(response)

        if parsed and isinstance(parsed, dict):
            for name, data in parsed.items():
                if isinstance(data, dict):
                    portrait = str(data.get('portrait', '')).strip()
                    clothing = str(data.get('clothing', '')).strip()
                    visual   = str(data.get('visual', '')).strip()
                    gender   = self._normalize_gender(data.get('gender', ''))
                    palette  = data.get('palette', [])
                    # ENFORCE creation-time gender over LLM output — the creation
                    # field is authoritative and prevents the LLM from re-guessing.
                    if name in creation_gender:
                        gender = creation_gender[name]
                    elif gender not in self._GENDER_TOKENS:
                        gender = self._infer_gender_from_text(portrait or visual)

                    # Guarantee the portrait starts with the gender token.
                    portrait = self._ensure_gender_prefix(portrait or visual, gender)

                    self.portrait_map[name]      = portrait
                    self.clothing_map[name]      = clothing
                    self.gender_map[name]        = gender
                    self.registry[name]          = visual or portrait
                    self.locked_appearance[name] = self._build_locked_clause(
                        gender, portrait, clothing
                    )

                    if isinstance(palette, list):
                        self.color_palettes[name] = ', '.join(str(p) for p in palette if p)
                    elif isinstance(palette, str):
                        self.color_palettes[name] = palette

                    gaze = str(data.get('gaze', '') or '').strip()
                    if gaze:
                        self.gaze_map[name] = gaze

                elif isinstance(data, str):
                    gender = creation_gender.get(name) or self._infer_gender_from_text(data)
                    portrait = self._ensure_gender_prefix(data, gender)
                    self.portrait_map[name]      = portrait
                    self.gender_map[name]        = gender
                    self.registry[name]          = data
                    self.locked_appearance[name] = self._build_locked_clause(
                        gender, portrait, ''
                    )
        else:
            # Hard fallback: build directly from the Character fields.
            for char in characters:
                gender = (creation_gender.get(char.name)
                          or self._infer_gender_from_text(
                              f"{getattr(char, 'physical_build', '')} {char.appearance}"))
                base = (getattr(char, 'physical_build', '')
                        or char.appearance
                        or f"a {char.age}-year-old {char.role}")
                portrait = self._ensure_gender_prefix(base, gender)
                self.portrait_map[char.name]      = portrait
                self.gender_map[char.name]        = gender
                self.clothing_map[char.name]      = char.appearance or ''
                self.registry[char.name]          = char.appearance or portrait
                self.locked_appearance[char.name] = self._build_locked_clause(
                    gender, portrait, char.appearance or ''
                )

        # ── Sanitize baseline clothing (general appropriateness pass) ─────────
        # A character's DEFAULT outfit must be actual clothing, never an
        # undressed/intimate descriptor. Stories legitimately undress characters
        # in intimate scenes — that is handled per-scene by the appearance
        # continuity tracker — but the BASELINE (what a character reverts to and
        # wears in establishing/public shots) must be clothed. A nude/lingerie
        # baseline is the root cause of a character appearing undressed in
        # public settings, because every scene-change reset falls back to it.
        self._sanitize_baseline_clothing()

    def _sanitize_baseline_clothing(self) -> None:
        """Replace any undressed/intimate DEFAULT outfit with neutral attire.

        Scans every character's baseline clothing for undressed/intimate
        descriptors (nude, lingerie, just a towel, etc.). When found, the
        baseline is replaced with a neutral, fully-covered default and the
        locked-appearance clause is rebuilt so the canonical portrait used in
        every prompt no longer carries the undressed baseline. Identity (face,
        body, hair, gender) is untouched — only the default wardrobe changes.
        """
        # Local copy of the exposed-clothing cues (kept in sync conceptually with
        # the art director's _EXPOSED_CLOTHING_CUES; duplicated here to avoid a
        # cross-module import at registry-build time).
        _exposed_cues = (
            'nude', 'naked', 'no clothing', 'no clothes', 'unclothed', 'undressed',
            'bare body', 'fully nude', 'completely nude', 'topless', 'bottomless',
            'wearing nothing', 'in nothing but', 'just a towel', 'only a towel',
            'just underwear', 'only underwear', 'in lingerie', 'lingerie only',
            'open robe', 'just a bra', 'just panties', 'in only her underwear',
            'in only his underwear', 'nothing on',
        )
        _NEUTRAL_DEFAULT = "casual everyday clothes, fully covered"
        for name, clothing in list(self.clothing_map.items()):
            cl = (clothing or '').lower()
            if clothing and any(cue in cl for cue in _exposed_cues):
                self.clothing_map[name] = _NEUTRAL_DEFAULT
                portrait = self.portrait_map.get(name, '')
                gender = self.gender_map.get(name, '')
                # Rebuild the locked clause with the neutral baseline so prompts
                # never inject the undressed default.
                try:
                    self.locked_appearance[name] = self._build_locked_clause(
                        gender, portrait, _NEUTRAL_DEFAULT
                    )
                except Exception:
                    pass
                logger.info(
                    f"  [Wardrobe] Baseline outfit for {name} was undressed/"
                    f"intimate; replaced with neutral default."
                )


    @classmethod
    def _normalize_gender(cls, raw) -> str:
        """Coerce a raw gender value to a valid token, or '' if unrecognized."""
        g = str(raw or '').lower().strip()
        if g in cls._GENDER_TOKENS:
            return g
        # Map common synonyms.
        synonyms = {
            'female': 'woman', 'f': 'woman', 'male': 'man', 'm': 'man',
            'nonbinary': 'nonbinary person', 'non-binary': 'nonbinary person',
            'enby': 'nonbinary person', 'androgynous': 'nonbinary person',
        }
        return synonyms.get(g, '')

    @classmethod
    def _ensure_gender_prefix(cls, text: str, gender: str) -> str:
        """Guarantee the descriptor starts with the correct gender token.

        If the text already begins with a DIFFERENT gender token (e.g. the LLM
        wrote "man, …" but the authoritative gender is "woman"), that wrong
        token is stripped first so the result isn't contradictory like
        "woman, man, …".
        """
        text = (text or '').strip().strip(',').strip()
        g = gender or 'person'
        if not text:
            return g
        low = text.lower()
        if low.startswith(g.lower()):
            return text
        # Strip any leading (wrong) gender token + following comma/space.
        for tok in sorted(cls._GENDER_TOKENS, key=len, reverse=True):
            if low.startswith(tok):
                # Remove the token and any immediately following separator.
                stripped = text[len(tok):].lstrip(' ,;')
                return f"{g}, {stripped}" if stripped else g
        return f"{g}, {text}"

    @staticmethod
    def _infer_gender_from_text(text: str) -> str:
        """Best-effort gender inference; returns 'person' when ambiguous."""
        lower = (text or '').lower()
        female_signals = {'woman', 'female', 'girl', ' she ', ' her ', 'lady', 'mother', 'sister'}
        male_signals   = {'man', 'male', 'boy', ' he ', ' his ', 'gentleman', 'father', 'brother'}
        f_score = sum(1 for w in female_signals if w in lower)
        m_score = sum(1 for w in male_signals   if w in lower)
        if f_score > m_score:
            return 'woman'
        if m_score > f_score:
            return 'man'
        return 'person'

    @staticmethod
    def _build_locked_clause(gender: str, portrait: str, clothing: str) -> str:
        """Build the canonical gender-first image-prompt clause.

        Format: "<portrait>. Wearing <clothing>."
        The portrait already begins with the gender token (highest weight), so
        the diffusion model cannot override sex via pose or action cues.
        """
        portrait = (portrait or '').strip()
        if not portrait:
            portrait = gender or 'person'
        clause = portrait
        if clothing:
            # Compress the clothing into concise comma-separated phrasing — this
            # clause is injected into the image prompt for every panel the
            # character appears in, so trimming grammatical filler here saves
            # tokens on every generation while preserving all visual detail. The
            # portrait (identity) is left untouched.
            compressed = _compress_clothing_text(clothing.strip())
            clause = f"{portrait}. Wearing {compressed}"
        return clause

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    # Tokens that are too generic to safely identify a character on their own.
    # Without this guard, a panel description or characters_in_frame entry
    # that uses only a shared title/rank ("Captain", "Doctor", "the Witch")
    # would resolve to WHICHEVER registered character happens to share that
    # first token — picked arbitrarily by dict iteration order, not by who
    # the scene actually means. That is the exact failure mode behind
    # characters appearing to "swap" mid-scene: the wrong character's locked
    # gender/portrait/outfit gets injected into the prompt with no warning.
    _NAME_RESOLUTION_STOPWORDS = frozenset({
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their',
        # Common titles/ranks/honorifics that frequently prefix a surname and
        # are shared across multiple characters (siblings, military units,
        # ensembles with a rank structure, etc.)
        'captain', 'major', 'colonel', 'general', 'sergeant', 'lieutenant',
        'commander', 'admiral', 'chief', 'officer', 'detective', 'agent',
        'doctor', 'dr', 'professor', 'prof', 'nurse', 'mister', 'mr', 'mrs',
        'ms', 'miss', 'sir', 'madam', 'lord', 'lady', 'king', 'queen',
        'prince', 'princess', 'father', 'mother', 'brother', 'sister',
        'uncle', 'aunt', 'grandma', 'grandpa',
    })

    def _resolve_name(self, name: str) -> str:
        """Resolve a possibly-partial character name to a registry key.

        Memoized entry point — see ``_resolve_name_uncached`` for the actual
        matching logic. Caching here is what stops one ambiguous name in one
        panel from logging the same warning multiple times: every getter
        (get_appearance, get_locked_appearance, get_portrait, get_gender,
        get_clothing) calls this independently for the same raw name, so without
        a cache the matching tiers — and any warning they log — re-run on every
        single getter call instead of once.
        """
        if not name:
            return name
        q = str(name).strip()
        if q in self._alias_map:
            return self._alias_map[q]
        if q in self._resolution_cache:
            return self._resolution_cache[q]
        resolved = self._resolve_name_uncached(q)
        self._resolution_cache[q] = resolved
        return resolved

    def _surname_tokens(self, key: str) -> List[str]:
        """All tokens of a registry key after the first (the 'surname' part,
        which may be more than one word: 'van der Berg', 'Al-Rashid Khan')."""
        parts = key.split()
        return [p.lower() for p in parts[1:]] if len(parts) > 1 else []

    def _resolve_name_uncached(self, q: str) -> str:
        """The actual matching logic (see _resolve_name for the public/cached
        entry point and the alias map checked before this ever runs).

        Resolution order (first UNAMBIGUOUS hit wins):
          1. Exact match.
          2. Case-insensitive exact match.
          3. First-name match in either direction — but ONLY when exactly one
             registry key matches, and the matching token isn't a generic
             title/stopword (see ``_NAME_RESOLUTION_STOPWORDS``).
          3.5 FULL-NAME (surname) disambiguation — when tier 3 finds MULTIPLE
             first-name candidates (e.g. query "Elena Morales" against
             registered "Elena Petrova" and "Elena Vasquez"), use the query's
             remaining tokens (the surname) to break the tie instead of
             refusing outright:
               a. exact surname-token match against exactly one candidate;
               b. otherwise a clear fuzzy winner — similarity above a high
                  floor AND a solid margin over the runner-up, so a genuine
                  near-miss spelling ("Petrova" vs "Petrov") resolves but two
                  close-but-different surnames do not.
             If neither produces a single confident winner, the name is still
             genuinely ambiguous (or refers to a character outside the
             registry entirely) and resolution correctly refuses to guess.
          4. Substring containment (query within key or key within query) —
             also ONLY when exactly one registry key matches.

        Whenever a tier finds MORE THAN ONE candidate (and 3.5 can't break the
        tie), that tier is rejected as ambiguous and resolution falls through to
        the next tier (or to "no match") rather than guessing — silently
        picking the wrong character is worse than failing to resolve at all.

        Returns the matched registry key, or the original name if no
        unambiguous match was found.
        """
        q_low = q.lower()
        if q in self.registry:
            return q

        keys = list(self.registry.keys())

        # 2. case-insensitive exact
        ci_matches = [k for k in keys if k.lower() == q_low]
        if len(ci_matches) == 1:
            return ci_matches[0]
        elif len(ci_matches) > 1:
            # Shouldn't happen (registry keys should be unique case-
            # insensitively) but if it does, don't guess.
            self._log_unresolved(
                q, ci_matches,
                f"matched {len(ci_matches)} registry keys case-insensitively "
                f"({ci_matches}); treating as unresolved."
            )
            return q

        # 3. first-name match (either direction) — require a SINGLE candidate
        # and refuse to match on a generic title/stopword token.
        q_first = q_low.split()[0] if q_low.split() else q_low
        if q_first not in self._NAME_RESOLUTION_STOPWORDS and len(q_first) >= 2:
            first_name_matches = []
            for k in keys:
                k_low = k.lower()
                k_first = k_low.split()[0] if k_low.split() else k_low
                if k_first in self._NAME_RESOLUTION_STOPWORDS:
                    continue  # never match on the KEY's title token either
                if q_first == k_first or q_first == k_low or q_low == k_first:
                    first_name_matches.append(k)
            if len(first_name_matches) == 1:
                return first_name_matches[0]
            elif len(first_name_matches) > 1:
                # 3.5 — try the query's remaining tokens (the surname) before
                # refusing. This is exactly "full name resolution": the query
                # shares a first name with several characters, but if it ALSO
                # carries a surname, that surname is usually enough to tell
                # them apart even though tier 3 only looked at the first token.
                winner = self._disambiguate_by_surname(q_low, q_first, first_name_matches)
                if winner:
                    return winner
                self._log_unresolved(
                    q, first_name_matches,
                    f"ambiguously matches {first_name_matches} on first-name/"
                    f"title token '{q_first}'; refusing to guess — treating as "
                    f"unresolved for this lookup so the wrong character's "
                    f"appearance is never substituted in."
                )
                return q

        # 4. substring containment (guard against trivially short tokens) —
        # require a SINGLE candidate.
        if len(q_low) >= 3:
            substring_matches = [
                k for k in keys
                if q_low in k.lower() or k.lower() in q_low
            ]
            if len(substring_matches) == 1:
                return substring_matches[0]
            elif len(substring_matches) > 1:
                self._log_unresolved(
                    q, substring_matches,
                    f"ambiguously matches {substring_matches} by substring "
                    f"containment; refusing to guess — treating as unresolved "
                    f"for this lookup."
                )
                return q

        return q  # no unambiguous match — return as-is (getters will yield "")

    def _disambiguate_by_surname(self, q_low: str, q_first: str,
                                  candidates: List[str]) -> Optional[str]:
        """Break a first-name tie using the query's remaining tokens.

        Returns the single winning registry key, or None if no confident,
        unambiguous winner emerges (in which case the caller still refuses to
        guess — this only ever ADDS a resolution path, never removes the
        existing safety).
        """
        remainder = q_low[len(q_first):].strip()
        if not remainder:
            return None  # query genuinely is just the first name — no surname to use

        # (a) exact surname-token match.
        exact = [k for k in candidates
                 if remainder in self._surname_tokens(k)
                 or ' '.join(self._surname_tokens(k)) == remainder]
        if len(exact) == 1:
            logger.info(
                f"[NameResolve] '{q_low}' disambiguated to '{exact[0]}' via "
                f"exact surname match (full-name resolution)."
            )
            return exact[0]
        if len(exact) > 1:
            return None  # remainder matches multiple candidates — still ambiguous

        # (b) fuzzy surname match — a clear winner only.
        import difflib
        scored = []
        for k in candidates:
            k_remainder = ' '.join(self._surname_tokens(k))
            if not k_remainder:
                continue
            ratio = difflib.SequenceMatcher(None, remainder, k_remainder).ratio()
            scored.append((ratio, k))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        best_ratio, best_key = scored[0]
        runner_up_ratio = scored[1][0] if len(scored) > 1 else 0.0
        # Require both a high absolute similarity AND a solid margin over the
        # next-best candidate, so "Petrova" vs "Petrov"/"Vasquez" resolves but
        # two genuinely close-but-different surnames do not.
        if best_ratio >= 0.82 and (best_ratio - runner_up_ratio) >= 0.15:
            logger.info(
                f"[NameResolve] '{q_low}' disambiguated to '{best_key}' via "
                f"fuzzy surname match (full-name resolution, similarity "
                f"{best_ratio:.2f} vs runner-up {runner_up_ratio:.2f})."
            )
            return best_key
        return None

    def _log_unresolved(self, query: str, candidates: List[str], message: str) -> None:
        """Log an ambiguous/unresolved name ONCE per unique query string (the
        cache in _resolve_name already prevents re-entry for repeat lookups of
        the same exact string, but this also tracks it for end-of-run
        visibility via get_unresolved_names())."""
        entry = self._unresolved_log.setdefault(
            query, {'count': 0, 'candidates': candidates}
        )
        entry['count'] += 1
        logger.warning(f"[NameResolve] '{query}' {message}")

    # ------------------------------------------------------------------
    # Alias registration & diagnostics
    # ------------------------------------------------------------------

    def register_alias(self, alias: str, canonical_key: str) -> bool:
        """Permanently map ``alias`` to an existing registry key.

        Use this once an ambiguous or unresolved name has been disambiguated
        with confidence — typically by ``resolve_ambiguous_character_names()``
        (an explicit, context-informed LLM pass; see comic_book_art_director.py)
        or by a human editing the script. Every subsequent lookup of ``alias``
        through any getter resolves instantly and correctly, with no further
        matching or warnings.

        Returns False (and registers nothing) if ``canonical_key`` is not an
        actual registry entry — refusing to guess applies here too.
        """
        if canonical_key not in self.registry:
            logger.warning(
                f"[NameResolve] register_alias('{alias}', '{canonical_key}') "
                f"refused: '{canonical_key}' is not a registered character."
            )
            return False
        self._alias_map[str(alias).strip()] = canonical_key
        # Clear any stale cached/unresolved state for this exact string so the
        # new alias takes effect immediately.
        self._resolution_cache.pop(alias, None)
        self._unresolved_log.pop(alias, None)
        logger.info(f"[NameResolve] Registered alias '{alias}' -> '{canonical_key}'.")
        return True

    def get_unresolved_names(self) -> Dict[str, Dict[str, Any]]:
        """Every distinct name that failed to resolve unambiguously at least
        once during this run, with how many times it was looked up and which
        registry keys it collided with. Intended for end-of-run diagnostics and
        as the input to an optional disambiguation pass — never consumed
        automatically in the hot path."""
        return dict(self._unresolved_log)

    def get_appearance(self, name: str) -> str:
        """Return the plain visual description (legacy API)."""
        key = self._resolve_name(name)
        return self.registry.get(key, "")

    def get_locked_appearance(self, name: str) -> str:
        """Return the canonical portrait + outfit clause for image prompts.

        Prefer this over ``get_appearance()``. Starts with the explicit gender
        token and uses a fixed attribute order so the same character renders
        consistently across every panel. Names are resolved fuzzily so a
        first-name reference still finds the full-name registry entry.

        The character's colour palette (3-4 hex codes capturing their
        signature look) is appended when available.  Diffusion models respond
        strongly to explicit hex palette cues — this is one of the most
        reliable free levers for reducing cross-panel identity drift without
        any extra generation cost.
        """
        key = self._resolve_name(name)
        clause = self.locked_appearance.get(key, self.registry.get(key, ""))
        palette = self.color_palettes.get(key, '')
        if palette and clause:
            clause = f"{clause}. Colour palette: {palette}."
        gaze = self.gaze_map.get(key, '')
        if gaze and clause:
            clause = f"{clause} Eyes: {gaze}."
        return clause

    def get_portrait(self, name: str) -> str:
        """Return the canonical fixed-order physical portrait (identity only)."""
        key = self._resolve_name(name)
        return self.portrait_map.get(key, self.registry.get(key, ""))

    def get_gender(self, name: str) -> str:
        """Return the explicit gender token for a character."""
        key = self._resolve_name(name)
        return self.gender_map.get(key, "")

    def get_clothing(self, name: str) -> str:
        """Return the base/default outfit description for a character."""
        key = self._resolve_name(name)
        return self.clothing_map.get(key, "")

    def get_characters_in_text(self, text: str) -> str:
        appearances = []
        for name, desc in self.registry.items():
            first_name = name.split()[0] if name else ''
            if first_name and (first_name.lower() in text.lower() or name.lower() in text.lower()):
                palette = self.color_palettes.get(name, '')
                entry = f"{name}: {desc}"
                if palette:
                    entry += f" (palette: {palette})"
                appearances.append(entry)
        return "; ".join(appearances) if appearances else ""


@dataclass
class Chapter:
    """Structured chapter — replaces the old 'Title\\n\\nBody' string format."""
    number: int
    title: str = ""
    body: str = ""
    epigraph: Optional[str] = None
    epigraph_attribution: Optional[str] = None
    is_epilogue: bool = False
    sanitization_log: List[str] = field(default_factory=list)
    
    @property
    def word_count(self) -> int:
        return len(self.body.split())
    
    @property
    def display_heading(self) -> str:
        if self.is_epilogue:
            return "Epilogue"
        return f"Chapter {self.number}: {self.title}" if self.title else f"Chapter {self.number}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def _coerce_to_string_list(value) -> List[str]:
    """Coerce arbitrary LLM output into a clean list of strings.
    
    Handles every shape an LLM might return for an array field:
      - list of strings:     ['brave', 'reckless']        -> as-is
      - list with dicts:     ['brave', {'name': 'x'}]     -> ['brave', 'x']
      - dict alone:          {'a': 'brave', 'b': 'wise'}  -> ['brave', 'wise']
      - string:              'brave, reckless'            -> ['brave, reckless']
      - None / empty:        None                          -> []
      - nested list:         [['a', 'b'], 'c']            -> ['a', 'b', 'c']
      - list with None:      ['a', None, 'b']             -> ['a', 'b']
    
    Used for traits, knowledge_domains, signature_habits, intelligence_markers,
    and any other field that should be a flat list of strings.
    """
    if value is None or value == "":
        return []
    
    if isinstance(value, list):
        result = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    result.append(stripped)
            elif isinstance(item, dict):
                # LLM mistake: trait as {'name': 'brave', 'evidence': '...'}
                extracted = (item.get('name') or item.get('trait')
                             or item.get('text') or item.get('value')
                             or item.get('description'))
                if extracted:
                    result.append(str(extracted).strip())
                else:
                    # Last resort: first scalar value
                    for v in item.values():
                        if v and isinstance(v, (str, int, float)):
                            result.append(str(v).strip())
                            break
            elif isinstance(item, list):
                result.extend(_coerce_to_string_list(item))
            else:
                stringified = str(item).strip()
                if stringified:
                    result.append(stringified)
        return [r for r in result if r]
    
    if isinstance(value, dict):
        result = []
        for v in value.values():
            if isinstance(v, str) and v.strip():
                result.append(v.strip())
            elif isinstance(v, (int, float)):
                result.append(str(v))
        return result
    
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    
    return [str(value).strip()]
    
def _coerce_event(event) -> str:
    """Coerce one item from a storyboard/plot_points events array to a string."""
    if isinstance(event, str):
        return event
    if isinstance(event, dict):
        return (event.get('event') or event.get('description') or event.get('text')
                or event.get('name') or event.get('summary')
                or ' '.join(str(v) for v in event.values() if v)) or ''
    if event is None:
        return ''
    return str(event)


def _safe_events(events) -> list:
    """Return a clean list[str] from whatever shape an 'events' field has."""
    if not events:
        return []
    raw = events if isinstance(events, list) else [events]
    return [s for s in (_coerce_event(e) for e in raw) if s]


def sanitize_text_for_prompt(text: str) -> str:
    """Convert smart quotes and special chars to ASCII for prompt use."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2026', '...').replace('\u00a0', ' ')
    return text.encode('ascii', errors='ignore').decode('ascii')


def parse_json_response(response: str):
    """Robustly parse JSON from an LLM response that may have markdown fences."""
    if not response:
        return None
    response = response.strip()
    response = re.sub(r'^```(?:json)?\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\s*```$', '', response).strip()
    if not response:
        return None
    try:
        result = json_repair.loads(response)
        if result == "" or result is None:
            return None
        if isinstance(result, list):
            result = _flatten_list(result)
        return result
    except Exception:
        return None


def _flatten_list(lst: list) -> list:
    """Flatten nested lists; drop non-dicts if mostly dicts."""
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(_flatten_list(item))
        else:
            flat.append(item)
    has_dicts = any(isinstance(x, dict) for x in flat)
    has_non_dicts = any(not isinstance(x, dict) for x in flat)
    if has_dicts and has_non_dicts:
        dict_count = sum(1 for x in flat if isinstance(x, dict))
        if dict_count > len(flat) * 0.5:
            flat = [x for x in flat if isinstance(x, dict)]
    return flat


def create_directory_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def cleanup_special_chars(name):
    cleanname = name.replace(" ", "_").replace(":", "_")
    cleanname = cleanname.replace("!", "").replace("&", "").replace("'", "")
    cleanname = cleanname.replace("\u2022", "").replace("\"", "")
    cleanname = re.sub(r'[<>:"/\\|?*]', '', cleanname)
    return cleanname


def reset_memory(dev=None):
    """Free GPU memory."""
    gc.collect()
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            target_device = dev if dev else device
            torch.cuda.reset_peak_memory_stats(target_device)
            torch.cuda.reset_accumulated_memory_stats(target_device)
        except Exception:
            pass


def release_image_vram():
    """Return per-image GPU memory to the allocator after one image is made.

    Called once per generated image. Every panel variant, cover, and
    act-break card routes through gen_ImageZ_image (below), which calls this
    just before returning, so a single chokepoint covers all of them.

    Why this is separate from reset_memory(): reset_memory() also calls
    reset_peak_memory_stats() / reset_accumulated_memory_stats(), which reset
    only the *reported* statistics. That makes torch.cuda.max_memory_allocated()
    drop to zero without any VRAM actually being freed — the likely reason peak
    usage looked like it "reset" inconsistently. This function does the real
    work instead, in the order that actually reclaims memory:

      1. synchronize() — wait for the just-finished diffusion kernels to
         complete so their memory is genuinely reclaimable. empty_cache() skips
         blocks still tied to in-flight async work, so without this the free
         can be a no-op.
      2. gc.collect() — break any reference cycles inside the pipeline that pin
         CUDA tensors (latents, attention/activation buffers) so they become
         unreferenced and therefore freeable.
      3. empty_cache() — hand the caching allocator's now-unused blocks back to
         the driver. This is what keeps *reserved* VRAM (what nvidia-smi shows)
         flat across a long multi-image run and prevents the size-varying
         fragmentation that drives OOM when panels differ in dimensions.

    Safe with enable_model_cpu_offload(): empty_cache() frees only unused cached
    blocks, so the offloaded weights and accelerate's offload hooks are left
    intact — it does not undo or interfere with offloading.
    """
    if not (_HAS_TORCH and torch.cuda.is_available()):
        gc.collect()
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def save_objects(data, save_path='saved_data.pkl'):
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {save_path}")
    except Exception as e:
        print(f"Error saving: {e}")


def load_objects(load_path='saved_data.pkl'):
    if not os.path.exists(load_path):
        print(f"No saved data at {load_path}.")
        return None
    try:
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {load_path}")
        return data
    except Exception as e:
        print(f"Error loading: {e}")
        return None


# =============================================================================
# LLM CLIENT MANAGEMENT
# =============================================================================

_openai_client = None
_grok_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=openai_api_key)
    return _openai_client


def _get_grok_client():
    global _grok_client
    if _grok_client is None:
        _grok_client = OpenAI(api_key=grok_api_key, base_url="https://api.x.ai/v1")
    return _grok_client


# ── Grok prompt caching ──────────────────────────────────────────────────────
# A single book synthesis fires hundreds of LLM calls whose prompts share a
# large, identical prefix: the fixed system persona plus the repeated story /
# cast / DNA context that every outline section, per-act storyboard, dialogue
# review, and enrichment call carries. xAI automatically caches common prefixes,
# and tagging related calls with the same conversation id (x-grok-conv-id) routes
# them to the same cache so that shared prefix is served from cache — billed at a
# large discount — instead of re-processed on every call. We set one id per run
# so the whole book shares a cache partition, and record how many prompt tokens
# came back cached so the saving is visible in the logs.
import uuid as _uuid

_GROK_CONV_ID = os.environ.get("GROK_CONV_ID") or f"conv_{_uuid.uuid4().hex[:16]}"
_GROK_CACHE_STATS = {"cached_tokens": 0, "prompt_tokens": 0, "calls": 0}


def set_grok_conv_id(conv_id: str = None) -> str:
    """Set/rotate the Grok conversation id used for prompt-cache routing.

    Call once at the start of a synthesis run so all of that run's calls share a
    cache partition (maximising prefix-cache reuse). Also resets cache telemetry.
    Returns the active id.
    """
    global _GROK_CONV_ID
    _GROK_CONV_ID = conv_id or f"conv_{_uuid.uuid4().hex[:16]}"
    _GROK_CACHE_STATS.update(cached_tokens=0, prompt_tokens=0, calls=0)
    return _GROK_CONV_ID


def _grok_extra_kwargs(use_grok: bool) -> dict:
    """Per-request kwargs that enable prompt-cache routing for Grok calls."""
    if not use_grok:
        return {}
    return {"extra_headers": {"x-grok-conv-id": _GROK_CONV_ID}}


def _record_cache_usage(response) -> None:
    """Accumulate cached-prompt-token telemetry from a response (best-effort)."""
    try:
        u = getattr(response, 'usage', None)
        if u is None:
            return
        cached = getattr(u, 'cached_prompt_text_tokens', None)
        if cached is None:
            details = getattr(u, 'prompt_tokens_details', None)
            cached = getattr(details, 'cached_tokens', 0) if details else 0
        _GROK_CACHE_STATS["cached_tokens"] += int(cached or 0)
        _GROK_CACHE_STATS["prompt_tokens"] += int(getattr(u, 'prompt_tokens', 0) or 0)
        _GROK_CACHE_STATS["calls"] += 1
    except Exception:
        pass


def get_grok_cache_stats() -> dict:
    """Return aggregate prompt-cache telemetry for the current run."""
    s = dict(_GROK_CACHE_STATS)
    pt = s.get("prompt_tokens", 0) or 0
    s["cache_hit_rate"] = (s.get("cached_tokens", 0) / pt) if pt else 0.0
    return s


def _get_client(use_grok: bool = False):
    return _get_grok_client() if use_grok else _get_openai_client()


def _select_model(use_grok: bool = False, openai_model_override: str = None,
                   grok_model: str = None) -> str:
    if use_grok:
        return grok_model or grok_fast_nonreasoning_model
    else:
        return openai_model_override or _host('openai_model', 'gpt-4')


def get_openai_prompt_response(prompt, max_completion_tokens=150000,
                                temperature=0.33, openai_model=None, use_grok=None,
                                cached_prefix: str = ""):
    """Main LLM completion entry point. Handles retries + truncation warnings.

    Prompt-size safety: every prompt is passed through a hard token guard before
    the first call so an oversized context is trimmed (and logged) rather than
    rejected by the API. If the provider STILL reports a token-limit error
    (model ceilings vary, and our estimate is heuristic), we treat that as
    deterministic and SHRINK the prompt for the next attempt instead of
    sleeping and resending the identical payload — which is what previously
    burned five retries and ~60s before failing.

    cached_prefix: an optional stable preamble (system-level instructions,
        schema, rules) that is byte-identical across many calls and should be
        served from Grok's prompt cache. When provided it is prepended to
        ``prompt`` so the full payload reads: <cached_prefix>\\n\\n<prompt>.
        Grok automatically caches the longest common prefix across requests,
        so grouping the stable text at the head maximises cache hits without
        any API-specific cache-control header.
    """
    # Merge the cached prefix into the prompt so it sits at the head of the
    # payload.  Grok's automatic prefix caching then serves this stable block
    # from cache on every subsequent call that shares the same prefix, giving
    # us the cache benefit without needing a separate API parameter.
    if cached_prefix:
        prompt = cached_prefix + "\n\n" + prompt
    if use_grok is None:
        use_grok = USE_GROK
    model = _select_model(use_grok, openai_model)
    client = _get_client(use_grok)

    # First-line defence: clamp to the global ceiling before we ever send it.
    if _tb is not None:
        prompt = _tb.guard_prompt(prompt, MAX_PROMPT_TOKENS,
                                  origin="get_openai_prompt_response")

    for retry_count in range(retry_limit):
        try:
            # Grok uses max_tokens; OpenAI uses max_completion_tokens.
            # Passing the wrong key is silently ignored, removing the cap entirely.
            token_key = 'max_tokens' if use_grok else 'max_completion_tokens'
            if use_grok:
                response = client.chat.completions.create(
                    **{token_key: max_completion_tokens},
                    **_grok_extra_kwargs(use_grok),
                    reasoning_effort="none",
                    messages=[
                        {"role": "system", "content": (
                            "You are an expert novelist and editor with deep knowledge of "
                            "literary craft, pacing, character voice, and commercial fiction. "
                            "You write prose that is vivid, emotionally resonant, and commercially appealing. "
                            "You never use em-dashes. You use double spaces between sentences. "
                            "All dialogue uses straight quotation marks."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    model=model, temperature=temperature,
                )
            else:
                response = client.chat.completions.create(
                    **{token_key: max_completion_tokens},
                    **_grok_extra_kwargs(use_grok),
                    messages=[
                        {"role": "system", "content": (
                            "You are an expert novelist and editor with deep knowledge of "
                            "literary craft, pacing, character voice, and commercial fiction. "
                            "You write prose that is vivid, emotionally resonant, and commercially appealing. "
                            "You never use em-dashes. You use double spaces between sentences. "
                            "All dialogue uses straight quotation marks."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    model=model, temperature=temperature,
                )                
            _record_cache_usage(response)
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            if finish_reason == 'length':
                model_name = "Grok" if use_grok else "OpenAI"
                print(f"WARNING ({model_name}): Response truncated. Got {len(content) if content else 0} chars.")
            if content and len(content.strip()) > 0:
                return content
            else:
                model_name = "Grok" if use_grok else "OpenAI"
                print(f"Empty content on retry {retry_count+1} ({model_name}). Finish: {finish_reason}")
        except Exception as e:
            model_name = "Grok" if use_grok else "OpenAI"
            # Token-limit rejections are deterministic: resending the same payload
            # will fail identically. Shrink hard and retry immediately (no sleep).
            if _tb is not None and _tb.is_token_limit_error(e):
                before = _tb.estimate_tokens(prompt)
                shrink_to = max(2000, int(before * 0.5))
                prompt = _tb.guard_prompt(
                    prompt, shrink_to,
                    origin="get_openai_prompt_response(token-limit shrink)",
                )
                print(f"Attempt {retry_count+1} ({model_name}): token-limit error; "
                      f"shrank prompt ~{before}->~{_tb.estimate_tokens(prompt)} tokens "
                      f"and retrying immediately.")
                continue
            wait_time = min(2 ** (retry_count + 1), 30)
            print(f"Attempt {retry_count+1} error ({model_name}): {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)
    print(f"ERROR: All {retry_limit} attempts failed")
    return ""


def get_vision_prompt_response(prompt: str, image_data_url: str,
                               max_completion_tokens: int = 1000,
                               temperature: float = 0.0,
                               use_grok: Optional[bool] = None) -> str:
    """Image-in, text-out completion. For panel-consistency / QA checks only.

    Sends ONE image alongside a text prompt to a vision-capable chat model and
    returns the raw text reply (expected to be short JSON for the callers that
    use this today). Mirrors get_openai_prompt_response's client selection and
    retry/error handling, but intentionally has NO token-budget guard on the
    prompt (these calls are short, fixed-shape QA prompts, not free-form
    generation) and a much lower default max_completion_tokens.

    Built to be handed to comic_book_visual_consistency.check_variant_against_ledger
    (and similar) as its injected ``vision_fn``:
        vision_fn = lambda prompt, url: get_vision_prompt_response(prompt, url)

    Returns "" on any failure (network, auth, malformed response) so callers
    that already treat an empty/unparseable reply as "unchecked" degrade
    exactly the way they do when no vision_fn is supplied at all.
    """
    if use_grok is None:
        use_grok = USE_GROK
    try:
        model = grok_vision_model if use_grok else openai_vision_model
        client = _get_client(use_grok)
        token_key = 'max_tokens' if use_grok else 'max_completion_tokens'
        response = client.chat.completions.create(
            **{token_key: max_completion_tokens},
            **_grok_extra_kwargs(use_grok),
            messages=[
                {"role": "system", "content": (
                    "You are a precise visual QA assistant. You look at exactly "
                    "one image and answer strictly in the JSON format requested. "
                    "No commentary outside the JSON."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]},
            ],
            model=model, temperature=temperature,
        )
        _record_cache_usage(response)
        content = response.choices[0].message.content
        return content or ""
    except Exception as e:
        print(f"Vision QA call failed ({'Grok' if use_grok else 'OpenAI'}): {e}")
        return ""


def get_openai_prompt_response_reasoning(prompt, max_completion_tokens=150000,
                                         openai_model=None, use_grok=None):
    """Reasoning-mode completion (no temperature, longer thinking)."""
    if use_grok is None:
        use_grok = USE_GROK
    if use_grok:
        model = grok_fast_reasoning_model
    else:
        model = openai_model or _host('openai_model_small_reasoning', 'gpt-4')
    client = _get_client(use_grok)

    if _tb is not None:
        prompt = _tb.guard_prompt(prompt, MAX_PROMPT_TOKENS,
                                  origin="get_openai_prompt_response_reasoning")

    retry_count = 0
    while retry_count < retry_limit:
        try:
            token_key = 'max_tokens' if use_grok else 'max_completion_tokens'
            if use_grok:
                response = client.chat.completions.create(
                    **{token_key: max_completion_tokens},
                    **_grok_extra_kwargs(use_grok),
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    reasoning_effort="none",
                )
            else:
                response = client.chat.completions.create(
                    **{token_key: max_completion_tokens},
                    **_grok_extra_kwargs(use_grok),
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                )                
            _record_cache_usage(response)
            content = response.choices[0].message.content
            if content and len(content.strip()) > 0:
                return content
            else:
                model_name = "Grok" if use_grok else "OpenAI"
                print(f"Empty reasoning response ({model_name}), retrying...")
        except Exception as e:
            model_name = "Grok" if use_grok else "OpenAI"
            if _tb is not None and _tb.is_token_limit_error(e):
                before = _tb.estimate_tokens(prompt)
                shrink_to = max(2000, int(before * 0.5))
                prompt = _tb.guard_prompt(
                    prompt, shrink_to,
                    origin="get_openai_prompt_response_reasoning(token-limit shrink)",
                )
                print(f"Reasoning attempt {retry_count+1} ({model_name}): token-limit "
                      f"error; shrank prompt ~{before}->~{_tb.estimate_tokens(prompt)} "
                      f"tokens and retrying immediately.")
                retry_count += 1
                continue
            wait_time = min(2 ** (retry_count + 1), 30)
            print(f"Reasoning API Error ({model_name}): {e}. Waiting {wait_time}s...")
            retry_count += 1
            time.sleep(wait_time)
    print(f"ERROR: All {retry_limit} reasoning attempts failed")
    return "[]"


# =============================================================================
# STORY & CHARACTER GENERATION
# =============================================================================

def generate_story_idea(story_idea_str: str) -> StoryIdea:
    """Develop a raw concept into a structured StoryIdea."""
    prompt = (
        f"You are a bestselling novelist's story consultant. Develop a compelling concept.\n\n"
        f"Raw idea: '{story_idea_str}'\n\n"
        f"Return as JSON: {{\"genre\": \"...\", \"themes\": [\"...\"], "
        f"\"mood\": \"...\", \"premise\": \"...\"}}\n"
        f"JSON only."
    )
    response = get_openai_prompt_response(prompt, temperature=0.5, use_grok=USE_GROK)
    story_data = parse_json_response(response)
    if isinstance(story_data, list) and len(story_data) > 0:
        story_data = story_data[0]
    if story_data and isinstance(story_data, dict):
        return StoryIdea(
            genre=story_data.get('genre', 'Any'),
            themes=story_data.get('themes', ['Any']),
            mood=story_data.get('mood', 'Any'),
            premise=story_data.get('premise', 'Any')
        )
    return StoryIdea(genre='Any', themes=['Any'], mood='Any', premise='Any')


def identify_required_characters(plot_points, additional_roles):
    """Identify a comprehensive 3-tiered cast.
    
    Accepts either:
      - plot_points as a list of dicts (canonical novel use)
      - plot_points as a list of dicts with 'events' / 'setting' fields
    
    Returns dict with keys 'tier1', 'tier2', 'tier3' — each is a list of
    role strings the casting director identified.
    """
    formatted_plot_points = ""
    if isinstance(plot_points, list):
        for idx, point in enumerate(plot_points, 1):
            if isinstance(point, dict):
                raw_events = point.get('events', [])
                events = "; ".join(_safe_events(raw_events)) if raw_events else "unknown"
                ch = point.get('chapter', idx)
                formatted_plot_points += f"{idx}. Chapter {ch}: {events}\n"
            else:
                formatted_plot_points += f"{idx}. {point}\n"
    
    if isinstance(additional_roles, str):
        roles_list = [r.strip() for r in additional_roles.strip().split('\n') if r.strip()]
        formatted_additional_roles = "\n".join(f"- {role}" for role in roles_list)
    else:
        formatted_additional_roles = "\n".join(f"- {role}" for role in (additional_roles or []))
    
    prompt = (
        "You are a casting director. Analyze this story and identify ALL characters needed.\n\n"
        f"REQUIRED ROLES (from user):\n{formatted_additional_roles}\n\n"
        f"PLOT POINTS:\n{formatted_plot_points}\n\n"
        "Create a COMPREHENSIVE CAST with 3 tiers:\n\n"
        "TIER 1 - MAIN CHARACTERS (8-12 characters):\n"
        "  Protagonists, antagonists, love interests, core allies/mentors.\n\n"
        "TIER 2 - SUPPORTING CHARACTERS (12-20 characters):\n"
        "  Secondary allies/enemies, family, colleagues, authority figures.\n\n"
        "TIER 3 - MINOR CHARACTERS (15-25 characters):\n"
        "  Shopkeepers, guards, neighbors, one-scene named characters.\n\n"
        "Return JSON:\n"
        '{"tier1": ["role1", "role2", ...], "tier2": ["role1", ...], "tier3": ["role1", ...]}\n\n'
        "JSON only, no markdown."
    )
    
    response = get_openai_prompt_response(
        prompt, temperature=0.5, openai_model=openai_model_large, use_grok=USE_GROK
    )
    parsed = parse_json_response(response)
    if parsed and isinstance(parsed, dict):
        return {
            'tier1': [str(r) for r in (parsed.get('tier1') or [])],
            'tier2': [str(r) for r in (parsed.get('tier2') or [])],
            'tier3': [str(r) for r in (parsed.get('tier3') or [])],
        }
    return {
        'tier1': [additional_roles] if isinstance(additional_roles, str) and additional_roles else [],
        'tier2': [],
        'tier3': [],
    }


def _flatten_character_dict(char_dict: Dict) -> Dict:
    """Flatten a nested character dict into a single flat dict."""
    flat = {}
    for key, value in char_dict.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[inner_key] = inner_value
        else:
            flat[key] = value
    return flat


def generate_characters(character_tiers, max_retries=3):
    """Generate Character objects across all tiers.
    
    Accepts character_tiers as:
      - dict with keys 'tier1', 'tier2', 'tier3' (each a list of role strings)
      - a single list of role strings (treated as tier1)
    """
    BATCH_SIZE = 8   # raised from 4 — large casts (15+ chars) need bigger batches
                     # so the LLM sees enough context to make them distinct
    all_characters = []
    
    # Normalize input
    if isinstance(character_tiers, list):
        character_tiers = {'tier1': character_tiers, 'tier2': [], 'tier3': []}
    
    for tier_name in ['tier1', 'tier2', 'tier3']:
        roles = character_tiers.get(tier_name, [])
        if not roles:
            continue
        
        if tier_name == 'tier1':
            detail_level = 'FULL'
            tier_label = 'Main'
        elif tier_name == 'tier2':
            detail_level = 'MEDIUM'
            tier_label = 'Supporting'
        else:
            detail_level = 'BASIC'
            tier_label = 'Minor'
        
        print(f"    {tier_label} characters ({len(roles)} roles)...")
        
        role_batches = [roles[i:i+BATCH_SIZE] for i in range(0, len(roles), BATCH_SIZE)]
        
        # Keywords that signal a role is a non-human entity rather than a human character.
        # Used to route roles to the entity-profile prompt instead of the human-biography prompt.
        _ENTITY_KEYWORDS = (
            'angel', 'archangel', 'seraph', 'cherub', 'throne', 'dominan',
            'principalit', 'elohim', 'eloah', 'lipika', 'thetan', 'logos',
            'demon', 'lucifer', 'demiurge', 'primordial',
            'god', 'abba', 'metatron', 'raziel', 'tsaphkiel', 'tsadkiel',
            'camal', 'raphael', 'haniel', 'michael', 'gabriel', 'geanel',
            'rehael', 'aes', 'fohat', 'prana', 'kundalini',
        )
        
        def _is_entity_role(role_str: str) -> bool:
            """Return True if the role string describes a non-human entity."""
            low = role_str.lower()
            return any(kw in low for kw in _ENTITY_KEYWORDS)
        
        for batch_idx, role_batch in enumerate(role_batches):
            roles_str = ', '.join(role_batch)
            existing_context = ""
            if all_characters:
                existing_names = ', '.join([c.name for c in all_characters])
                existing_types = ', '.join([c.personality_type for c in all_characters
                                            if c.personality_type])
                existing_context = (
                    f"\nALREADY CREATED (don't duplicate names/types):\n"
                    f"Names: {existing_names}\nPersonality types: {existing_types}\n\n"
                )

            # If every role in this batch is a non-human entity, use the entity
            # profile prompt — it asks for vibrational/energy appearance and
            # spiritual function rather than human biography and clothing.
            all_entities = all(_is_entity_role(r) for r in role_batch)
            any_entities = any(_is_entity_role(r) for r in role_batch)
            
            if all_entities or (any_entities and detail_level == 'MEDIUM'):
                prompt = (
                    f"Create visual and spiritual profiles for these non-human "
                    f"entities, angels, cosmic forces, or spiritual beings.\n\n"
                    f"Entities: {roles_str}\n{existing_context}"
                    f"These are NOT human characters. Do NOT give them human biographies,\n"
                    f"human social roles, or human personality types.\n\n"
                    f"For each entity, provide (flat JSON, no nesting):\n"
                    f"- name: the entity's canonical name as given\n"
                    f"- age: 'eternal', 'primordial', or a cosmic timeframe\n"
                    f"- role: their function in the spiritual/cosmic hierarchy\n"
                    f"- gender: 'nonbinary entity' for most; use canonical gender if known\n"
                    f"- physical_build: how they appear when perceived — describe their\n"
                    f"  energetic/light form, NOT a human body. Examples:\n"
                    f"  'pure supra-vibrational white light, appearing as an intense\n"
                    f"   line or brilliant point visible only to mediums'\n"
                    f"  'a blinding column of radiance that fills the frame, featureless\n"
                    f"   and impossible to look at directly'\n"
                    f"  'a geometric fractal of golden light, constantly shifting'\n"
                    f"  'a deep wailing darkness at the edge of creation'\n"
                    f"- appearance: how they manifest visually when they choose to be\n"
                    f"  seen — their light signature, energy pattern, symbolic form.\n"
                    f"  This is their DEFAULT visual across the story. Be specific:\n"
                    f"  colors of their light, shape, scale, any symbolic elements.\n"
                    f"- traits: array of 3-4 spiritual/cosmic characteristics\n"
                    f"- backstory: 1-2 sentences on their cosmic origin and mission\n"
                    f"- personality_type: their spiritual archetype or cosmic function\n"
                    f"- speech_pattern: how they communicate — telepathy, light pulses,\n"
                    f"  compressed conceptual blocks, vibrational tone, silence, etc.\n"
                    f"- knowledge_domains: array of 2-3 cosmic domains they govern\n\n"
                    f"Return JSON array of {len(role_batch)} entity objects. No markdown."
                )
            elif detail_level == 'FULL':
                prompt = (
                    f"Create memorable character profiles - larger than life yet relatable.\n\n"
                    f"Roles: {roles_str}\n{existing_context}"
                    f"For each, provide (flat JSON, no nesting):\n"
                    f"- name (memorable, distinctive), age, role\n"
                    f"- gender: ONE word — woman, man, girl, boy, nonbinary person. "
                    f"This is REQUIRED and must be unambiguous; it locks the character's "
                    f"sex across every illustrated panel.\n"
                    f"- physical_build: a COMPLETE physical description in this EXACT order: "
                    f"apparent age, skin tone, body frame/height, hair (color + length + style), "
                    f"eye color, and 1-2 distinctive facial features. "
                    f"Example: 'late-30s, warm brown skin, tall and lean, "
                    f"short black curls graying at the temples, dark brown eyes, "
                    f"a thin scar across the left eyebrow'.\n"
                    f"- appearance: signature clothing/outfit with specific colors and "
                    f"materials (this is their default look across the story)\n"
                    f"- traits (array, 4-6, include contradictions and flaws)\n"
                    f"- backstory (2-3 sentences with specific details)\n"
                    f"- arc (how they change)\n"
                    f"- personality_type, cognitive_style, humor_style\n"
                    f"- knowledge_domains (array 2-4), coping_mechanism\n"
                    f"- social_energy, inner_world\n"
                    f"- signature_habits (array 2-3)\n"
                    f"- relationship_style, romantic_energy\n"
                    f"- creative_expression, sensory_orientation\n"
                    f"- speech_pattern (with example line)\n"
                    f"- origin: SPECIFIC place and cultural background this character is\n"
                    f"  from (city + country/region, e.g. 'Guadalajara, Mexico', 'rural\n"
                    f"  Appalachia, USA', 'Mumbai, India', 'Lagos, Nigeria'), or the\n"
                    f"  regional US background if no immigration is implied (e.g. 'small-\n"
                    f"  town Georgia', 'working-class Boston'). Ground this in the story's\n"
                    f"  setting and the role — where the cast naturally supports it, aim\n"
                    f"  for a believable RANGE of backgrounds rather than an all-neutral\n"
                    f"  cast; never force a background where the story gives no basis for\n"
                    f"  one, and never make a character's origin their only trait.\n"
                    f"\n"
                    f"VOICE PROFILE (critical - this is how the character SOUNDS on the\n"
                    f"page; make every character audibly distinct so a reader could\n"
                    f"identify the speaker with the name removed):\n"
                    f"- dialect: how THIS origin actually colours their speech on the\n"
                    f"  page - specific word choices, contractions, grammar, idioms, and\n"
                    f"  (where the character is bilingual/immigrant/first-generation)\n"
                    f"  natural CODE-SWITCHING into short phrases of their first language.\n"
                    f"  Be concrete, e.g.:\n"
                    f"    'rural Appalachian English: drops g's, reckon, a-fixin to,\n"
                    f"     double modals like might could'\n"
                    f"    'Mexican-American Spanglish: swaps in short Spanish phrases and\n"
                    f"     exclamations naturally mid-sentence (¡Ay, no manches!, órale,\n"
                    f"     mijo/mija), code-switches when emotional'\n"
                    f"    'Indian English: sing-song rhythm, \"only\"/\"na\" sentence-enders,\n"
                    f"     present-continuous where American English wouldn't, occasional\n"
                    f"     Hindi endearments or invocations (beta, arre, \"Hey Bhagwan\")'\n"
                    f"  If standard/neutral, say so plainly rather than inventing a flavor.\n"
                    f"- cadence: the MUSIC of their speech. The most vivid voices\n"
                    f"  alternate between short, punchy statements and longer, winding\n"
                    f"  clauses - describe THIS character's particular rhythm (clipped\n"
                    f"  and terse? breathless run-ons? lyrical and measured? halting?).\n"
                    f"- vocabulary_level: register and word precision, from gutter slang\n"
                    f"  to plain everyday to erudite. Expert/advanced vocabulary is\n"
                    f"  welcome where it fits - a learned character may wield words with\n"
                    f"  Emily-Dickinson-level precision, choosing the exact right word and\n"
                    f"  never padding with fluff.\n"
                    f"- verbal_tics (array 1-3): involuntary speech fingerprints - filler\n"
                    f"  words, sentence-enders, a grammatical quirk, a recurring hedge.\n"
                    f"- catchphrases (array 0-2): signature phrases the reader comes to\n"
                    f"  recognise. Empty if the character wouldn't have one.\n"
                    f"- humor_style: if they're funny, HOW - does their wit poke at\n"
                    f"  societal hypocrisy, human foolishness, or romanticized illusions\n"
                    f"  about the world? Dry irony, absurdist, self-deprecating, biting?\n"
                    f"- voice_profile: ONE flowing paragraph (40-70 words) synthesising\n"
                    f"  the above into a portrait of how this person talks.\n\n"
                    f"AUTHENTICITY RULE: render culture/dialect through real, specific word\n"
                    f"choice and idiom, with dignity - never a mocking phonetic caricature\n"
                    f"or a checklist of stereotypes. A few true, well-observed details beat\n"
                    f"a costume of clichés.\n\n"
                    f"Return JSON array of {len(role_batch)} character objects. No markdown.\n"
                    f"All voice fields are REQUIRED for human characters."
                )
            elif detail_level == 'MEDIUM':
                prompt = (
                    f"Create memorable supporting characters with distinctive voices.\n\n"
                    f"Roles: {roles_str}\n{existing_context}"
                    f"For each, provide (flat JSON):\n"
                    f"- name, age, role, traits (array, 3-4)\n"
                    f"- gender: ONE word — woman, man, girl, boy, nonbinary person (REQUIRED).\n"
                    f"- physical_build: complete physical description in this order: "
                    f"apparent age, skin tone, body frame, hair (color + style), eye color, "
                    f"one distinctive facial feature.\n"
                    f"- appearance: signature outfit with specific colors/materials\n"
                    f"- backstory (1-2 sentences)\n"
                    f"- personality_type, humor_style, knowledge_domains (array 1-2)\n"
                    f"- signature_habits (array 1-2), speech_pattern\n"
                    f"- origin: place/culture this character is from (city + country/\n"
                    f"  region, or a US regional background), grounded in the story's\n"
                    f"  setting — or 'unspecified' if it wouldn't matter for this role.\n"
                    f"VOICE (make this character audibly distinct):\n"
                    f"- dialect: concrete regional/cultural speech flowing from their\n"
                    f"  origin, rendered as on-page word choices and grammar — including\n"
                    f"  light code-switching into a first language where that fits (e.g.\n"
                    f"  a Spanish phrase, a Hindi endearment) — or 'standard/neutral'.\n"
                    f"- cadence: their speech rhythm (e.g. alternates short punchy lines\n"
                    f"  with long winding clauses; clipped; lyrical; breathless).\n"
                    f"- vocabulary_level: slang / plain / erudite - the exact right words,\n"
                    f"  no fluff.\n"
                    f"- verbal_tics (array 0-2): speech fingerprints.\n\n"
                    f"Render culture with authenticity and dignity, never caricature.\n"
                    f"Return JSON array of {len(role_batch)} character objects. No markdown."
                )
            else:
                prompt = (
                    f"Create textured minor characters - brief but memorable.\n\n"
                    f"Roles: {roles_str}\n{existing_context}"
                    f"For each, provide (flat JSON):\n"
                    f"- name, age, role, traits (array, 2)\n"
                    f"- gender: ONE word — woman, man, girl, boy, nonbinary person (REQUIRED).\n"
                    f"- physical_build: skin tone, body frame, hair (color + style), "
                    f"eye color, one distinctive feature.\n"
                    f"- appearance: signature outfit with colors\n"
                    f"- personality_type, speech_pattern (one quirk)\n\n"
                    f"Return JSON array of {len(role_batch)} character objects. No markdown."
                )
            
            batch_chars = []
            for attempt in range(1, max_retries + 1):
                response = get_openai_prompt_response(
                    prompt, temperature=0.7, openai_model=openai_model_large,
                    max_completion_tokens=16000 if detail_level == 'FULL' else 8000,
                    use_grok=USE_GROK
                )
                if not response or len(response.strip()) < 50:
                    continue
                
                characters_data = parse_json_response(response)
                if characters_data is not None and isinstance(characters_data, list):
                    characters_data = [c for c in characters_data if isinstance(c, dict)]
                    characters_data = [_flatten_character_dict(c) for c in characters_data]
                    characters_data = [
                        {k.lower().strip().replace(' ', '_'): v for k, v in c.items()}
                        for c in characters_data
                    ]
                    valid_chars = [
                        c for c in characters_data
                        if c.get('name') and len(str(c.get('name', '')).strip()) > 1
                        and str(c.get('name', '')).strip().lower() != 'unknown'
                    ]
                    if not valid_chars:
                        continue
                    
                    batch_chars = [
                        Character(
                            name=char.get('name', 'Unknown'),
                            age=char.get('age', 'Unknown'),
                            role=char.get('role', 'Unknown'),
                            traits=char.get('traits', []),
                            backstory=char.get('backstory', ''),
                            appearance=char.get('appearance', ''),
                            gender=char.get('gender', ''),
                            physical_build=char.get('physical_build', ''),
                            speech_pattern=char.get('speech_pattern', ''),
                            arc=char.get('arc', ''),
                            personality_type=char.get('personality_type', ''),
                            cognitive_style=char.get('cognitive_style', ''),
                            humor_style=char.get('humor_style', ''),
                            knowledge_domains=char.get('knowledge_domains', []),
                            coping_mechanism=char.get('coping_mechanism', ''),
                            social_energy=char.get('social_energy', ''),
                            inner_world=char.get('inner_world', ''),
                            signature_habits=char.get('signature_habits', []),
                            relationship_style=char.get('relationship_style', ''),
                            romantic_energy=char.get('romantic_energy', ''),
                            creative_expression=char.get('creative_expression', ''),
                            sensory_orientation=char.get('sensory_orientation', ''),
                            dialect=char.get('dialect', ''),
                            mythic_archetype=char.get('mythic_archetype', ''),
                            wit_level=char.get('wit_level', 'basic'),
                            creative_medium=char.get('creative_medium', ''),
                            intelligence_markers=char.get('intelligence_markers', []),
                            cadence=char.get('cadence', ''),
                            vocabulary_level=char.get('vocabulary_level', ''),
                            verbal_tics=char.get('verbal_tics', []),
                            catchphrases=char.get('catchphrases', []),
                            voice_profile=char.get('voice_profile', ''),
                            origin=char.get('origin', ''),
                        )
                        for char in valid_chars
                    ]
                    
                    for c in batch_chars:
                        c.build_voice_guide()
                    
                    names_preview = ', '.join(c.name for c in batch_chars[:3])
                    if len(batch_chars) > 3:
                        names_preview += f", +{len(batch_chars)-3} more"
                    print(f"      Batch {batch_idx + 1}/{len(role_batches)}: "
                          f"{len(batch_chars)} chars ({names_preview})")
                    break
            
            all_characters.extend(batch_chars)
    
    if not all_characters:
        print("    WARNING: All character batches failed.")
    else:
        print(f"\n    Total characters generated: {len(all_characters)}")
    
    return all_characters


def synthesize_seeds_to_story(seeds: Dict, base_idea: str = "") -> Optional[Dict]:
    """Convert creative seeds dict to a story concept."""
    if not any((seeds or {}).values()) and not base_idea:
        return None
    
    seeds_text = "\n".join([f"{k}: {v}" for k, v in (seeds or {}).items() if v])
    
    prompt = (
        "You are a story architect. Synthesize these creative seeds into a "
        "compelling novel premise.\n\n"
        f"CREATIVE SEEDS:\n{seeds_text}\n\n"
    )
    if base_idea:
        prompt += f"INITIAL CONCEPT:\n{base_idea}\n\n"
    prompt += (
        "Return JSON:\n"
        '{"genre": "...", "themes": ["...", "..."], "mood": "...", '
        '"premise": "2-3 paragraph story concept"}\n\n'
        "JSON only."
    )
    
    response = get_openai_prompt_response(
        prompt, temperature=0.7, openai_model=openai_model_large, use_grok=USE_GROK
    )
    parsed = parse_json_response(response)
    if parsed and isinstance(parsed, dict):
        return parsed
    return None


# =============================================================================
# CHARACTER GRAPH — Emotional Intelligence Engine
# =============================================================================

@dataclass
class DynamicTrait:
    """A personality trait with baseline + current activation level."""
    name: str
    baseline: float = 0.0
    current: float = 0.0
    volatility: float = 0.3
    history: List[str] = field(default_factory=list)
    
    def nudge(self, delta: float, reason: str = ""):
        self.current = max(-1.0, min(1.0, self.current + delta * self.volatility))
        if reason:
            self.history.append(f"{delta:+.2f}: {reason}")
            if len(self.history) > 20:
                self.history = self.history[-20:]
    
    def decay_toward_baseline(self, rate: float = 0.1):
        gap = self.baseline - self.current
        self.current += gap * rate


@dataclass
class SelfComponent:
    """Something a character identifies with."""
    name: str
    category: str  # physical | relational | ideological | goal | identity
    importance: int = 5
    state: str = "intact"
    attached_to: Optional[str] = None


@dataclass
class EmotionalMoment:
    """A single emotional beat at a specific chapter/scene."""
    chapter: int
    scene_index: int
    primary: str
    intensity: float
    secondary: List[str] = field(default_factory=list)
    trigger: str = ""
    felt: str = ""
    shown: str = ""
    withheld: str = ""
    subtext_tag: str = ""
    salience: float = 0.0
    reference_count: int = 0
    wound_link: bool = False
    longing_link: bool = False


@dataclass
class CharacterNode:
    """A character in the graph — rich psychological profile."""
    name: str
    role: str = ""
    archetype: str = ""
    traits: Dict[str, DynamicTrait] = field(default_factory=dict)
    self_components: List[SelfComponent] = field(default_factory=list)
    core_wound: str = ""
    core_longing: str = ""
    defense_mechanism: str = ""
    shadow: str = ""
    growth_edge: str = ""
    voice_signature: str = ""
    lexical_habits: List[str] = field(default_factory=list)
    rhythm: str = ""
    metaphor_pool: List[str] = field(default_factory=list)
    recent_emotions: List[EmotionalMoment] = field(default_factory=list)
    arc_stage: str = "act1_setup"
    arc_pressure: float = 0.0
    # ── Physical / behavioural SIGNATURE ────────────────────────────────────
    # The 1-3 distinct, recognisable tells that make this character unmistakably
    # THEMSELVES across every panel — the visual counterpart of voice_signature.
    # These may be:
    #   * tangible & permanent  — "a small crescent scar under the left eye",
    #                             "a faded compass tattoo on the right forearm"
    #   * behavioural/physical  — "tilts head left when lying", "a bark of a
    #                             laugh with the head thrown back"
    #   * intangible/emotional  — "goes very still and quiet when truly afraid",
    #                             "looks at people a half-second too long"
    # signature_look is the ONE most identity-defining item (rendered in every
    # panel where the character appears); physical_signatures holds the fuller
    # set that the amplifier and gaze passes can draw from.
    signature_look: str = ""
    physical_signatures: List[str] = field(default_factory=list)
    # How this character characteristically LOOKS AT others (baseline gaze),
    # before the relationship graph modulates it per-scene: "measures people
    # like an appraiser", "won't meet an eye it respects".
    gaze_signature: str = ""
    # ── Cast register (extraordinariness ↔ relatability spread) ─────────────
    # Deliberately assigned across the ensemble so the cast has contrast: some
    # larger-than-life "you-won't-believe-what-they-did" figures, some grounded,
    # relatable, desirable, or enviable ordinary people the reader bonds with.
    #   mythic       — legendary, larger-than-life; does the unbelievable
    #   aspirational — desirable/enviable; who the reader wishes they were
    #   relatable    — ordinary, flawed, deeply human; the reader IS them
    #   grounded     — the steady, real-world anchor; keeps the story humane
    cast_register: str = ""
    register_note: str = ""   # one line on how this register should read on the page
    # Whether this character is a natural vehicle for earned wisdom / karmic
    # reflection (an Uncle-Iroh-style mentor voice). Set at cast-spread time from
    # archetype/role. "" = ordinary voice; "mentor" = can carry a lesson where it
    # lands; "trickster-sage" = wisdom smuggled inside humour.
    wisdom_disposition: str = ""

    def top_emotions(self, n: int = 3) -> List[EmotionalMoment]:
        return sorted(self.recent_emotions, key=lambda m: m.intensity, reverse=True)[:n]

    def dominant_trait(self) -> Optional[DynamicTrait]:
        if not self.traits:
            return None
        return max(self.traits.values(), key=lambda t: abs(t.current))

    def visual_signature_lock(self, permanent_only: bool = True) -> str:
        """Render the identity-defining physical tells as an image-prompt lock.

        ``permanent_only`` keeps just the *tangible, always-visible* markers
        (scar, tattoo, distinctive feature) suitable for injecting into EVERY
        panel's Subject slot so the character stays recognisably the same person.
        When False, includes behavioural/emotional tells too (for passes that
        want the fuller palette). Returns "" when nothing is set.
        """
        items: List[str] = []
        if self.signature_look:
            items.append(self.signature_look.strip())
        for s in self.physical_signatures:
            s = str(s or "").strip()
            if not s or s == self.signature_look.strip():
                continue
            if permanent_only and not _is_tangible_marking(s):
                continue
            items.append(s)
        # De-dup preserving order; cap so the lock stays short.
        seen, out = set(), []
        for it in items:
            k = it.lower()
            if k not in seen:
                seen.add(k)
                out.append(it)
        return "; ".join(out[:3])


# Words that indicate a TANGIBLE, always-renderable physical marking (as opposed
# to a behavioural or emotional tell). Used to decide which signatures are safe
# to hard-lock into every panel.
_TANGIBLE_MARKING_TERMS = (
    "scar", "tattoo", "birthmark", "freckle", "mole", "burn", "brand",
    "piercing", "prosthetic", "missing", "cleft", "gap", "dimple", "streak",
    "patch", "eyepatch", "glasses", "spectacles", "beard", "stubble", "braid",
    "curl", "bun", "shaved", "heterochromia", "eyes", "hair", "jaw",
    "cheekbone", "nose", "brow", "lip", "chin", "forearm", "wrist", "neck",
    "shoulder", "hand", "knuckle", "temple", "collarbone",
)


def _is_tangible_marking(text: str) -> bool:
    low = str(text or "").lower()
    return any(term in low for term in _TANGIBLE_MARKING_TERMS)


@dataclass
class RelationshipEdge:
    """Directed edge: how source sees target."""
    source: str
    target: str
    trust: float = 0.0
    affection: float = 0.0
    respect: float = 0.0
    attraction: float = 0.0
    fear: float = 0.0
    resentment: float = 0.0
    envy: float = 0.0
    empathy: float = 0.0
    perceived_power: float = 0.0
    power_type: str = ""
    debt: float = 0.0
    shared_history: List[str] = field(default_factory=list)
    unspoken_truths: List[str] = field(default_factory=list)
    secrets_kept_from: List[str] = field(default_factory=list)
    grievances: List[str] = field(default_factory=list)
    gifts_given: List[str] = field(default_factory=list)
    trend: str = "stable"
    last_significant_beat: str = ""
    tom_believes_trust: float = 0.0
    tom_believes_affection: float = 0.0
    tom_believes_respect: float = 0.0
    tom_believes_attraction: float = 0.0
    tom_believes_fear: float = 0.0
    tom_notes: str = ""
    feared_trajectory: str = ""
    hoped_trajectory: str = ""
    anticipation_intensity: float = 0.0
    anticipation_updated_at_chapter: int = 0
    
    def intensity(self) -> float:
        return sum(abs(v) for v in [
            self.trust, self.affection, self.respect, self.attraction,
            self.fear, self.resentment, self.envy, self.empathy
        ]) / 8.0
    
    def label(self) -> str:
        dominant = max(
            [("love", self.affection), ("trust", self.trust), ("respect", self.respect),
             ("attraction", self.attraction), ("fear", -self.fear),
             ("resentment", -self.resentment), ("envy", -self.envy)],
            key=lambda kv: abs(kv[1])
        )
        name, val = dominant
        if abs(val) < 2:
            return "neutral"
        return f"{name}{'+' if val > 0 else '-'}({abs(val):.0f})"


@dataclass
class SceneEngine:
    """The dramatic structure of a single scene."""
    scene_index: int
    chapter_num: int
    setting: str
    characters_present: List[str]
    time_of_day: str = ""
    mood: str = ""
    desires: Dict[str, str] = field(default_factory=dict)
    obstacles: Dict[str, str] = field(default_factory=dict)
    strategies: Dict[str, str] = field(default_factory=dict)
    fears: Dict[str, str] = field(default_factory=dict)
    inciting_beat: str = ""
    turning_point: str = ""
    revelation: str = ""
    outcome: Dict[str, str] = field(default_factory=dict)
    subtext_goals: List[str] = field(default_factory=list)
    dramatic_irony: str = ""
    sensory_palette: str = ""
    edge_deltas: List[Dict[str, Any]] = field(default_factory=list)
    emotional_moments: List[EmotionalMoment] = field(default_factory=list)


@dataclass
class ChapterProcessingResult:
    """Output of one chapter's pass through the graph."""
    chapter_num: int
    scenes: List[SceneEngine] = field(default_factory=list)
    group_mood: str = ""
    coalitions: List[Dict[str, Any]] = field(default_factory=list)
    dramatic_tensions: List[str] = field(default_factory=list)
    chapter_emotional_arc: str = ""


class CharacterGraph:
    """The full character interaction graph."""
    
    def __init__(self, story_idea=None):
        self.nodes: Dict[str, CharacterNode] = {}
        self.edges: Dict[Tuple[str, str], RelationshipEdge] = {}
        self.story_idea = story_idea
        self.snapshots: List[Dict[str, Any]] = []
        self.chapter_results: Dict[int, ChapterProcessingResult] = {}
        self.global_tensions: List[str] = []
    
    def add_node(self, node: CharacterNode):
        self.nodes[node.name] = node
    
    def get_node(self, name: str) -> Optional[CharacterNode]:
        return self.nodes.get(name)
    
    def all_names(self) -> List[str]:
        return list(self.nodes.keys())
    
    def add_edge(self, edge: RelationshipEdge):
        self.edges[(edge.source, edge.target)] = edge
    
    def get_edge(self, source: str, target: str) -> Optional[RelationshipEdge]:
        return self.edges.get((source, target))
    
    def get_or_create_edge(self, source: str, target: str) -> RelationshipEdge:
        key = (source, target)
        if key not in self.edges:
            self.edges[key] = RelationshipEdge(source=source, target=target, trend="new")
        return self.edges[key]
    
    def edges_from(self, source: str) -> List[RelationshipEdge]:
        return [e for (s, _), e in self.edges.items() if s == source]
    
    def edges_to(self, target: str) -> List[RelationshipEdge]:
        return [e for (_, t), e in self.edges.items() if t == target]
    
    def significant_edges(self, threshold: float = 3.0) -> List[RelationshipEdge]:
        return [e for e in self.edges.values() if e.intensity() >= threshold]
    
    def tom_gaps(self) -> List[Dict[str, Any]]:
        gaps = []
        for (a, b), a_edge in self.edges.items():
            b_edge = self.get_edge(b, a)
            if not b_edge:
                continue
            dims = [
                ("trust", a_edge.tom_believes_trust, b_edge.trust),
                ("affection", a_edge.tom_believes_affection, b_edge.affection),
                ("respect", a_edge.tom_believes_respect, b_edge.respect),
                ("attraction", a_edge.tom_believes_attraction, b_edge.attraction),
                ("fear", a_edge.tom_believes_fear, b_edge.fear),
            ]
            for dim, believed, actual in dims:
                gap = abs(believed - actual)
                if gap >= 4.0:
                    gaps.append({
                        "observer": a, "subject": b, "dimension": dim,
                        "observer_believes": believed, "actual": actual,
                        "gap": gap,
                        "direction": "overestimate" if believed > actual else "underestimate",
                    })
        return sorted(gaps, key=lambda g: g["gap"], reverse=True)


def _call_json(prompt: str, *, temperature: float = 0.6,
                max_tokens: int = 8000, large: bool = True) -> Any:
    """Helper: AI call that must return JSON. Returns None on failure."""
    model_large = _host("openai_model_large")
    model_small = _host("openai_model")
    use_grok = _host("USE_GROK", True)
    response = get_openai_prompt_response(
        prompt=prompt, temperature=temperature, max_completion_tokens=max_tokens,
        openai_model=(model_large if large else model_small), use_grok=use_grok,
    )
    return parse_json_response(response)


def _fallback_node(character) -> CharacterNode:
    """Minimal node if AI build fails."""
    node = CharacterNode(name=character.name, role=getattr(character, "role", ""))
    node.traits["stability"] = DynamicTrait(name="stability", baseline=0.3, current=0.3)
    node.self_components.append(SelfComponent(
        name="sense of self", category="identity", importance=6
    ))
    node.voice_signature = f"Speaks as a {character.role}."
    return node


def _lightweight_minor_node(character, story_idea) -> CharacterNode:
    """Build a CharacterNode for a minor (tier3) character with a single,
    compact LLM call.

    Full ai_build_character_node calls are rich but expensive — typically 800+
    tokens of output per character. For minor characters who appear in 1-3
    scenes, a compact profile (archetype, wound, longing, voice signature, and
    relationship disposition) is sufficient for the EI graph to generate
    meaningful interaction guidance without the full psychological deep-dive.

    Falls back to _fallback_node on any failure so graph construction never
    blocks.
    """
    try:
        traits_str = ", ".join(getattr(character, "traits", []) or [])
        prompt = (
            f"You are a character analyst building a compact EI profile for a "
            f"MINOR character in a {story_idea.genre} graphic novel.\n\n"
            f"CHARACTER:\n"
            f"  Name: {character.name}\n"
            f"  Role: {character.role}\n"
            f"  Traits: {traits_str}\n"
            f"  Backstory: {getattr(character, 'backstory', '')[:200]}\n\n"
            f"STORY PREMISE: {story_idea.premise[:300]}\n\n"
            f"Produce a compact JSON profile (no nesting, all values strings "
            f"or simple lists):\n"
            f'{{\n'
            f'  "archetype": "<12 words: their dramatic function>",\n'
            f'  "core_wound": "<10 words: their deepest fear or loss>",\n'
            f'  "core_longing": "<10 words: what they secretly want>",\n'
            f'  "voice_signature": "<15 words: how they speak/sound>",\n'
            f'  "relationship_default": "<8 words: how they treat strangers>",\n'
            f'  "narrative_function": "<10 words: what they do for the story>"\n'
            f'}}\n\n'
            f"JSON only."
        )
        resp = get_openai_prompt_response(
            prompt, temperature=0.4,
            max_completion_tokens=512,
            openai_model=openai_model,  # use fast model for minor chars
            use_grok=USE_GROK,
        )
        parsed = parse_json_response(resp) or {}

        node = CharacterNode(name=character.name, role=getattr(character, "role", ""))
        node.archetype = str(parsed.get("archetype", character.role))
        node.core_wound = str(parsed.get("core_wound", ""))
        node.core_longing = str(parsed.get("core_longing", ""))
        node.voice_signature = str(parsed.get("voice_signature", f"Speaks as {character.role}."))
        node.traits["stability"] = DynamicTrait(name="stability", baseline=0.4, current=0.4)
        node.self_components.append(SelfComponent(
            name=str(parsed.get("narrative_function", "minor role")),
            category="function", importance=4
        ))
        return node
    except Exception as e:
        logger.warning(f"[EI] Minor node build failed for {character.name}: {e}")
        return _fallback_node(character)


def ai_build_character_node(character, story_idea, all_characters: List = None) -> CharacterNode:
    """Build a rich CharacterNode from a Character object using AI."""
    all_characters = all_characters or []
    cast_context = ""
    if all_characters:
        cast_context = "\nOTHER CHARACTERS:\n" + "\n".join([
            f"- {c.name} ({c.role}): {c.personality_type or ''}"
            for c in all_characters if c.name != character.name
        ][:20])
    
    traits_str = ", ".join(character.traits) if getattr(character, "traits", None) else ""
    domains = ", ".join(getattr(character, "knowledge_domains", []) or [])
    
    prompt = f"""You are a psychological profiler building a character dossier.
 
STORY: {story_idea.genre} | {story_idea.mood}
PREMISE: {story_idea.premise}
{cast_context}
 
CHARACTER:
Name: {character.name}
Age: {character.age}
Role: {character.role}
Traits: {traits_str}
Backstory: {character.backstory}
Personality: {getattr(character, 'personality_type', '')}
Knowledge: {domains}
Arc: {getattr(character, 'arc', '')}
 
Produce a JSON dossier:
 
{{
  "archetype": "one evocative phrase",
  "core_wound": "the defining hurt driving behavior (1 sentence)",
  "core_longing": "what they unconsciously want (1 sentence)",
  "defense_mechanism": "how they protect the wound (1 sentence)",
  "shadow": "what they refuse to see in themselves (1 sentence)",
  "growth_edge": "who they could become (1 sentence)",
  "traits": [
    {{"name": "trait_snake_case", "baseline": -1.0 to 1.0, "volatility": 0.0 to 1.0, "evidence": "why"}},
    ... 5-8 traits including at least one CONTRADICTION
  ],
  "self_components": [
    {{"name": "thing they identify with", "category": "physical|relational|ideological|goal|identity",
      "importance": 1-10, "state": "intact|threatened|lost|strengthened",
      "attached_to": "other character name or null"}},
    ... 6-12 entries
  ],
  "voice_signature": "2-3 sentences on how they speak. Capture the MUSIC and PRECISION of their voice: do they alternate short punchy statements with long complex clauses? Any regional dialect rendered as concrete on-page word choices? Do they wield vocabulary with surgical, Emily-Dickinson-level exactness, or speak in plain everyday slang? If funny, does their wit puncture societal hypocrisy, human foolishness, or romanticized illusions? Every character must be audibly distinct from the rest of the cast.",
  "lexical_habits": ["3-6 distinctive words or phrases"],
  "rhythm": "clipped | flowing | halting | lyrical | terse | alternating-short-and-long | etc.",
  "metaphor_pool": ["3-5 domains they pull imagery from"],
  "signature_look": "the ONE most identity-defining, ALWAYS-VISIBLE physical marker that makes this character instantly recognisable in every panel — a specific permanent feature with a precise location (e.g. 'a thin white scar bisecting the left eyebrow', 'a faded blue swallow tattoo on the right forearm', 'mismatched eyes, one grey one hazel'). Must be concrete and drawable. Never a costume item (those change).",
  "physical_signatures": ["2-4 distinct recognisable TELLS beyond the signature_look. Mix registers: a behavioural tell ('tilts head left when lying'), a characteristic gesture or laugh ('a soundless laugh, all shoulders'), AND an emotional/physical tell ('goes utterly still when truly afraid'). These make the character feel like one continuous person."],
  "gaze_signature": "how this character characteristically LOOKS AT other people as a baseline — the way their attention lands (e.g. 'measures everyone like a threat to be priced', 'meets eyes a half-second too long', 'never looks directly at those they love')",
  "arc_stage": "act1_setup",
  "arc_pressure": 0.0
}}
 
Return ONLY the JSON."""
    
    data = _call_json(prompt, temperature=0.7, max_tokens=4000)
    if not data or not isinstance(data, dict):
        return _fallback_node(character)
    
    node = CharacterNode(
        name=character.name,
        role=character.role,
        archetype=data.get("archetype", ""),
        core_wound=data.get("core_wound", ""),
        core_longing=data.get("core_longing", ""),
        defense_mechanism=data.get("defense_mechanism", ""),
        shadow=data.get("shadow", ""),
        growth_edge=data.get("growth_edge", ""),
        voice_signature=data.get("voice_signature", ""),
        lexical_habits=data.get("lexical_habits", []) or [],
        rhythm=data.get("rhythm", ""),
        metaphor_pool=data.get("metaphor_pool", []) or [],
        signature_look=str(data.get("signature_look", "") or ""),
        physical_signatures=_coerce_to_string_list(data.get("physical_signatures", [])),
        gaze_signature=str(data.get("gaze_signature", "") or ""),
        arc_stage=data.get("arc_stage", "act1_setup"),
        arc_pressure=float(data.get("arc_pressure", 0.0) or 0.0),
    )
    
    for t in data.get("traits", []) or []:
        try:
            name = t.get("name", "").strip()
            if not name:
                continue
            baseline = float(t.get("baseline", 0.0))
            volatility = float(t.get("volatility", 0.3))
            node.traits[name] = DynamicTrait(
                name=name,
                baseline=max(-1.0, min(1.0, baseline)),
                current=max(-1.0, min(1.0, baseline)),
                volatility=max(0.0, min(1.0, volatility)),
            )
        except (TypeError, ValueError):
            continue
    
    for c in data.get("self_components", []) or []:
        try:
            name = c.get("name", "").strip()
            if not name:
                continue
            node.self_components.append(SelfComponent(
                name=name,
                category=c.get("category", "identity"),
                importance=int(c.get("importance", 5)),
                state=c.get("state", "intact"),
                attached_to=c.get("attached_to") or None,
            ))
        except (TypeError, ValueError):
            continue
    
    return node


def _build_relationship_batch(source_names: List[str], all_char_summaries: List[str],
                               story_idea, storyboard_brief: List[str]) -> List[Dict]:
    """Build edges for one batch of source characters (called by ai_initialize_relationships).

    Asks only for edges FROM the source_names characters to any other character.
    Keeping each call to 5 source characters caps the response to ~6-10 edges
    × ~600 chars each = well under Grok's output ceiling.

    Returns a raw list of edge dicts (not yet validated against the graph).
    """
    source_set = ', '.join(f'"{n}"' for n in source_names)
    prompt = f"""You are mapping the relationship web BEFORE chapter 1 for a story.
Focus ONLY on edges FROM these source characters: {source_set}
Include edges to ANY character they interact with. Make both A->B and B->A
when the relationship is significant — asymmetry is where drama lives.

STORY: {story_idea.genre} | {story_idea.premise}

STORYBOARD:
{chr(10).join(storyboard_brief)}

FULL CAST (for context):
{chr(10).join(all_char_summaries)}

Produce JSON: {{"edges": [...]}}

Each edge entry:
{{
  "source": "name", "target": "name",
  "edge_importance": "critical|supporting|background",
  "trust": -10 to 10, "affection": -10 to 10, "respect": -10 to 10,
  "attraction": -10 to 10, "fear": -10 to 10, "resentment": -10 to 10,
  "envy": -10 to 10, "empathy": -10 to 10,
  "perceived_power": -10 to 10,
  "power_type": "informational|physical|social|economic|emotional|moral",
  "debt": -10 to 10,
  "shared_history": ["1-2 facts"],
  "unspoken_truths": ["1-2 things never told"],
  "secrets_kept_from": ["1-2 hidden things"],
  "grievances": ["0-2 unresolved hurts"],
  "gifts_given": ["0-2 acts of love"],
  "trend": "strengthening|eroding|volatile|stable|new",
  "believes_trust": -10 to 10,
  "believes_affection": -10 to 10,
  "believes_respect": -10 to 10,
  "believes_attraction": -10 to 10,
  "believes_fear": -10 to 10,
  "notes": "one sentence"
}}

Only include edges where source is one of: {source_set}
Return ONLY the JSON."""

    data = _call_json(prompt, temperature=0.65, max_tokens=150000)
    if not data or not isinstance(data, dict):
        return []
    return data.get("edges", []) or []


def ai_initialize_relationships(graph: CharacterGraph, story_idea,
                                  storyboard: Dict) -> None:
    """Build the initial edge set between all characters, batched by source character.

    Decomposed into batches of 5 source characters per API call so no single
    response approaches Grok's output ceiling. Results from all batches are
    merged and deduplicated (last-write-wins on duplicate source→target pairs).
    """
    BATCH_SIZE = 5

    char_summaries = []
    for name, node in graph.nodes.items():
        char_summaries.append(
            f"- {name} ({node.role}) [{node.archetype}]: "
            f"wound={node.core_wound[:80]} | longing={node.core_longing[:80]}"
        )

    storyboard_brief = []
    if isinstance(storyboard, dict):
        for k, v in list(storyboard.items())[:12]:
            if isinstance(v, dict):
                events = _safe_events(v.get("events", []))
                storyboard_brief.append(f"Ch{k}: {events[0] if events else ''}")

    char_names = list(graph.nodes.keys())
    all_raw_edges: List[Dict] = []

    for batch_start in range(0, len(char_names), BATCH_SIZE):
        batch_sources = char_names[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(char_names) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"[EI]   Relationship batch {batch_num}/{total_batches}: "
                    f"{', '.join(batch_sources)}")
        try:
            edges = _build_relationship_batch(
                batch_sources, char_summaries, story_idea, storyboard_brief
            )
            all_raw_edges.extend(edges)
        except Exception as ex:
            logger.warning(f"[EI]   Relationship batch {batch_num} failed: {ex}")

    # Deduplicate: if the same source→target pair appears from multiple batches,
    # keep the last one (later batches have richer context from prior edges).
    seen: Dict[tuple, Dict] = {}
    for e in all_raw_edges:
        src = (e.get("source", "") or "").strip()
        tgt = (e.get("target", "") or "").strip()
        if src and tgt and src != tgt:
            seen[(src, tgt)] = e

    for (src, tgt), e in seen.items():
        try:
            if src not in graph.nodes or tgt not in graph.nodes:
                continue
            edge = RelationshipEdge(
                source=src, target=tgt,
                trust=float(e.get("trust", 0)),
                affection=float(e.get("affection", 0)),
                respect=float(e.get("respect", 0)),
                attraction=float(e.get("attraction", 0)),
                fear=float(e.get("fear", 0)),
                resentment=float(e.get("resentment", 0)),
                envy=float(e.get("envy", 0)),
                empathy=float(e.get("empathy", 0)),
                perceived_power=float(e.get("perceived_power", 0)),
                power_type=e.get("power_type", ""),
                debt=float(e.get("debt", 0)),
                shared_history=e.get("shared_history", []) or [],
                unspoken_truths=e.get("unspoken_truths", []) or [],
                secrets_kept_from=e.get("secrets_kept_from", []) or [],
                grievances=e.get("grievances", []) or [],
                gifts_given=e.get("gifts_given", []) or [],
                trend=e.get("trend", "stable"),
                tom_believes_trust=float(e.get("tom_believes_trust", 0)),
                tom_believes_affection=float(e.get("tom_believes_affection", 0)),
                tom_believes_respect=float(e.get("tom_believes_respect", 0)),
                tom_believes_attraction=float(e.get("tom_believes_attraction", 0)),
                tom_believes_fear=float(e.get("tom_believes_fear", 0)),
                tom_notes=e.get("tom_notes", ""),
            )
            graph.add_edge(edge)
        except (TypeError, ValueError):
            continue

    logger.info(f"[EI] Initialized {len(graph.edges)} edges across {len(graph.nodes)} characters")


def build_character_graph(characters: List, story_idea, storyboard: Dict,
                           parallel_nodes: bool = False) -> CharacterGraph:
    """MAIN ENTRY POINT for graph construction.

    Builds EI nodes for all characters, then initialises the relationship web.

    Node build strategy:
      • Tier 1 / Tier 2 characters (main + supporting): full
        ai_build_character_node — rich psychological profiling.
      • Tier 3 characters (minor, role contains 'minor' or index > first 8):
        _lightweight_minor_node — compact single-call profile.
        This keeps the graph comprehensive without quadrupling build time when
        a story has 10+ minor characters.

    A character is treated as minor when:
      (a) its `role` attribute contains the word 'minor', OR
      (b) it has no backstory, traits, or arc (effectively a skeleton entry), OR
      (c) it appears after position 8 in the cast list when the list > 10 long.
    """
    graph = CharacterGraph(story_idea=story_idea)

    logger.info(f"[EI] Building character graph for {len(characters)} characters...")

    def _is_minor(char, idx: int, total: int) -> bool:
        """Heuristic: should we use the lightweight builder for this char?"""
        role_low = (getattr(char, 'role', '') or '').lower()
        if 'minor' in role_low or 'one-off' in role_low or 'cameo' in role_low:
            return True
        # Characters with very thin profiles (no backstory, no traits) get lightweight
        has_detail = bool(
            getattr(char, 'backstory', '') or
            getattr(char, 'traits', None) or
            getattr(char, 'arc', '')
        )
        if not has_detail and idx >= 4:
            return True
        # Beyond 8 main/supporting in a large cast, use lightweight for the rest
        if total > 10 and idx >= 8:
            return True
        return False

    total = len(characters)
    full_count = 0
    minor_count = 0
    for i, ch in enumerate(characters):
        try:
            if _is_minor(ch, i, total):
                node = _lightweight_minor_node(ch, story_idea)
                minor_count += 1
            else:
                node = ai_build_character_node(ch, story_idea, characters)
                full_count += 1
            graph.add_node(node)
            if (i + 1) % 5 == 0:
                logger.info(f"[EI]   built {i+1}/{total} nodes")
        except Exception as ex:
            logger.warning(f"[EI] Node build failed for {ch.name}: {ex}")
            graph.add_node(_fallback_node(ch))

    logger.info(
        f"[EI] Nodes: {full_count} full + {minor_count} lightweight = "
        f"{full_count + minor_count} total"
    )
    logger.info(f"[EI] Initializing relationship web...")
    try:
        ai_initialize_relationships(graph, story_idea, storyboard)
    except Exception as ex:
        logger.warning(f"[EI] Relationship init failed: {ex}")

    # Initial snapshot
    graph.snapshots.append({
        'chapter': 0,
        'nodes': {n: {'arc_stage': node.arc_stage} for n, node in graph.nodes.items()},
        'key_edges': [{'source': e.source, 'target': e.target, 'label': e.label()}
                      for e in graph.significant_edges(threshold=4.0)[:10]],
        'tom_gap_count': len(graph.tom_gaps()),
    })

    return graph


# =============================================================================
# CAST SPREAD — deliberate extraordinariness ↔ relatability balance
# =============================================================================
# A rich cast needs CONTRAST: some larger-than-life figures who do the
# unbelievable, and some grounded, relatable, desirable people the reader bonds
# with. Left to chance, an LLM tends to make everyone equally epic (exhausting)
# or equally mundane (forgettable). This pass assigns each lead a register and
# writes a one-line note on how it should read, so downstream beat/dialogue
# passes can honour the spread. Deterministic fallback; never blocks.

_CAST_REGISTERS = {
    "mythic": "Larger-than-life. Reveal something they did or can do that the "
              "reader almost won't believe. Their presence should feel like myth "
              "walking. Give them at least one 'you-won't-believe-what-they-did' beat.",
    "aspirational": "Desirable and enviable — poised, gifted, or free in a way the "
                    "reader wishes they were. Let the reader covet their life a "
                    "little, then show the private cost underneath it.",
    "relatable": "Ordinary and deeply human — flawed, funny, uncertain. The reader "
                 "should see themselves here. Small real moments matter more than "
                 "spectacle; this is the emotional home base of the book.",
    "grounded": "The steady real-world anchor who keeps the story humane and "
                "believable — reacts the way a real person would, voices what the "
                "reader is thinking, earns trust through decency or competence.",
}


def assign_cast_spread(graph: 'CharacterGraph', characters: List,
                       story_idea=None, use_grok: bool = None) -> int:
    """Assign each lead a cast register so the ensemble has deliberate contrast.

    Tries an LLM pass that reads the whole cast and distributes registers for
    maximum dramatic contrast (ensuring at least one mythic and one relatable
    lead when the cast is large enough). Falls back to a deterministic
    round-robin so it never blocks. Annotates each CharacterNode with
    ``cast_register`` + ``register_note``. Returns the number assigned.
    """
    if graph is None or not getattr(graph, 'nodes', None):
        return 0
    leads = [c for c in (characters or [])][:12]
    if not leads:
        return 0

    names = [str(getattr(c, 'name', '')).strip() for c in leads if getattr(c, 'name', '')]
    assignments: Dict[str, str] = {}

    # ── LLM distribution (preferred) ────────────────────────────────────────
    try:
        roster = "\n".join(
            f"- {getattr(c, 'name', '')}: {getattr(c, 'role', '')} — "
            f"{(getattr(c, 'backstory', '') or getattr(c, 'description', '') or '')[:120]}"
            for c in leads
        )
        prompt = (
            "You are casting a graphic novel for maximum emotional range. Assign "
            "each character ONE register so the ensemble has deliberate contrast "
            "— not everyone epic, not everyone ordinary.\n\n"
            "REGISTERS:\n"
            "  mythic — larger-than-life, does the unbelievable\n"
            "  aspirational — desirable, enviable, who the reader wishes they were\n"
            "  relatable — ordinary, flawed, deeply human; the reader IS them\n"
            "  grounded — the steady real-world anchor who keeps it humane\n\n"
            f"GENRE: {getattr(story_idea, 'genre', '')}\n"
            f"CAST:\n{roster}\n\n"
            "Rules: the protagonist is usually 'relatable' or 'aspirational' (the "
            "reader rides with them); give at least one 'mythic' if any character "
            "warrants awe; ensure at least one 'relatable' or 'grounded' anchor. "
            "Spread the registers — avoid making everyone the same.\n\n"
            'Return ONLY JSON: {"assignments": [{"name": "...", "register": "mythic"}, ...]}'
        )
        # Reuse the module's JSON caller if present.
        data = _call_json(prompt, temperature=0.5, max_tokens=1500) \
            if '_call_json' in globals() else None
        if isinstance(data, dict):
            for a in (data.get('assignments') or []):
                nm = str(a.get('name', '')).strip()
                reg = str(a.get('register', '')).strip().lower()
                if nm and reg in _CAST_REGISTERS:
                    assignments[nm] = reg
    except Exception as e:
        logger.warning(f"[CastSpread] LLM distribution failed ({e}); using fallback.")

    # ── Deterministic fallback / gap-fill ───────────────────────────────────
    if len(assignments) < len(names):
        # Protagonist (index 0) → relatable; then cycle mythic/aspirational/
        # grounded/relatable so contrast is guaranteed.
        cycle = ["relatable", "mythic", "aspirational", "grounded"]
        for i, nm in enumerate(names):
            if nm in assignments:
                continue
            assignments[nm] = cycle[i % len(cycle)] if i > 0 else "relatable"

    # ── Apply to nodes ──────────────────────────────────────────────────────
    applied = 0
    for nm, reg in assignments.items():
        node = None
        key = nm.lower()
        for gk, gn in graph.nodes.items():
            if key == gk.lower() or key in gk.lower() or gk.lower() in key:
                node = gn
                break
        if node is not None:
            node.cast_register = reg
            node.register_note = _CAST_REGISTERS.get(reg, "")
            applied += 1

    # ── Wisdom disposition — who can carry an earned lesson ─────────────────
    # A subset of the cast are natural vehicles for karmic reflection / hard-won
    # wisdom (the Uncle-Iroh voice). Detect from archetype, role, and register so
    # the dialogue pass only puts lessons in mouths that can bear them — never
    # every character, never preachy.
    _MENTOR_MARKERS = (
        'mentor', 'sage', 'elder', 'teacher', 'master', 'monk', 'priest',
        'grandmother', 'grandfather', 'guardian', 'oracle', 'shaman', 'guide',
        'healer', 'wise', 'veteran', 'crone', 'hermit',
    )
    _TRICKSTER_MARKERS = ('trickster', 'jester', 'fool', 'rogue', 'clown', 'bard')
    wisdom_n = 0
    for nm, gn in graph.nodes.items():
        blob = " ".join(str(getattr(gn, a, '') or '') for a in
                        ('archetype', 'role', 'core_longing', 'growth_edge')).lower()
        # A character record may also carry a role; fold it in when present.
        for c in leads:
            if str(getattr(c, 'name', '')).lower() == nm.lower():
                blob += " " + " ".join(str(getattr(c, a, '') or '') for a in
                                       ('role', 'mythic_archetype')).lower()
                break
        if any(m in blob for m in _TRICKSTER_MARKERS) and any(
                m in blob for m in _MENTOR_MARKERS):
            gn.wisdom_disposition = "trickster-sage"
            wisdom_n += 1
        elif any(m in blob for m in _MENTOR_MARKERS):
            gn.wisdom_disposition = "mentor"
            wisdom_n += 1
        elif any(m in blob for m in _TRICKSTER_MARKERS):
            gn.wisdom_disposition = "trickster-sage"
            wisdom_n += 1
    if wisdom_n:
        logger.info("[CastSpread] %d character(s) can carry earned wisdom.", wisdom_n)

    # Log the spread so the balance is visible.
    from collections import Counter
    spread = Counter(assignments.values())
    logger.info("[CastSpread] Registers: %s",
                ", ".join(f"{k}×{v}" for k, v in spread.items()))
    return applied


# =============================================================================
# SCENE PROCESSING (per-chapter EI engine)
# =============================================================================

def ai_design_scene_engines(graph: CharacterGraph, chapter_num: int,
                             chapter_outline: Dict, storyboard_entry: Dict,
                             rolling_summary: str = "") -> List[SceneEngine]:
    """Generate dramatic structure for each scene in the chapter."""
    raw_scenes = chapter_outline.get("scenes", []) or []
    characters_present = chapter_outline.get("characters_present", []) or []
    setting = chapter_outline.get("setting", "")
    
    if not raw_scenes:
        raw_scenes = [f"Chapter {chapter_num} opens in {setting}"]
    
    # Minimal fallback path — the full version is much longer but the
    # comic generator never actually invokes this; it only needs the
    # function to exist for graph.process_chapter_scenes to chain.
    engines = []
    for i, s in enumerate(raw_scenes):
        engines.append(SceneEngine(
            scene_index=i, chapter_num=chapter_num, setting=setting,
            characters_present=characters_present,
            inciting_beat=s if isinstance(s, str) else str(s),
            turning_point=s if isinstance(s, str) else str(s),
        ))
    return engines


def process_chapter_scenes(graph, chapter_num, chapter_outline,
                            storyboard_entry, story_idea, rolling_summary=""):
    """Run a chapter through the graph engine.
    
    This is a simplified version. The novel generator calls a much richer
    version that does appraisal, observer-channel updates, contagion, and
    group-dynamics detection per scene. For the comic generator's needs
    (which mainly wants the *characters and graph*, not chapter-level
    processing), the simplified version is sufficient.
    """
    engines = ai_design_scene_engines(
        graph, chapter_num, chapter_outline, storyboard_entry, rolling_summary
    )
    result = ChapterProcessingResult(
        chapter_num=chapter_num,
        scenes=engines,
        group_mood=chapter_outline.get("emotional_tone", ""),
        chapter_emotional_arc=chapter_outline.get("emotional_tone", ""),
    )
    graph.chapter_results[chapter_num] = result
    return result


def generate_chapter_guidance(graph, chapter_result, include_voice_notes=True):
    """Build prose-level guidance string for the chapter writer."""
    parts = []
    parts.append("\n===== CHARACTER GRAPH GUIDANCE =====")
    parts.append(f"(Chapter {chapter_result.chapter_num})")
    
    if chapter_result.group_mood:
        parts.append(f"GROUP MOOD: {chapter_result.group_mood}")
    
    if include_voice_notes:
        present_chars = set()
        for scene in chapter_result.scenes:
            present_chars.update(scene.characters_present)
        if present_chars:
            parts.append("\n--- VOICE NOTES ---")
            for name in sorted(present_chars):
                node = graph.get_node(name)
                if not node:
                    continue
                voice_line = f"  {name}"
                if node.voice_signature:
                    voice_line += f": {node.voice_signature}"
                if node.rhythm:
                    voice_line += f" | rhythm: {node.rhythm}"
                parts.append(voice_line)
    
    parts.append("===== END GUIDANCE =====\n")
    return "\n".join(parts)


def snapshot_character_state(graph: CharacterGraph, chapter_num: int) -> Dict[str, Any]:
    """Record a snapshot of the graph state at this point."""
    snap = {
        "chapter": chapter_num,
        "nodes": {},
        "key_edges": [],
        "tom_gap_count": len(graph.tom_gaps()),
    }
    for name, node in graph.nodes.items():
        snap["nodes"][name] = {
            "arc_stage": node.arc_stage,
            "arc_pressure": node.arc_pressure,
            "dominant_trait": (node.dominant_trait().name if node.dominant_trait() else ""),
        }
    for edge in graph.significant_edges(threshold=5.0)[:15]:
        snap["key_edges"].append({
            "source": edge.source, "target": edge.target,
            "label": edge.label(), "trend": edge.trend,
        })
    graph.snapshots.append(snap)
    return snap


def report_graph_evolution(graph: CharacterGraph) -> str:
    """End-of-story growth report."""
    if len(graph.snapshots) < 2:
        return "[EI] Not enough snapshots for evolution report."
    
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("CHARACTER GRAPH EVOLUTION REPORT")
    lines.append("=" * 60)
    
    first = graph.snapshots[0]
    last = graph.snapshots[-1]
    lines.append(f"\nSnapshots: {len(graph.snapshots)}")
    lines.append(f"ToM gaps: {first['tom_gap_count']} -> {last['tom_gap_count']}")
    
    return "\n".join(lines)


def export_graph_json(graph: CharacterGraph, path: str) -> None:
    """Dump graph state to JSON."""
    data = {
        "nodes": {},
        "edges": [],
        "snapshots": graph.snapshots,
        "global_tensions": graph.global_tensions,
    }
    for name, node in graph.nodes.items():
        data["nodes"][name] = {
            "name": node.name, "role": node.role, "archetype": node.archetype,
            "core_wound": node.core_wound, "core_longing": node.core_longing,
            "defense_mechanism": node.defense_mechanism,
            "shadow": node.shadow, "growth_edge": node.growth_edge,
            "traits": {k: asdict(v) for k, v in node.traits.items()},
            "self_components": [asdict(c) for c in node.self_components],
            "voice_signature": node.voice_signature,
            "lexical_habits": node.lexical_habits,
            "rhythm": node.rhythm,
            "metaphor_pool": node.metaphor_pool,
            "arc_stage": node.arc_stage,
            "arc_pressure": node.arc_pressure,
            "recent_emotions": [asdict(m) for m in node.recent_emotions[-6:]],
        }
    for (s, t), edge in graph.edges.items():
        data["edges"].append(asdict(edge))
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# IMAGE GENERATION PIPELINE
# =============================================================================

def load_ImageZ_pipe():
    """Load the image generation pipeline based on IMAGE_MODEL setting."""
    if not _HAS_DIFFUSERS or not _HAS_TORCH:
        raise RuntimeError(
            "Image generation requires `torch` and `diffusers` installed. "
            "Install with: pip install torch diffusers"
        )
    
    if IMAGE_MODEL == 'ERNIE':
        if ErnieImagePipeline is None:
            raise RuntimeError("ErnieImagePipeline not available")
        pipe = ErnieImagePipeline.from_pretrained(
            "Baidu/ERNIE-Image-Turbo",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    elif IMAGE_MODEL == 'ZIMAGE':
        if ZImagePipeline is None:
            raise RuntimeError("ZImagePipeline not available")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    elif IMAGE_MODEL == 'CHROMA':
        if ChromaPipeline is None:
            raise RuntimeError(
                "ChromaPipeline not available — update diffusers "
                "(pip install -U diffusers) to a version that includes Chroma."
            )
        pipe = ChromaPipeline.from_pretrained(
            CHROMA_MODEL_ID, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        # VAE tiling/slicing reduce peak VRAM on the larger Chroma VAE when
        # available; guard each call so older diffusers builds don't error.
        if hasattr(pipe, 'vae'):
            if hasattr(pipe.vae, 'enable_tiling'):
                pipe.vae.enable_tiling()
            if hasattr(pipe.vae, 'enable_slicing'):
                pipe.vae.enable_slicing()
    elif IMAGE_MODEL == 'LENS':
        if LensPipeline is None:
            raise RuntimeError(
                "LensPipeline not available — install Microsoft's Lens package "
                "from github.com/microsoft/Lens (and run from a context where "
                "`from lens import LensPipeline` resolves, which registers the "
                "custom LensGptOssEncoder / LensTransformer2DModel components)."
            )
        pipe = LensPipeline.from_pretrained(
            LENS_MODEL_ID, torch_dtype=torch.bfloat16
        )
        # CPU offload trades speed for VRAM (the 3.8B model + GPT-OSS text
        # encoder + Flux2 VAE are large). Set LENS_CPU_OFFLOAD=False to pin the
        # whole pipeline on the GPU instead (faster, needs much more VRAM).
        if LENS_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    elif IMAGE_MODEL == 'KLEIN2':
        if _Flux2DiffusionPipeline is None:
            raise RuntimeError(
                "DiffusionPipeline not available for KLEIN2 — update diffusers "
                "to a version with FLUX.2 support (pip install -U diffusers)."
            )
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers AutoModel/AutoTokenizer not available — needed to "
                "load the KLEIN2 uncensored text encoder."
            )
        # Performance flags from the reference (TF32 + high matmul precision).
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Load the uncensored drop-in text encoder + its tokenizer, then swap
        # them into the FLUX.2 Klein pipeline (replacing the stock encoder).
        tokenizer = AutoTokenizer.from_pretrained(
            KLEIN2_ENCODER_ID, trust_remote_code=True
        )
        text_encoder = AutoModel.from_pretrained(
            KLEIN2_ENCODER_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        pipe = _Flux2DiffusionPipeline.from_pretrained(
            KLEIN2_MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        pipe.tokenizer = tokenizer
        pipe.text_encoder = text_encoder
        if KLEIN2_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        raise ValueError(f"Unknown IMAGE_MODEL: {IMAGE_MODEL}")
    
    return pipe


def gen_ImageZ_image(pipe, prompt, height=1024, width=1024, seed=-1,
                      num_inference_steps=9, promptEnhance=False,
                      extra_negative: str = ""):
    """Generate one image from a text prompt using the loaded pipeline.

    Parameters
    ----------
    extra_negative : str
        Additional negative-prompt terms appended after the base list.
        Use this to pass caller-specific exclusions (e.g. comic-format
        artifacts from the comic book pipeline) without hardcoding them here.
    """
    # Base anatomical / quality negatives — always present.
    # Text / lettering / speech-bubble section: diffusion models strongly
    # associate "comic book" and "graphic novel" style prompts with rendered
    # text elements (speech bubbles, caption boxes, onomatopoeia).  Listing
    # them here as hard negatives is the most reliable suppression layer
    # because the negative prompt is weighted equally against every denoising
    # step, whereas positive-prompt "no text" clauses compete with content
    # tokens and can be overridden as context length grows.
    negative_prompt = (
        # Anatomy / quality
        "ugly, low quality, blurry, bad anatomy, bad hands, bad fingers, "
        "multiple views, jpeg artifacts, extra limbs, extra fingers, "
        "bad face, long neck, short neck, worst quality, bad anatomy, "
        "long chin, deformed hand, too many arms, mutated hands, bad perspective, "
        # Text / lettering — the primary cause of speech-bubble hallucination
        "speech bubble, speech balloon, word balloon, thought bubble, "
        "dialog box, dialogue box, caption box, caption text, text box, "
        "text overlay, words, letters, lettering, typography, subtitles, "
        "captions, annotations, watermark, logo, signature, "
        "onomatopoeia, sound effects text, sfx text, "
        # Comic-format structure cues that trigger text rendering
        "comic panel border, panel gutter, panel frame, panel grid, "
        "comic book page layout, manga panel, narration box, "
        "handwriting, graffiti, foreground signage, readable text, "
        # Quality / anatomy / artifact suppression (all models)
        "multiple views, jpeg artifacts, patreon logo, patreon username, "
        "web address, signature, watermark, text, logo, artist name, censored, "
        "uncanny valley, weibo watermark, small face, "
    )
    if extra_negative:
        negative_prompt = negative_prompt + ", " + extra_negative.strip(", ")

    if seed == -1:
        seed = torch.randint(0, MAX_SEED, (1,)).item()
    # CHROMA's reference usage seeds a CPU generator (it runs with model CPU
    # offload). ZIMAGE/ERNIE/LENS use a CUDA generator as in their references.
    # CHROMA and KLEIN2 run with model CPU offload; seed a CPU generator to
    # avoid a device mismatch. ZIMAGE/ERNIE/LENS use a CUDA generator.
    _gen_device = "cpu" if IMAGE_MODEL in ('CHROMA', 'KLEIN2') else "cuda"
    generator = torch.Generator(_gen_device).manual_seed(seed)

    # -----------------------------------------------------------------------
    # IMPORTANT — negative-prompt behaviour on these distilled models.
    #
    # ZImage runs at guidance_scale=0.0 and ERNIE at 1.0. Both are
    # guidance-DISTILLED few-step models (9 / 8 steps); at these scales there
    # is no classifier-free-guidance contrast, so a negative_prompt has
    # essentially NO EFFECT even when supplied. For that reason the negative
    # prompt was historically not passed to the pipe at all.
    #
    # We now pass it conditionally, controlled by NEGATIVE_PROMPT_GUIDANCE.
    # By default it stays 0 → negative prompt is NOT passed (preserves the
    # exact current behaviour and avoids handing a kwarg to a pipeline that
    # may reject it). If you switch to a CFG-capable model, set
    # NEGATIVE_PROMPT_GUIDANCE to a real scale (e.g. 4.0) and the negatives
    # become active automatically — no other code changes needed.
    #
    # Bottom line for callers: at the current settings the POSITIVE prompt is
    # the only control surface that works. All text/bubble suppression must be
    # achieved by NOT naming those elements in the positive prompt (handled in
    # comic_book_generator.compose_panel_prompt), not by the negatives here.
    # -----------------------------------------------------------------------
    use_negative = float(globals().get('NEGATIVE_PROMPT_GUIDANCE', 0.0)) > 1.0
    neg_guidance = float(globals().get('NEGATIVE_PROMPT_GUIDANCE', 0.0))

    # Defensive size guard: every supported backend (Z-Image, ERNIE, Chroma,
    # Lens, FLUX.2 Klein) requires height & width divisible by 16. Round here so
    # a caller that passes an off-grid size (e.g. 888 = 8x111) can never reach a
    # pipeline and trigger "Height must be divisible by 16". LENS/KLEIN2 also
    # re-round inside their branches with their own clamps; that stays a no-op
    # on an already-rounded value.
    def _round16(v: int) -> int:
        v = int(v)
        if v <= 0:
            return 1024
        r = int(round(v / 16)) * 16
        return max(16, r)
    height = _round16(height)
    width = _round16(width)

    with torch.no_grad():
        if IMAGE_MODEL == 'ZIMAGE':
            if use_negative:
                with torch.no_grad():
                    image = pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        height=height, width=width,
                        num_inference_steps=9, guidance_scale=neg_guidance,
                        generator=generator
                    ).images[0]
            else:
                with torch.no_grad():
                    image = pipe(
                        prompt=prompt, height=height, width=width,
                        num_inference_steps=9, guidance_scale=0.0, generator=generator
                    ).images[0]
        elif IMAGE_MODEL == 'ERNIE':
            if use_negative:
                with torch.no_grad():
                    image = pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        height=height, width=width,
                        num_inference_steps=8, guidance_scale=neg_guidance,
                        generator=generator, use_pe=promptEnhance
                    ).images[0]
            else:
                with torch.no_grad():
                    image = pipe(
                        prompt=prompt, height=height, width=width,
                        num_inference_steps=8, guidance_scale=1.0, generator=generator,
                        use_pe=promptEnhance
                    ).images[0]
        elif IMAGE_MODEL == 'CHROMA':
            # CHROMA is a full classifier-free-guidance model (NOT distilled),
            # so the negative prompt genuinely improves output and is ALWAYS
            # passed. It uses its own step count + guidance scale. The caller's
            # requested num_inference_steps is honoured when explicitly larger
            # than the CHROMA default; otherwise CHROMA_N_STEPS is used so a
            # caller passing the distilled-model default of 8-9 still gets a
            # proper Chroma render.
            chroma_steps = max(
                int(num_inference_steps or 0),
                int(globals().get('CHROMA_N_STEPS', 40)),
            )
            chroma_guidance = float(globals().get('CHROMA_GUIDANCE', 3.0))
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height, width=width,
                    num_inference_steps=chroma_steps,
                    guidance_scale=chroma_guidance,
                    generator=generator,
                    num_images_per_prompt=1,
                ).images[0]
        elif IMAGE_MODEL == 'LENS':
            # LENS (microsoft/Lens) is a full CFG model — the negative prompt
            # genuinely improves output and is ALWAYS passed. It uses its own
            # step count + guidance scale. We drive it with explicit height/width
            # (the pipeline accepts these directly instead of base_resolution/
            # aspect_ratio), rounded to the model's VAE factor (16) and clamped to
            # its native max dimension so the panel aspect ratio is preserved.
            lens_factor  = int(globals().get('LENS_VAE_FACTOR', 16))
            lens_max     = int(globals().get('LENS_MAX_DIM', 1440))

            def _round_lens(v: int) -> int:
                v = int(v)
                v = min(v, lens_max)                 # clamp to native max
                v = max(lens_factor, round(v / lens_factor) * lens_factor)
                return v

            lens_h = _round_lens(height)
            lens_w = _round_lens(width)
            # LENS_N_STEPS is the model's proper step count (20 default; 4 for
            # Lens-Turbo). Unlike CHROMA we do NOT take max() with the caller's
            # value, because a few-step variant legitimately wants a LOW count —
            # honour the configured LENS_N_STEPS directly.
            lens_steps = int(globals().get('LENS_N_STEPS', 20))
            lens_guidance = float(globals().get('LENS_GUIDANCE', 3.0))
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    base_resolution=1440,
                    height=lens_h, width=lens_w,
                    num_inference_steps=lens_steps,
                    guidance_scale=lens_guidance,
                    generator=generator,
                    num_images_per_prompt=1,
                ).images[0]
        elif IMAGE_MODEL == 'KLEIN2':
            # KLEIN2 (FLUX.2 Klein 9B) uses its own step count + guidance scale
            # and accepts explicit height/width, rounded to its VAE factor (16)
            # and clamped to its native max dimension (1536) so the panel aspect
            # ratio is preserved.
            #
            # NOTE: Flux2KleinPipeline.__call__() does NOT accept a
            # `negative_prompt` (nor `num_images_per_prompt`) argument — passing
            # either raises a TypeError. The uncensored text encoder and the
            # positive-prompt content are the control surface here; text/bubble
            # suppression is handled by NOT naming those elements in the positive
            # prompt (see comic_book_generator.compose_panel_prompt).
            klein_factor = int(globals().get('KLEIN2_VAE_FACTOR', 16))
            klein_max    = int(globals().get('KLEIN2_MAX_DIM', 1536))

            def _round_klein(v: int) -> int:
                v = int(v)
                v = min(v, klein_max)                 # clamp to native max
                v = max(klein_factor, round(v / klein_factor) * klein_factor)
                return v

            klein_h = _round_klein(height)
            klein_w = _round_klein(width)
            klein_steps = int(globals().get('KLEIN2_N_STEPS', 28))
            klein_guidance = float(globals().get('KLEIN2_GUIDANCE', 3.75))
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,
                    height=klein_h, width=klein_w,
                    num_inference_steps=klein_steps,
                    guidance_scale=klein_guidance,
                    generator=generator,
                ).images[0]

    # Per-image VRAM release. At this point `image` is a CPU-side PIL object,
    # so the pipeline's transient GPU tensors (latents, attention/activation
    # buffers) are no longer referenced now that we've left the no_grad block.
    # Release them before returning so reserved VRAM does not creep up
    # image-by-image and OOM partway through a long run. Panels, covers, and
    # act-break cards all route through this function, so this one call covers
    # every image the comic pipeline generates.
    release_image_vram()
    return image


# =============================================================================
# IMAGE PROCESSING — pipeline of enhancements applied after generation
# =============================================================================

def apply_auto_contrast(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c]
        lo, hi = np.percentile(channel, (1, 99))
        if hi - lo < 10:
            result[:, :, c] = channel
        else:
            result[:, :, c] = np.clip(
                (channel.astype(float) - lo) * 255.0 / (hi - lo), 0, 255
            )
    return result.astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                 tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2RGB)


def apply_vibrance(image: np.ndarray, intensity: float = 0.25) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
    saturation = hsv[:, :, 1]
    boost = intensity * (1.0 - saturation / 255.0)
    hsv[:, :, 1] = np.clip(saturation * (1.0 + boost), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_cinematic_color_grade(image: np.ndarray, warmth: float = 0.05) -> np.ndarray:
    result = image.astype(float)
    brightness = np.mean(result, axis=2, keepdims=True) / 255.0
    result[:, :, 0] = np.clip(result[:, :, 0] + warmth * 255 * brightness[:, :, 0], 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] + warmth * 128 * (1 - brightness[:, :, 0]), 0, 255)
    return result.astype(np.uint8)


def apply_vignette(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * 0.6)
    kernel_y = cv2.getGaussianKernel(rows, rows * 0.6)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignetted = (image.astype(float) * mask[:, :, np.newaxis]).astype(np.uint8)
    return cv2.addWeighted(image, 1.0 - strength, vignetted, strength, 0)


def apply_subtle_grain(image: np.ndarray, intensity: float = 5.0) -> np.ndarray:
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)


def denoise_and_sharpen(image: np.ndarray, h: float = 6, h_color: float = 6,
                         template_window_size: int = 7, search_window_size: int = 21,
                         sharpen_strength: float = 0.3) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None.")
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_denoised = cv2.fastNlMeansDenoising(l, None, h, template_window_size, search_window_size)
    a_denoised = cv2.fastNlMeansDenoising(a, None, h_color, template_window_size, search_window_size)
    b_denoised = cv2.fastNlMeansDenoising(b, None, h_color, template_window_size, search_window_size)
    denoised_bgr = cv2.cvtColor(cv2.merge([l_denoised, a_denoised, b_denoised]), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened_bgr = cv2.filter2D(denoised_bgr, -1, kernel)
    result_bgr = cv2.addWeighted(denoised_bgr, 1.0 - sharpen_strength,
                                  sharpened_bgr, sharpen_strength, 0)
    result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def full_image_pipeline(image_np: np.ndarray, is_cover: bool = False) -> np.ndarray:
    """Apply the full image enhancement chain."""
    result = denoise_and_sharpen(image_np)
    result = apply_auto_contrast(result)
    result = apply_clahe(result, clip_limit=2.5 if is_cover else 2.0)
    result = apply_vibrance(result, intensity=0.30 if is_cover else 0.20)
    result = apply_cinematic_color_grade(result, warmth=0.06 if is_cover else 0.04)
    result = apply_vignette(result, strength=0.25 if is_cover else 0.15)
    result = apply_subtle_grain(result, intensity=3.0)
    return result


def compress_and_save_image(image, save_path: str,
                              jpeg_quality: int = IMAGE_JPEG_QUALITY,
                              max_dimension: int = IMAGE_MAX_DIMENSION) -> str:
    """Compress and save an image with proper JPEG handling."""
    if image.mode in ('RGBA', 'P', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        if 'A' in image.mode:
            background.paste(image, mask=image.split()[-1])
        else:
            background.paste(image)
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    w, h = image.size
    if max(w, h) > max_dimension:
        ratio = max_dimension / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    
    image.save(save_path, format='JPEG', quality=jpeg_quality, optimize=True)
    return save_path


# =============================================================================
# MODULE EXPORTS
# =============================================================================
# When the comic generator imports from this module, these are the names it
# expects to find. This explicit __all__ documents the public API.
# =============================================================================

__all__ = [
    # Story / character
    'StoryIdea', 'Character', 'CharacterAppearanceRegistry',
    'generate_story_idea', 'generate_characters', 'identify_required_characters',
    'synthesize_seeds_to_story',
    # Chapter dataclass (for novel pipeline use)
    'Chapter',
    # Character graph (EI engine)
    'CharacterGraph', 'CharacterNode', 'RelationshipEdge', 'DynamicTrait',
    'SelfComponent', 'EmotionalMoment', 'SceneEngine',
    'ChapterProcessingResult',
    'build_character_graph', 'process_chapter_scenes',
    'generate_chapter_guidance', 'snapshot_character_state',
    'export_graph_json', 'report_graph_evolution',
    'ai_build_character_node', 'ai_initialize_relationships',
    # LLM helpers
    'get_openai_prompt_response', 'get_openai_prompt_response_reasoning',
    'parse_json_response', 'sanitize_text_for_prompt', '_safe_events',
    # Image generation
    'load_ImageZ_pipe', 'gen_ImageZ_image', 'full_image_pipeline',
    'compress_and_save_image',
    # Utility
    'create_directory_if_not_exists', 'cleanup_special_chars', 'reset_memory',
    'release_image_vram',
    'save_objects', 'load_objects',
    # Configuration
    'USE_GROK', 'openai_model', 'openai_model_large', 'openai_model_small_reasoning',
    'grok_fast_reasoning_model', 'grok_fast_nonreasoning_model',
    'MAX_SEED', 'NEGATIVE_PROMPT_GUIDANCE', 'device', 'dtype', 'retry_limit',
    'IMAGE_JPEG_QUALITY', 'IMAGE_MAX_DIMENSION',
    'COVER_WIDTH', 'COVER_HEIGHT', 'COVER_N_STEPS', 'COVER_GUIDANCE',
    'width', 'height', 'guidance_scale', 'n_steps',
    'lora_triggers', 'IMAGE_MODEL', 'IMAGE_STYLE',
    'CHROMA_MODEL_ID', 'CHROMA_N_STEPS', 'CHROMA_GUIDANCE', 'CHROMA_COVER_N_STEPS',
    'LENS_MODEL_ID', 'LENS_N_STEPS', 'LENS_GUIDANCE', 'LENS_COVER_N_STEPS',
    'LENS_VAE_FACTOR', 'LENS_MAX_DIM', 'LENS_CPU_OFFLOAD',
    'KLEIN2_MODEL_ID', 'KLEIN2_ENCODER_ID', 'KLEIN2_N_STEPS', 'KLEIN2_GUIDANCE',
    'KLEIN2_COVER_N_STEPS', 'KLEIN2_VAE_FACTOR', 'KLEIN2_MAX_DIM', 'KLEIN2_CPU_OFFLOAD',
    'get_model_gen_geometry',
]


# =============================================================================
# MAIN ENTRY POINT (when run as a script)
# =============================================================================
# This module is primarily designed to be imported. When you run the full
# novel generation pipeline you would use the original monolithic script.
# Running this file directly just prints a status message.
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
    print()
    print("This is the importable module version of the novel generator.")
    print("To use it, import from another script:")
    print()
    print("    from novel_generator import (")
    print("        StoryIdea, Character, CharacterAppearanceRegistry,")
    print("        generate_story_idea, generate_characters,")
    print("        build_character_graph, load_ImageZ_pipe, gen_ImageZ_image,")
    print("        ...")
    print("    )")
    print()
    print("Optional dependencies status:")
    print(f"  torch:        {'OK' if _HAS_TORCH else 'MISSING'}")
    print(f"  diffusers:    {'OK' if _HAS_DIFFUSERS else 'MISSING'}")
    print(f"  lens:         {'OK' if _HAS_LENS else 'MISSING (needed only for IMAGE_MODEL=LENS)'}")
    print(f"  klein2:       {'OK' if _HAS_KLEIN2 else 'MISSING (needed only for IMAGE_MODEL=KLEIN2)'}")
    print(f"  aura_sr:      {'OK' if _HAS_AURA else 'MISSING'}")
    print(f"  transformers: {'OK' if _HAS_TRANSFORMERS else 'MISSING'}")
    print()
    print("To run the full novel generation pipeline, use the original")
    print("monolithic script. The comic_book_generator.py module uses")
    print("this file for its building blocks.")

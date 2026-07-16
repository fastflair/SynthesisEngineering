# NOTE: This variant was patched for notebook-safe local animation on a 24GB RTX 4090.
# It avoids Wan-S2V completely by default and uses I2V + external lip-sync.

'''
=============================================================================
STORY → ANIMATION  ·  voiced animated film generator
=============================================================================
Turns a *story idea* (or a single character + a hand-written script) into a
fully voiced, animated film.

It reuses the graphic-novel brain you already have:
  • novel_generator        — StoryIdea / Character, KLEIN2 image pipeline,
                             LLM helpers (get_openai_prompt_response).
  • comic_book_generator   — synthesize_comic_story() + generate_comic_script()
                             to turn a story idea into a cast + a panel-by-panel
                             script with structured dialogue.

…and adds three new layers on top:

  1. RICH DIALOGUE          An emotional-intelligence pass rewrites whole
     (EI engine)            multi-character exchanges using the CharacterGraph
                            (wounds, longings, voice signatures + relationship
                            edges: trust/fear/resentment, unspoken truths,
                            theory-of-mind). This is what gives the dialogue real
                            subtext instead of characters taking turns talking.
                            Falls back to a per-line spoken rewrite when no graph
                            is available (and can synthesize one for script mode).

  2. PER-CHARACTER TTS      Pluggable voice-cloning engines. Default = Higgs-Audio-v3.
     (voice cloning)        Also: Chatterbox, F5-TTS, E2-TTS.
                            Each character gets a distinct cloned voice (hybrid
                            sourcing: your reference clip if supplied, otherwise
                            auto-cast from a local voice bank; NARRATOR gets its
                            own voice). With no bank and no supplied clip,
                            un-cloned characters share the selected engine's
                            default voice (no separate bootstrap/minting model —
                            XTTS, Tortoise, and VibeVoice were all dropped: XTTS's
                            Coqui `TTS` package pins transformers<5 while every
                            other engine here needs >=5; Tortoise depends on
                            other old, unmaintained libraries; and the standalone
                            VibeVoice package conflicts with transformers>=5's
                            own internals). F5/E2/Higgs reference clips are
                            auto-transcribed with Whisper. bubble_type/emphasis
                            drive the emotional delivery.

  3. ANIMATION              Pluggable video engines, auto-routed per shot:
                              • dialogue shots → Wan2.2-S2V  (audio-driven lip-sync)
                              • everything else → Wan2.2 TI2V-5B (default, I2V)
                              • long takes      → FramePack   (HunyuanVideo packed I2V)
                            LONG TAKES on EVERY engine are built by chaining
                            short segments — each segment's last frame seeds
                            the next. Engines that don't do long video natively
                            are capped at max_segments_per_chain (3) per chain;
                            beyond that a fresh starting image is generated to
                            reset drift. FramePack handles long takes natively
                            (exempt). The lip-sync engine slices the audio per
                            segment to stay aligned. After animation, a LIP-SYNC
                            pass (LatentSync / MuseTalk / Wav2Lip) re-syncs each
                            dialogue shot's mouth to its TTS speech — the non-
                            audio-driven engines don't track words on their own.
                            Routing is overridable (single engine, per-shot tag,
                            compare-all). Tuned for a single RTX 4090 (24 GB
                            VRAM) + 64 GB RAM via sequential load/unload, CPU
                            offload, VAE tiling, quantization where supported.

No background music is added — mux your own afterward if you want it.

-----------------------------------------------------------------------------
TWO-PHASE WORKFLOW  (plan → edit → produce)
-----------------------------------------------------------------------------
  PHASE 1 — plan()         story idea → storyboard: cast, scenes, dialogue
                           (EI-rewritten), structured image prompts, and a
                           voice assigned to every character. Writes an EDITABLE
                           plan.json (+ plan.md). Loads no image/video models.

      anim = StoryAnimator.from_story_idea("a lighthouse keeper who ...", project)
      plan_path = anim.plan()        # → animation_out/<title>/plan.json

  …open plan.json, tweak anything (a line, a character's ref_wav to clone your
  own voice, an image_prompt, a per-shot engine, a duration)…

  PHASE 2 — produce()      reads the (edited) plan and renders audio + images +
                           animation, then the final film. Idempotent: re-run to
                           resume (finished artifacts are reused).

      produce_from_plan(plan_path)   # or: python story_to_animation.py produce plan.json

INPUT MODES
  A) STORY IDEA  →  StoryAnimator.from_story_idea("a lighthouse keeper who ...")
  B) SCRIPT      →  StoryAnimator.from_script(character=Character(...), script=[...])
  Both produce a plan you can edit before producing.

VOICES
  Default voice_mode="random": each character gets a DISTINCT random voice
  from your voice bank if present. With no bank, every un-cloned character
  falls back to sharing the SELECTED engine's single default voice — there's
  no separate bootstrap/minting model (XTTS, Tortoise, and the standalone
  VibeVoice package were all dropped as dependency dead ends: XTTS's Coqui
  `TTS` pins transformers<5 while every engine here needs >=5; Tortoise
  depends on other old, unmaintained libraries; VibeVoice's package conflicts
  with transformers>=5's own internals). Override any character by setting
  its ref_wav (in the plan, or via character_refs={name: clip}) → that voice
  is cloned instead. Add clips to voice_bank_dir for real per-character
  variety without supplying full refs.

IMAGE PROMPTS
  Phase 1 has the LLM author each shot's prompt in the structured labelled
  format (Subject / Clothing / Action / Environment / Objects / Lighting /
  Camera / Style Details), matching the comic generator's format.

-----------------------------------------------------------------------------
INSTALL (one engine's deps are only needed if you select that engine)
-----------------------------------------------------------------------------
  core      : torch diffusers transformers accelerate safetensors
              imageio imageio-ffmpeg moviepy librosa soundfile numpy pillow json-repair
              pydub openai-whisper            # ref-clip prep + auto-transcription
              Keep diffusers current (`pip install -U diffusers`, or from git for the
              newest model integrations) — Wan2.2/Z-Image support has shifted across
              releases; an outdated diffusers is the source of most "unexpected config
              attribute"-style warnings on model load (harmless; the model still loads).
  chatterbox: pip install chatterbox-tts
  f5 / e2   : pip install f5-tts              # E2-TTS uses the same repo (E2TTS_Base)
  higgs     : none beyond transformers>=5 — loads via plain AutoModelForCausalLM
              from multimodalart/higgs-audio-v3-tts-4b-transformers, trust_remote_code=True
              (avoids the original bosonai release's custom boson-multimodal package).
              (default engine)
  framepack : the diffusers_helper package from your MVSW notebook must be importable
  lipsync   : clone one of LatentSync / MuseTalk / Wav2Lip and point
              VideoConfig.lipsync_repo_dir + lipsync_checkpoint at it (the stage
              shells out to the repo's inference CLI). Disable with lipsync_engine=None.
  wan_i2v   : recent diffusers with WanImageToVideoPipeline (TI2V-5B)
  wan_s2v   : the LOCAL Wan-Video/Wan2.2 repo (no diffusers S2V pipeline exists).
              Auto-cloned/downloaded into VideoConfig.cache_dir on first use if
              VideoConfig.auto_download_models=True (needs `git` + huggingface_hub);
              or set wan_repo_dir / wan_s2v_ckpt_dir yourself. Fully local inference
              either way — the one-time fetch is the only network use.

SINGLE-IMAGE TALKING VIDEO
  produce_talking_image(image_path, text=... or audio_path=..., ...) — a
  lightweight alternative to the full story pipeline: one reference image +
  either text (synthesized to speech) or a ready audio file → one continuous
  Wan2.2-S2V take as long as the audio. The image is resized to match its OWN
  resolution, capped to a 720p-equivalent budget for a 4090
  (VideoConfig.cap_resolution_for_4090, on by default for every engine here).
'''

from __future__ import annotations
STORY_TO_ANIMATION_PATCH_LEVEL = "notebook-styleanchor-mouthvis-imgmatch-fatalgpu-earlybail-charseed-lipsyncmotion-expressive-voice-image-higgs-placement-affirmativemouth-mouthvisdetect-stylizedrelax-narratorgate-cbgactionseq-fpsyncsave-fpdtype-fpvaeslice-nospeakmotion-fpwritevideo-nospeakplan-nospeakllm-ltx2engine-ltx2setup-ltx2abspath-ltx2fp8prequant-2026-07-09"

import os
os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# This pipeline sequentially loads/unloads several multi-GB models on the
# same 24GB card (KLEIN2/Z-Image, an LLM-adjacent TTS engine, then whichever
# video engines a story needs) — repeated alloc/free of very different
# tensor shapes is exactly the pattern that fragments CUDA's allocator over
# a long run, eventually OOMing even when nvidia-smi shows free memory.
# expandable_segments lets the allocator grow a segment instead of hunting
# for a new contiguous block, which is the standard mitigation for this
# (torch>=2.1; harmlessly ignored by older builds that don't recognize it).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import logging
import math
import re
import shutil
import subprocess
import shlex
import sys
import time
import warnings
import contextlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("story_to_animation")

# ── Soft deps (only imported when actually used) ─────────────────────────────
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

if _HAS_TORCH:
    try:
        # TF32 matmul on Ampere/Ada (the 4090 is Ada) trades a sliver of
        # precision most diffusion/transformer workloads don't notice for a
        # real throughput win on every matmul-heavy model in this pipeline
        # (Z-Image, Higgs, every video engine) — cheap, broadly safe, set
        # once globally rather than per-engine. Isolated in its own
        # try/except so a failure here (e.g. an exotic CPU-only build)
        # can't take down torch availability for the whole module.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    sf = None
    _HAS_SF = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False

# ── Reuse the graphic-novel brain ────────────────────────────────────────────
# Imported defensively so this module can at least be inspected / partially used
# (e.g. the TTS layer) on a machine that doesn't have the heavy comic deps.
try:
    import novel_generator as ng
    from novel_generator import StoryIdea, Character
    _HAS_NG = True
except Exception as _e:  # pragma: no cover
    ng = None
    StoryIdea = Character = object  # type: ignore
    _HAS_NG = False
    logger.warning("novel_generator not importable (%s) — story/image stages disabled.", _e)

try:
    import comic_book_generator as cbg
    _HAS_CBG = True
except Exception as _e:  # pragma: no cover
    cbg = None
    _HAS_CBG = False
    logger.warning("comic_book_generator not importable (%s) — story-idea mode disabled.", _e)
	
# =============================================================================
# CONFIG
# =============================================================================

# Model IDs (override per project if you keep local copies / different quants).
KLEIN2_IMAGE_MODEL = "KLEIN2"                       # routed through novel_generator
WAN22_S2V_MODEL_ID  = "Wan-AI/Wan2.2-S2V-14B"       # audio-driven, lip-sync
WAN22_I2V_MODEL_ID  = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"   # plain I2V (non-dialogue Wan)
WAN22_WAV2VEC_ID    = "facebook/wav2vec2-large-xlsr-53"   # S2V audio encoder

# FramePack (matches your MVSW notebook exactly).
FP_HUNYUAN_ID    = "hunyuanvideo-community/HunyuanVideo"
FP_REDUX_ID      = "lllyasviel/flux_redux_bfl"
FP_TRANSFORMER   = "lllyasviel/FramePackI2V_HY"

# TTS model IDs.
TTS_MODEL_IDS = {
    "chatterbox": "resemble-ai/chatterbox",
    "f5":         "SWivid/F5-TTS",
    # Community port to plain transformers (AutoModelForCausalLM), avoiding the
    # original bosonai release's custom `boson_multimodal` serving package.
    "higgs":      "multimodalart/higgs-audio-v3-tts-4b-transformers",
}

@dataclass
class TTSConfig:
    engine: str = "higgs"                       # chatterbox|f5|e2|higgs
    sample_rate: int = 24000
    device: str = "cuda"
    voice_bank_dir: str = "./voice_bank"        # *.wav refs + optional index.json
    # Folder-matched reference voices. For voice_mode="folder_match", each
    # subfolder name is treated as a voice description, and the first audio file
    # in that subfolder becomes the character's baseline/reference voice.
    character_voice_dir: str = "/mnt/d/data/audio/characters"
    character_voice_audio_exts: Tuple[str, ...] = (
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"
    )
    # If novel_generator is available, ask the LLM to choose the closest unused
    # folder voice for each character. If not, or if the answer is invalid,
    # fallback scoring is used. One pick removes that folder from the pool.
    use_ai_folder_voice_matching: bool = True
    narrator_ref: Optional[str] = None          # wav for the NARRATOR voice (auto if None)
    character_refs: Dict[str, str] = field(default_factory=dict)  # name -> ref wav (your clips)
    gap_ms: int = 320                           # silence between lines in a shot
    lead_silence_ms: int = 120                  # base lead-in for a plain continuation
    tail_silence_ms: int = 350                  # base tail-out for a plain continuation
    # Extra lead/tail silence ADDED on top of the base above, depending on
    # what's happening at this shot's edit boundary. This matters most for
    # Wan-S2V: the audio's length directly drives the take's length there, so
    # padding it is literally how you give a cut/transition room to breathe,
    # or a beat of pause before the next person starts talking. Less padding
    # is added where none of that is happening (a shot continuing the same
    # speaker in the same scene gets just the base silence above).
    scene_transition_pad_ms: int = 700           # this shot borders a scene/setting change
    speaker_change_pad_ms: int = 350             # same scene, but the speaker switches
    # When a dialogue shot opens a NEW CONVERSATION after a scene change —
    # i.e. the scene changes AND the previous shot was not already in dialogue
    # with the same characters — viewers need extra time to orient to the new
    # location before the first line lands.  This pad is used instead of
    # scene_transition_pad_ms at those boundaries (it must be >= it).
    #
    # A reasonable derivation:
    #   dialogue_scene_entry_pad_ms
    #       ≈ (scene_transition_seconds × 1000)   ← let the dissolve finish
    #       + orientation_beat_ms                  ← a beat to read the new space
    #
    # With the default scene_transition_seconds=0.5 s and a 900 ms orientation
    # beat this comes to ~1400 ms (1.4 s), which matches standard film practice
    # for a cold cut into a new location before the first spoken line.  Increase
    # orientation_beat_ms if your transitions are slower or your locations are
    # visually complex; decrease it for fast-cut action sequences.
    dialogue_scene_entry_pad_ms: int = 1400      # scene changes AND new conversation starts
    orientation_beat_ms: int = 900               # read-the-room beat after dissolve completes
    # Audio-DRIVEN engines (LTX-2 a2vid, Wan-S2V) generate the mouth FROM the
    # waveform, so leading silence at the head of a dialogue shot's driver audio
    # can surface as an A/V onset offset if the model anchors speech near frame 0.
    # When True, dialogue shots keep only a minimal head lead (audio_driven_lead_ms)
    # and let the cut's breathing room ride on the PREVIOUS shot's tail (which
    # already carries the same boundary pad) — unless that previous shot has no
    # audio/tail of its own, in which case the pad is kept on the lead so the cut
    # doesn't turn abrupt. Non-dialogue/narration shots are unaffected. Set False
    # to restore the old symmetric lead+tail padding on every shot.
    trim_lead_for_audio_driven: bool = True
    audio_driven_lead_ms: int = 60               # head lead kept on driven dialogue shots
    max_chars_per_chunk: int = 320              # long lines get chunked then concatenated
    cross_fade_ms: int = 150                    # cross-fade when stitching chunks of one line
    trim_line_silence: bool = False             # trim leading/trailing silence per line
    # ── Sound polish ─────────────────────────────────────────────────────────
    loudness_normalize: bool = True             # level every shot to a consistent loudness
    target_rms: float = 0.12                     # target RMS (~consistent perceived level)
    peak_ceiling: float = 0.97                   # hard limiter so normalization never clips
    edge_fade_ms: int = 12                        # tiny fade in/out per shot → no clicks at cuts
    # Inline control tokens (<|emotion:...|>, <|style:...|>, <|sfx:...|>,
    # <|prosody:...|>) that Higgs Audio v3 reads directly out of the spoken
    # text — see add_voice_cue_tokens(). Only applied for engines that
    # actually understand this vocabulary (currently just "higgs"); on any
    # other engine these would just be read aloud as literal text, so this
    # flag is harmless to leave on regardless of engine.
    voice_cue_tokens: bool = True
    # Deterministic safety layer: every Higgs line gets at least one valid
    # <|emotion:...|> token before synthesis, even if the optional LLM cue pass
    # skipped it or the plan was edited by hand.
    require_emotion_cue_tokens: bool = True
    # Also add obvious style/sfx/prosody cues where the line/emotion clearly
    # calls for them, using only documented Higgs tokens.
    add_style_sfx_prosody_tokens: bool = True
    # Lean the delivery MORE expressive/dramatic: the LLM cue pass is allowed to
    # layer emotion + prosody (and write in a real sigh/laugh/gasp where the
    # moment earns it) instead of tagging sparingly, and the deterministic layer
    # adds dramatic pauses at ellipses/em-dashes and emphasis on shouted words.
    # Still uses ONLY documented tokens and never rewrites the spoken words.
    expressive_delivery: bool = True
    model_id_override: Optional[str] = None
    # Voice assignment
    voice_mode: str = "random"                  # random | match | clone | folder_match
    #   random : a distinct random voice per character (override any via character_refs).
    #            With no bank match, a per-character BASELINE clip is generated
    #            and used as that character's cloning reference from then on —
    #            see voice_baseline_template below — so every line stays
    #            consistent even though no human reference clip was supplied.
    #            (There's no separate bootstrap/minting model: XTTS and
    #            Tortoise were removed as dependency dead ends. Add real
    #            voice_bank clips or character_refs whenever you want an
    #            actual human voice instead of the engine's own.)
    #   match  : auto-cast from the bank by gender/descriptor
    #   clone  : require a supplied ref per character (character_refs)
    # Per-character baseline (closes the consistency gap for un-cloned voices):
    # generated once per character via the SAME engine that will voice the
    # film, reading this line (with {name} substituted), then used as that
    # character's fixed cloning reference for every subsequent line. Override
    # per-character text/regen via VoiceProfile.baseline_text /
    # .regenerate_baseline (editable in the plan). Skipped entirely for
    # characters that already have a real ref (character_refs or a bank
    # match) — those are already a fixed file, already consistent.
    voice_baseline_template: str = (
        "I am {name} and this is my sound baseline. The quick brown fox jumps "
        "over the lazy dog. Pack my box with five dozen liquor jugs while the "
        "quick brown fox jumps.")
    # Reference-clip handling (for cloning engines that want a transcript: F5/E2/Higgs)
    auto_transcribe_refs: bool = True           # Whisper-transcribe a ref clip if no text given
    whisper_model: str = "openai/whisper-large-v3-turbo"
    prepare_refs: bool = True                   # silence-split + clip refs to <=15s
    ref_texts: Dict[str, str] = field(default_factory=dict)  # name -> known transcript (skips Whisper)


@dataclass
class VideoConfig:
    # routing: "auto" | "single" | "per_shot" | "compare_all"
    routing: str = "auto"
    single_engine: str = "wan_i2v"              # used when routing == "single"
    width: int = 960
    height: int = 544                           # 960x544 ≈ 16:9, FramePack-friendly bucket
    fps: int = 24
    # per-engine default seconds when a shot has no dialogue to time against
    default_seconds: float = 4.0
    max_seconds: float = 12.0                   # soft cap — ONLY for shots with no audio at all;
    #   a shot carrying any audio (dialogue or narration) is never capped below
    #   its own audio length, on any engine — see min_audio_padding below.
    min_audio_padding: float = 0.35             # extra seconds held after audio ends —
    #   room for a cut/transition so dialogue never feels clipped at the edit
    seed: int = -1
    # Long takes: render in short segments, feeding each segment's last frame in
    # as the next segment's start image, then stitch. Works for every engine.
    chain_segments: bool = True
    segment_seconds: float = 5.0                # target length of each segment
    max_segments_per_chain: int = 3             # non-native engines: cap a chain at N
    #   …then start a fresh chain from a newly generated still (resets drift).
    # 4090 memory strategy
    high_vram_threshold_gb: float = 40.0        # below this → aggressive offload
    try_sage_attention: bool = True             # Wan benefits a lot from SageAttention
    use_fp8: bool = True                        # Wan I2V: try fp8 dtype first, fall back to bf16 if unsupported
    # ── Auto-download & resolution policy ────────────────────────────────────
    auto_download_models: bool = True           # fetch missing model files automatically
    cache_dir: str = "./model_cache"            # where auto-downloaded repos/checkpoints land
    cap_resolution_for_4090: bool = True        # clamp any new image/video to 720p-equivalent
    # (long side <= 1280, short side <= 720) — see _cap_resolution_for_4090.
    # dimensions are divisible by 64, so use the same 64-safe policy for stills
    # before animation starts. Example: project 960x544 → image stills 960x576.
    image_resolution_multiple: int = 64
    image_resolution_rounding: str = "ceil"    # ceil | floor | nearest
    image_max_long_side: int = 960             # default cap for still generation (3090)
    # ── LTX-2 generation resolution ──────────────────────────────────────────
    # apply_gpu_preset() sets these (and vcfg.width/height) together so that
    # every part of the pipeline — image generation, LTX2._resolution(), compose,
    # and log messages — all agree on the same dimensions:
    #
    #   RTX 3090  →  960 × 576  (16:9, both exact 64-multiples)
    #   RTX 4090  → 1280 × 720  (16:9 / 720p, 720 is a 16-multiple)
    #   RTX 5090  → 1280 × 720  (16:9 / 720p, same target as 4090)
    #
    # The defaults below match the 3090 preset and are overwritten when you call
    # apply_gpu_preset() for a different card.  Override ltx2_max_long/short
    # here (or pass gen_width/gen_height to apply_gpu_preset) to use a custom
    # resolution; remember that LTX-2 requires dimensions to be multiples of 16.
    ltx2_max_long: int = 960                   # generation long-side ceiling for LTX-2
    ltx2_max_short: int = 576                  # generation short-side ceiling for LTX-2
    ltx2_use_upsampler: bool = False           # upsampler disabled by default; presets
    #   set this explicitly. True → output is 2× the generation dims.
    # Wan2.2-S2V's local repo (auto-cloned/downloaded here if missing)
    wan_repo_url: str = "https://github.com/Wan-Video/Wan2.2"
    wan_s2v_hf_repo: str = "Wan-AI/Wan2.2-S2V-14B"
    # Wan2.2 — S2V (14B, audio-driven) and I2V (5B, plain image-to-video) are
    # different models with different cost profiles, so they get independent
    # steps/guidance rather than one shared setting:
    #   • I2V (5B): cheap enough that the extra quality from more steps is
    #     worth it — Wan-AI's own model card recommends steps=50, guidance=5.0
    #     for real quality, and that's what stays the default here.
    #   • S2V (14B): a much heavier model where steps cost more per-step, and
    #     audio conditioning already constrains the output a lot — defaults
    #     to a faster 8 steps / guidance 3.0. Raise these (e.g. 20–30 / 4–5)
    #     for a higher-quality pass once you're happy with a take's content.
    wan_i2v_steps: int = 50
    wan_i2v_guidance_scale: float = 5.0
    wan_s2v_steps: int = 8
    wan_s2v_guidance_scale: float = 3.0
    wan_frames_per_chunk: int = 77              # S2V batches frames in ~77s windows
    # Wan2.2-S2V runs via the OFFICIAL local repo (github.com/Wan-Video/Wan2.2),
    # not diffusers — diffusers has no S2V pipeline. Fully local: a cloned repo
    # + downloaded checkpoint on disk, GPU inference. No cloud/remote calls.
    wan_repo_dir: str = ""                      # path to your clone of Wan-Video/Wan2.2
    #   (skip this if you `pip install -e` the repo so `import wan` already works)
    wan_s2v_ckpt_dir: str = ""                  # path to the downloaded s2v-14B checkpoint dir
    wan_s2v_size: str = "1280*720"              # must be a key in that repo's wan.configs.SIZE_CONFIGS
    wan_s2v_infer_frames: int = 80               # frames per internal clip (multiple of 4): 48 or 80
    wan_s2v_num_clip: int = 0                    # 0 = let the model pace itself off the whole audio file
    wan_s2v_shift: Optional[float] = None        # None = the repo's own per-task default (sample_shift)
    # Wan2.2-S2V-14B is a LARGE model — 14B params is ~28GB just for weights
    # at bf16, already over a 4090's 24GB. Worse: the official repo's
    # WanS2V.__init__ loads the DiT directly onto CUDA at construction time
    # (device_map=self.device, unconditionally), which can OOM before
    # generate() even runs — its own t5_cpu/offload_model machinery is meant
    # to stream the model between the 64GB of system RAM and the GPU per
    # clip, but only works if construction doesn't front-load it onto CUDA
    # first. wan_s2v_cpu_init patches around that (loads on CPU instead, then
    # generate()'s existing offload cycle takes over normally); int8
    # runs on the CPU-resident model, before anything touches the GPU.
    wan_s2v_cpu_init: bool = True
    # bnb4 (recommended for 24GB cards): bitsandbytes NF4 4-bit, injected at
    # LOAD TIME via a from_pretrained patch — loads the 14B DiT already
    # quantized (~4GB instead of ~28GB at bf16), the technically correct way
    # to use bnb (vs. quantizing after a full-precision load) and the most
    # comfortable fit for a 4090. Falls back automatically (with a clear log)
    # if bitsandbytes isn't installed or this repo's from_pretrained doesn't
    # accept quantization_config.
    # int8/fp8: torchao weight-only, applied AFTER a full CPU-staged load
    # (see _quantize_dit) — works regardless of from_pretrained's kwargs, but
    # still stages the full-size model through CPU RAM first.
    wan_s2v_quantize: str = "bnb4"                # bnb4 | int8 | fp8 | none
    wan_s2v_quantize_attr: str = ""              # override: exact attribute name of the
    #   DiT submodule on your installed wan.WanS2V, if the auto-guess misses it
    # If wan_s2v fails to LOAD even after a retry (e.g. a transient CUDA
    # driver hiccup, common right after a long sustained heavy workload on a
    # different engine — "device not ready" is usually the driver still
    # settling, not a real config problem), its dialogue shots are rerouted
    # to this engine instead of being dropped from the film entirely. They'll
    # still get audio muxed in either way; lipsync_shots() will then pick
    # them up for a real lip-sync pass IF a lipsync engine is configured
    # (lipsync_repo_dir/lipsync_checkpoint) — without one, the fallback shots
    # play with correct audio but without lip-sync, which is still strictly
    # better than vanishing from the film. Set to "" to disable the fallback
    # and restore the old skip-and-warn behavior.
    wan_s2v_fallback_engine: str = "wan_i2v"
    # Hard kill-switch: skip wan_s2v entirely, never even attempt to load it.
    # Routes every dialogue+audio shot straight to wan_s2v_fallback_engine
    # (default wan_i2v) instead, and the standalone render_dialogue_take()
    # helper does the same rather than force-using wan_s2v. Useful if S2V is
    # reliably crashing the machine outright (e.g. a system-level GPU/driver
    # crash, not just a Python exception the retry/fallback logic can catch)
    # — in that case the in-process load-failure fallback never even gets a
    # chance to run, so this flag skips the attempt up front instead.
    # Dialogue shots still get real lip-sync via lipsync_engine (e.g.
    # latentsync) as long as lipsync_repo_dir/lipsync_checkpoint are set —
    # _needs_lipsync() only skips shots actually rendered by wan_s2v, so
    # routing them elsewhere makes them lipsync_shots() targets automatically.
    wan_s2v_disable: bool = True
    # Notebook-safe mode treats S2V as unavailable even if an old plan or a
    # per-shot override requests it. This is intentionally stronger than the
    # load-failure fallback: when S2V crashes the whole computer/kernel, Python
    # never gets a chance to catch the exception. The safe local alternative is
    # I2V first, then an external lip-sync pass in a separate subprocess.
    notebook_safe_mode: bool = True
    notebook_safe_dialogue_engine: str = "wan_i2v"
    notebook_safe_disable_framepack: bool = False
    # General video-engine fallback. This prevents optional/heavy engines such
    # fail to load or hit a CUDA driver/internal allocator issue. The fallback
    # keeps the shot in the film, with its normal dialogue/narration audio.
    video_engine_fallback_engine: str = "wan_i2v"
    video_engine_fallback_on_load_failure: bool = True
    video_engine_fallback_on_render_failure: bool = True
    # ── LTX-2 (Lightricks) — joint audio-video DiT ─────────────────────────────
    # One engine, two internal paths:
    #   • shots WITH audio  → A2VidPipelineTwoStage: still image anchors the
    #     visual identity while TTS audio drives motion AND native lip-sync.
    #   • shots WITHOUT audio → DistilledPipeline: fastest image-to-video.
    # Runs the runner script (sta_ltx2_runner.py written into the repo) as a
    # subprocess using the configured Python interpreter. No venv is required —
    # if ltx-core and ltx-pipelines are pip-installed into the same conda/system
    # env that runs this notebook, sys.executable is used automatically.
    # Install once:  pip install -e ~/repos/LTX-2/packages/ltx-core
    #                pip install -e ~/repos/LTX-2/packages/ltx-pipelines
    prefer_ltx2: bool = False
    ltx2_disable: bool = False                  # hard off-switch (plans with engine:"ltx2" reroute)
    ltx2_repo_dir: str = "~/repos/LTX-2"        # cloned repo root (runner script is written here)
    ltx2_python: str = ""                       # blank → repo .venv/bin/python if present,
                                                 #         otherwise sys.executable (current env)
    ltx2_models_dir: str = ""                   # extra dir searched for models (besides repo/models/)
    # Explicit model paths (blank → auto-discovered by find_ltx2_assets):
    ltx2_checkpoint: str = ""                   # any of:
                                                 #   ltx-2.3-22b-distilled-fp8.safetensors  (29.5GB, pre-quantized fp8)
                                                 #   ltx-2.3-22b-distilled-1.1.safetensors  (46GB, bf16)
                                                 #   ltx-2.3-22b-dev.safetensors            (46GB, bf16, slower)
                                                 # fp8 file notes:
                                                 #   • fp8-cast works with both the fp8 and bf16 files on RTX 3090
                                                 #   • fp8-scaled-mm requires Ada Lovelace (RTX 40xx+) — NOT for 3090
    ltx2_spatial_upsampler: str = ""            # ltx-2.3-spatial-upscaler-x2-1.1.safetensors (1GB)
    ltx2_distilled_lora: str = ""               # ltx-2.3-22b-distilled-lora-384-1.1.safetensors
                                                 # only needed with the DEV (non-distilled) checkpoint
    ltx2_distilled_lora_strength: float = 0.8
    ltx2_gemma_root: str = ""                   # dir of google/gemma-3-12b-it-qat-q4_0-unquantized
    # Quantization — RTX 3090 (Ampere) recommendation:
    #   CHECKPOINT CHOICE MATTERS ON RTX 3090 (Ampere, no native FP8 tensor cores):
    #     • bf16 checkpoint (ltx-2.3-22b-distilled-1.1.safetensors, 46GB) + fp8-cast
    #       → CORRECT for 3090. fp8-cast compresses weights to FP8 for storage but
    #         the matmul runs in a way Ampere supports (weight-only). This is the
    #         path the LTX docs recommend for non-Ada GPUs.
    #     • pre-quantized fp8 checkpoint (ltx-2.3-22b-distilled-fp8.safetensors, 29.5GB)
    #       → NEEDS native FP8 matmul (fp8e4m3 tensor cores) at runtime, which only
    #         Ada Lovelace (RTX 40xx) / Hopper / Blackwell have. On a 3090 it raises
    #         'RuntimeError: fp8 matmul not supported' no matter the loading policy.
    #         load() detects this and refuses early unless ltx2_allow_fp8_on_ampere=True.
    #   fp8-scaled-mm: native FP8 scaled matmul — RTX 40xx (Ada)+ only.
    ltx2_quantization: str = "fp8-cast"         # fp8-cast | fp8-scaled-mm (40xx+ only) | none
    ltx2_allow_fp8_on_ampere: bool = False      # override the early refusal (will likely crash on 3090)
    ltx2_auto_download_bf16: bool = False        # on Ampere w/ only an fp8 ckpt, auto-fetch the bf16
                                                 # distilled checkpoint (46GB) via huggingface_hub
    ltx2_offload_mode: str = "cpu"              # none | cpu | disk  ('disk' streams weights
                                                 # from disk — use if a 46GB mmap fails with ENOMEM)
    # CLI module names — the engine invokes these as `python -m <module>`. Their
    # own argparse builds the correct QuantizationPolicy for --quantization, so
    # we never call build_policy() (which mmaps the whole checkpoint). Override
    # only if a future ltx-pipelines renames the modules.
    ltx2_distilled_module: str = "ltx_pipelines.distilled"      # no-audio image→video
    ltx2_a2v_module: str = "ltx_pipelines.a2vid_two_stage"      # image+audio→video (lip-sync)
    ltx2_num_inference_steps: int = 24          # ignored for distilled ckpt (uses fixed 8-sigma schedule)
    ltx2_a2v_guidance_scale: float = 3.0        # modality CFG for AV sync (dev ckpt only; >1 = tighter sync)
    ltx2_frame_rate: float = 0.0                # 0 → use project fps. LTX-2 was trained around 25fps.
    ltx2_segment_seconds: float = 6.0           # one generation ≈ 145 frames @24fps; the harness
                                                # chains segments (with per-segment audio slices) for
                                                # longer takes, so dialogue of any length works
    ltx2_resolution_multiple: int = 64          # two-stage pipelines assert W/H % 64 == 0
    ltx2_timeout_sec: int = 5400                # per-segment subprocess watchdog
    ltx2_extra_env: str = ""                    # extra "K=V K=V" env pairs for the subprocess
    # FramePack
    # FramePack's Hunyuan sampling schedule is internally calibrated around
    # 30 fps. Keep that as the native generation rate, then normalize to the
    # project fps/duration after render. This avoids the old 30→24 fps duration
    # stretch when project fps is 24.
    fp_latent_window: int = 9
    fp_internal_fps: int = 30
    fp_section_rounding: str = "ceil"            # ceil | round | floor
    fp_force_exact_duration: bool = True         # trim/pad native output to requested shot seconds
    framepack_min_seconds: float = 6.0           # auto-route shots this long or longer to FramePack
    dialogue_long_takes_use_framepack: bool = True  # long dialogue: FramePack motion, then external lip-sync
    fp_save_section_progress: bool = False       # optional: save intermediate section previews
    fp_section_progress_dir: str = ""            # blank = <clip_dir>/framepack_sections
    fp_steps: int = 32                           # user's FramePack notebook default
    fp_cfg: float = 1.0
    fp_gs: float = 10.0
    fp_rs: float = 0.0
    fp_gpu_preserve_gb: int = 6
    fp_transformer_offload_preserve_gb: int = 8
    fp_use_teacache: bool = False                # user's notebook uses False; set True for speed tests
    fp_mp4_crf: int = 16                         # set lower, e.g. 8 or 1, for very large HQ intermediates
    # ── Motion & finishing quality ───────────────────────────────────────────
    cinematic_motion: bool = True               # add camera-move + motion cues to video prompts
    # Current video/animation models handle two patterns badly: stepping
    # motion up/down stairs (foot placement + perspective shift over many
    # frames) and coordinated multi-person physical contact (merged limbs,
    # interpenetration, desynced motion between the two figures). When on,
    # motion cues/prompts steer toward a simpler equivalent for shots that
    # hit either pattern instead of asking for the complex motion outright
    # — see _has_stair_motion / _is_multi_person_contact_risk.
    avoid_complex_motion: bool = True
    # Dialogue shots get lip-synced downstream (Wan-S2V natively, or I2V + an
    # external LatentSync pass). Large camera moves and body motion make the
    # face harder to track and visibly degrade lip-sync. When on, talking shots
    # are steered toward a near-locked frame with subtle head motion and natural
    # micro-expressions so the most-watched shots read clean. On by default.
    lipsync_friendly_motion: bool = True
    # Keep every speaking character's mouth visible and unobstructed in the
    # motion prompt (drops "hand over mouth", eating/drinking/smoking, etc.).
    # A covered mouth both breaks lip-sync face detection and reads as "not
    # talking". On by default.
    keep_speaker_mouth_visible: bool = True
    # After a dialogue still is rendered, verify the speaker actually has a
    # detectable, unobstructed mouth; if not (hand over mouth, profile, no face),
    # regenerate with a fresh seed up to N times and keep the best result. This
    # is the deterministic backstop behind the prompt-level steering: prompts
    # reduce occlusion, this catches the residual cases before they reach the
    # animator and LatentSync. Requires mediapipe (preferred) or opencv; if
    # neither is importable it degrades to a no-op (logged once), so it is safe
    # to leave on. Only ever inspects dialogue shots with a visible speaker.
    verify_speaker_mouth_visible: bool = True
    mouth_visibility_max_retries: int = 2       # extra seeds to try on a bad still
    # Relax face-detection thresholds for stylized / cartoon / non-photorealistic
    # art styles.  mediapipe FaceMesh is trained on photos; it locates anime,
    # painted, or non-human character faces less reliably, so a stricter threshold
    # would cause spurious regeneration loops. Lower confidence widens the net;
    # the smaller span floor lets slim cartoon mouths pass; the smaller min-face
    # fraction lets a Haar cascade find stylized heads at smaller relative size.
    # "auto" (default) uses relaxed values when the project theme contains typical
    # stylized-art keywords; set True/False to override explicitly.
    mouth_visibility_stylized_art: object = "auto"   # True | False | "auto"
    mouth_visibility_min_confidence: float = 0.25    # mediapipe detection confidence for stylized art
    mouth_visibility_min_span_frac: float = 0.008    # min mouth width as fraction of image width
    mouth_visibility_min_face_frac: float = 0.06     # min face height as fraction of image height (Haar)
    # Skip the check when a face is genuinely not expected (already handled by
    # _shot_has_visible_people), so silhouettes/cutaways are never regenerated.
    # Only animate elements that are actually depicted in the still image /
    # visual scene. Strips motion clauses that introduce a person, animal,
    # vehicle, or object not present in the image (a common motion-model
    # hallucination). On by default.
    motion_prompts_only_visible_elements: bool = True
    motion_prompt_max_chars: int = 900          # keep animation prompts short (~200 tokens) so
    video_negative: str = (                     # steer engines away from common artifacts
        # NOTE: this negative prompt only takes effect on engines that run real
        # classifier-free guidance. The DEFAULT engine (FramePack) runs with
        # real CFG=1.0, which zeroes the negative embeddings (see the sampler
        # call in the FramePack backend), so these terms do NOTHING on the
        # default path. Anti-occlusion for dialogue lives in the POSITIVE motion
        # prompt (_strip_mouth_occlusion_in_motion) and in the still-level
        # mouth-visibility check — do not "fix" occlusion by editing this string.
        # It is still honoured by wan_i2v and other true-CFG fallbacks.
        "blurry, low detail, distorted, deformed, warping, morphing, melting, "
        "flickering, strobing, extra limbs, extra fingers, duplicated face, "
        "identity drift, jitter, stutter, watermark, text, caption, subtitles, speech bubble, "
        "dialogue bubble, dialogue text, written words, lettering, title text, frame border, "
        "face morphing, shifting facial features, inconsistent face, changing identity, "
        "merged bodies, interpenetrating limbs, tangled limbs, bodies fused "
        "together, impossible multi-person contact, unnatural synchronized "
        "movement between people, glitching through stairs, feet sliding on "
        "stairs, floating up stairs, awkward stair climbing, "
        "hand over mouth, hand covering mouth, hand covering face, hand obscuring face, "
        "fingers over lips, object covering mouth, mouth obscured, mouth hidden, face obscured by hand")
    end_fade: bool = False                       # gentle fade-out (video+audio) on the final film
    fade_seconds: float = 0.6                    # …of this length (hook-friendly: no fade-IN)
    # ── Scene transitions (final assembly) ─────────────────────────────────────
    # How to join one SCENE to the next (a "scene" = a run of consecutive shots
    # sharing the same setting). Cuts WITHIN a scene stay hard. The transition
    # rides in the silent tail/head padding already added at scene boundaries
    # (scene_transition_pad_ms on the TTS side), so it never dissolves over
    # dialogue. Styles:
    #   "none"        — hard cut (old behaviour)
    #   "dissolve"    — cross-dissolve between the two scenes (default; gentle
    #                   "time/place has moved" cue that reads well when the
    #                   pipeline can't know the meaning of each scene change)
    #   "fade_black"  — dip through black (stronger act/chapter break)
    #   "fade_white"  — dip through white (flashback / memory / harsh cut feel)
    scene_transition: str = "dissolve"
    scene_transition_seconds: float = 0.5        # nominal transition length
    # Never let a transition eat more than this fraction of the SHORTER of the
    # two neighbouring scenes (keeps short scenes from being swallowed and keeps
    # xfade offsets valid). The effective length per boundary is the smaller of
    # scene_transition_seconds and this ratio × the shorter neighbour.
    scene_transition_max_ratio: float = 0.5
    scene_transition_min_seconds: float = 0.12   # below this, just hard-cut
    # ── Lip-sync post-processing ─────────────────────────────────────────────
    # After animation, re-sync dialogue shots' mouths to the generated speech.
    # match lips to audio on their own; this stage fixes that. Wan-S2V already
    # lip-syncs, so it's skipped by default.
    lipsync_engine: Optional[str] = "latentsync"   # latentsync | musetalk | wav2lip | None(off)
    lipsync_repo_dir: str = ""                  # path to the cloned engine repo (required to run)
    lipsync_checkpoint: str = ""                # checkpoint file/dir the repo's inference needs
    lipsync_inference_script: str = ""          # override the engine's default script path
    lipsync_command_template: str = ""          # full override; see BaseLipSync for placeholders
    lipsync_python_exe: str = ""                # default: the running interpreter
    lipsync_extra_args: str = ""                # appended to the command (e.g. extra flags)
    lipsync_unet_config: str = "auto"  # LatentSync-specific: required by its
    #   inference.py; relative to lipsync_repo_dir. Use "configs/unet/stage2_efficient.yaml" if
    #   you want LatentSync's lower-VRAM variant (slight quality tradeoff, not needed at ~6.5GB).
    lipsync_only_dialogue: bool = True          # only sync shots that actually have non-narrator speech
    lipsync_skip_narration: bool = True         # narration/caption-only audio never needs face lip-sync
    lipsync_skip_if_no_face: bool = True        # no detectable face is a valid cinematic case; keep muxed clip
    lipsync_require_visible_speaker: bool = True  # don't sync offscreen/different speakers to a visible face
    lipsync_skip_audio_driven: bool = True      # don't re-sync Wan-S2V (already lip-synced)
    # ── Audio/video synchronization guardrails ───────────────────────────────
    # Every shot clip is normalized to an exact duration so audio/video drift
    # cannot accumulate when many clips are concatenated. This is especially
    # important when dialogue falls back from Wan-S2V to I2V + external lip-sync.
    exact_audio_video_duration: bool = True
    sync_tolerance_sec: float = 0.08            # warn when |video-audio| exceeds this
    write_sync_report: bool = True              # writes clips/sync_report_*.json + .csv
    # For highest lip-sync quality, keep S2V shots to a single speaking face.
    # Multi-speaker audio should normally be split into single-speaker shots;
    # when it exists anyway, route it to the fallback engine and post lip-sync.
    prefer_single_speaker_s2v: bool = True
    max_s2v_dialogue_seconds: float = 18.0      # long monologues drift; fallback for post-sync
    require_lipsync_for_dialogue_fallback: bool = False  # warn instead of fail by default


@dataclass
class ProjectConfig:
    title: str = "Untitled"
    out_root: str = "./animation_out"
    image_negative: str = (                     # steer the image model away from common artifacts
        # NOTE: the DEFAULT image model (Z-Image-Turbo) is distilled CFG-free and
        # runs at zimage_guidance=0.0, so diffusers disables classifier-free
        # guidance and this negative prompt is IGNORED entirely on the default
        # path. Mouth/hand occlusion is handled instead by affirmative POSITIVE
        # prompting (_lipsync_face_safety_instruction / _HANDS_AWAY_IMG_CUE) plus
        # the post-generation mouth-visibility check. These terms are still used
        # by the KLEIN2 backend, which runs true CFG — keep them for that path.
        "blurry, low detail, distorted, deformed, extra limbs, extra fingers, "
        "missing fingers, fused fingers, malformed hands, asymmetric features, "
        "watermark, signature, text, caption, subtitles, speech bubble, dialog bubble, "
        "dialogue text, written words, lettering, title text, labels, frame border, jpeg artifacts, "
        "hand over mouth, hand covering mouth, hand covering face, hand obscuring face, "
        "fingers over lips, object covering mouth, mouth obscured, mouth hidden, face obscured by hand, "
        "oversaturated, washed out, low contrast")
    # Image generator: Z-Image-Turbo (default) or KLEIN2 (via novel_generator).
    image_model: str = "zimage"                 # zimage | klein2
    zimage_model_id: str = "Tongyi-MAI/Z-Image-Turbo"
    zimage_steps: int = 9                       # Turbo: ~8 DiT forwards
    zimage_guidance: float = 0.0                # Turbo is distilled CFG-free → 0.0
    target_pages: int = 12                      # story-idea mode: rough length knob
    panels_per_page_avg: int = 3
    cast_size: str = "small"
    enrich_dialogue: bool = True                # the "rich dialogue" spoken pass
    use_ei_dialogue: bool = True                # use the EI graph for subtext-rich rewrites
    build_ei_graph_for_script: bool = False     # in script-mode, synthesize an EI graph too
    voice_narration: bool = True                # speak NARRATOR captions
    add_background_music: bool = False          # kept False by request
    # ── Cinematic direction (Phase 1, quality-oriented planning) ─────────────
    opening_hook: bool = True                   # craft a gripping cold-open as shot 0
    hook_seconds: float = 3.0                   # length of that opening hook
    shape_pacing: bool = True                   # set intentional durations by emotional energy
    shot_variety: bool = True                    # avoid 3+ identical compositions in a row
    # Reuse ONE seed for every still that features the same primary character,
    # so a character's face/build/hair stay recognisably the same across shots
    # instead of drifting each generation. The differing prompt per shot still
    # gives each shot its own pose/framing/setting. Ignored when a fixed global
    # VideoConfig.seed is set (that already pins every shot). On by default.
    consistent_character_seed: bool = True
    # Combine short back-and-forth exchanges into ONE shot covering both
    # speakers, instead of always hard-cutting to a new shot per line. Only
    # merges consecutive shots that already frame the SAME 2+ characters in a
    # non-close-up composition (a held two-shot/wide) and alternate speakers —
    # i.e. genuine exchanges, not just adjacent monologue beats. Merged shots
    # are explicitly routed to wan_i2v (general motion) rather than wan_s2v,
    # since their audio now carries more than one voice and S2V lip-syncs to
    # a single face — close-ups always stay one-speaker-per-shot so S2V can
    # still lip-sync those precisely.
    merge_dialogue_shots: bool = False
    merge_max_lines_per_shot: int = 3
    # External lip-sync engines receive ONE video and ONE audio file. If a shot
    # shows one face but its audio contains narrator speech or a different
    # character, the visible mouth will try to sync to the wrong voice. Keep
    # this on: before producing, split any mixed-audio shot into consecutive
    # speaker-specific shots whenever the audio speaker changes. Narrator-only
    # split shots are treated as cinematic cutaways and are never lip-synced.
    split_audio_on_speaker_change: bool = True
    # For I2V + external lip-sync, the first non-narrator spoken line in a shot
    # establishes the face that should be visible in that shot. If the script
    # generator creates a shot with a visible character but the first character
    # dialogue belongs to someone else, normalize the shot to cut to the actual
    # first speaker before image prompts are authored.
    enforce_visible_speaker_first_dialogue: bool = True
    # Keep spoken words out of image/video prompts. Dialogue belongs only in
    # audio generation. Image prompts describe the initial still; motion prompts
    # describe only physical/camera movement. This prevents image models from
    # rendering captions, subtitles, speech bubbles, or quoted dialogue text.
    exclude_dialogue_text_from_visual_prompts: bool = True
    # If the image prompt / shot contains no visible character, the motion
    # prompt must stay non-human as well: no invented people, no faces, no
    # mouth/eye/expression cues, and no human actions. Motion should describe
    # only relevant environment/object/camera movement already implied by the
    # image prompt.
    motion_prompts_respect_no_people_scenes: bool = True
    # Frame each shot for what it actually needs: a short, emotionally-charged
    # single-speaker line gets a close-up (drama reads best tight); a long
    # dialogue block gets pushed to a wide/landscape framing (faces small —
    # easier to sustain convincingly over a long take, and lip-sync precision
    # matters less when the face isn't filling the frame); physical action
    # gets pulled OUT of close-up so the body and the interaction are
    # actually visible. Only ever nudges composition toward the better
    # choice — never overrides a shot that's already framed deliberately for
    # the same reason.
    direct_shot_composition: bool = True
    heavy_dialogue_word_threshold: int = 35      # this many+ words → push wide/landscape
    # When a scene has 2+ characters PRESENT (characters_in_frame) but speech
    # is concentrated in too few of them, insert a few short reactive lines
    # for the silent-but-present ones — into existing reaction/cutaway shots
    # that already frame them and have no line yet, so it never creates a
    # same-shot two-speaker conflict for the lip-synced engine. The upstream
    # script generator decides the initial cast distribution; this is the one
    # pass that can actually change WHO speaks, not just polish what's there.
    balance_scene_dialogue: bool = True
    balance_max_new_lines_per_scene: int = 2
    cinematic_prompts: bool = True              # fold emotion→lighting/lens grammar into prompts
    # Push the still prompts toward richer, more imaginative, more dramatic
    # imagery: fine texture/material detail, atmospheric depth (haze, particles,
    # rim light), emotionally expressive faces + body language mapped from the
    # beat's feeling, and dynamic/dramatic composition. Purely additive to the
    # prompt wording; never overrides character locks, mouth-visibility, or the
    # style anchor. On by default.
    expressive_detail: bool = True
    emotional_arc: bool = True                  # nudge the dialogue pass toward a deliberate arc
    theme: str = "cinematic, dramatic lighting, filmic color, shallow depth of field"

    # ── Creative quality passes (all optional, all on by default) ────────────
    # 1. Story doctor: pressure-test the premise before committing to a script.
    #    Checks for: a character who wants something, an obstacle, a cost, and
    #    a change. Returns an enriched premise if the original is weak; otherwise
    #    passes through unchanged. Only fires in story-idea mode.
    story_doctor: bool = True
    # 2. Visual treatment: generate a per-film style bible (signature palette,
    #    light motif, compositional grammar, texture motif) once at plan() time,
    #    then thread it into every image prompt as creative vocabulary.
    visual_treatment: bool = True
    # 3. Emotional arc map: compute the intended emotional trajectory of the
    #    whole film once (lows, turn, peak, release) keyed by shot position.
    #    Every image prompt and pacing decision references the same authored map.
    emotional_arc_map: bool = True
    # 4. Prompt review-and-revise: after generate_image_prompts(), a critic LLM
    #    checks each batch for story-service, visual monotony vs. adjacent shots,
    #    and lock-consistency. Flagged prompts are revised in-place.
    prompt_review: bool = True
    prompt_review_batch_size: int = 6
    # 5. Shot-to-shot connective tissue: each image prompt receives a one-line
    #    summary of the previous shot so colour, eyeline, and compositional
    #    echoes can carry across the cut.
    shot_continuity_context: bool = True

    def workdir(self) -> Path:
        p = Path(self.out_root) / _slug(self.title)
        (p / "images").mkdir(parents=True, exist_ok=True)
        (p / "audio").mkdir(parents=True, exist_ok=True)
        (p / "clips").mkdir(parents=True, exist_ok=True)
        return p


# =============================================================================
# CORE DATA MODEL
# =============================================================================

@dataclass
class Line:
    speaker: str
    text: str
    emotion: str = "neutral"      # derived from bubble_type/emphasis
    audio_path: Optional[str] = None
    duration: float = 0.0


@dataclass
class Shot:
    index: int
    description: str = ""
    setting: str = ""
    mood: str = ""
    composition: str = "medium_shot"
    characters_in_frame: List[str] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)
    image_prompt: Optional[str] = None         # structured KLEIN2/Z-Image prompt (long, detailed)
    motion_prompt: Optional[str] = None        # SHORT prompt for the animation engine (motion only)
    # Pre-extracted motion brief produced by comic_book_generator._extract_action_sequence().
    # Contains ONLY the action/motion clauses from the panel description — already
    # stripped of static-scene language — ready to seed the animation model directly.
    # Empty string for static panels with no detected motion.  None means CBG did not
    # populate this field (older manifests or user-script mode), in which case
    # generate_motion_prompts() falls back to its normal LLM+description path.
    action_sequence: Optional[str] = None
    duration_hint: Optional[float] = None       # optional manual seconds override
    anchor_images: List[str] = field(default_factory=list)  # extra starting stills for long chains
    anchor_count: int = 0                        # how many re-anchor stills this shot needs
    # produced artifacts
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    duration: float = 0.0
    engine: str = ""               # filled by the router
    video_path: Optional[str] = None
    extra_videos: Dict[str, str] = field(default_factory=dict)  # compare_all mode

    @property
    def is_dialogue(self) -> bool:
        return any(l.text.strip() and l.speaker.upper() != "NARRATOR" for l in self.lines)

    @property
    def speaking_character(self) -> Optional[str]:
        for l in self.lines:
            if l.text.strip() and l.speaker.upper() != "NARRATOR":
                return l.speaker
        return None


@dataclass
class VoiceProfile:
    name: str
    ref_wav: Optional[str]         # reference clip for cloning (may be None → engine default)
    gender: str = ""
    descriptor: str = ""           # e.g. "warm, low, measured"
    source: str = "engine_default"  # clone | bank | baseline | engine_default
    seed: int = 0                  # the random seed assigned to this character
    ref_text: str = ""             # transcript of ref_wav (for F5/E2)
    # Self-generated baseline reference (closes the consistency gap for
    # "engine_default" characters — no clone/bank ref means no fixed file to
    # clone from, so nothing guarantees the same voice line-to-line). Edit
    # baseline_text and set regenerate_baseline=True in the plan to redo it.
    baseline_text: str = ""        # the line synthesized into the baseline clip
    baseline_path: Optional[str] = None   # where that clip lives once generated
    regenerate_baseline: bool = False     # force a redo even if cached
    # engine-specific knobs (e.g. chatterbox exaggeration baseline)
    params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "" for c in s).strip()
    return ("_".join(keep.split()) or "project")[:64]


def _notebook_project_root() -> Path:
    """Project root used to resolve notebook-relative paths.

    configure_notebook_local_paths() sets STORY_ANIMATION_PROJECT_ROOT. This
    matters because LatentSync runs with cwd changed to its own repo, while your
    project assets live under ./animation_out and ./model_cache.
    """
    return Path(os.environ.get("STORY_ANIMATION_PROJECT_ROOT", ".")).expanduser().resolve()


def _resolve_project_path(pathlike: Optional[str], *, base: Optional[Path] = None) -> Path:
    """Resolve a user path against the notebook project root unless absolute."""
    if pathlike is None or str(pathlike).strip() == "":
        return Path("")
    p = Path(str(pathlike)).expanduser()
    if p.is_absolute():
        return p
    return ((base or _notebook_project_root()) / p).resolve()


def _safe_symlink_or_copy(src: Path, dst: Path) -> str:
    """Create dst pointing at src. Prefer a relative symlink; copy if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        rel = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel)
        return "symlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def ensure_latentsync_whisper_layout(repo_dir: str = "./model_cache/LatentSync") -> Dict[str, Any]:
    """Repair the Whisper checkpoint layout expected by LatentSync inference.py.

    LatentSync's official scripts/inference.py hard-codes these repo-relative
    paths based on the UNet config:
        checkpoints/whisper/tiny.pt
        checkpoints/whisper/small.pt

    Some local downloads place the weights at:
        LatentSync/whisper/tiny.pt
        LatentSync/whisper/small.pt

    This creates the expected path using a symlink, falling back to copying if
    symlinks are unavailable. It does not download anything.
    """
    repo = _resolve_project_path(repo_dir)
    result: Dict[str, Any] = {
        "repo_dir": str(repo),
        "repo_exists": repo.is_dir(),
        "actions": {},
        "expected": {},
        "ok": False,
        "missing": [],
    }
    if not repo.is_dir():
        result["missing"].append(f"LatentSync repo not found: {repo}")
        return result

    for name in ("tiny.pt", "small.pt"):
        expected = repo / "checkpoints" / "whisper" / name
        result["expected"][name] = str(expected)
        if expected.exists():
            result["actions"][name] = "exists"
            continue

        candidates = [
            repo / "whisper" / name,
            repo / name,
            repo / "checkpoints" / name,
        ]
        try:
            candidates.extend([q for q in repo.rglob(name) if q != expected])
        except Exception:
            pass

        src = next((q for q in candidates if q.exists() and q.is_file()), None)
        if src is None:
            result["actions"][name] = "missing_source"
            if name == "tiny.pt":
                result["missing"].append(
                    f"{name} not found. Put it at {expected} or {repo / 'whisper' / name}."
                )
            continue
        result["actions"][name] = _safe_symlink_or_copy(src, expected)

    result["ok"] = (repo / "checkpoints" / "whisper" / "tiny.pt").exists()
    return result


def _free_vram():
    if _HAS_TORCH and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        # FramePack/Hunyuan swaps large models repeatedly. Resetting CUDA peak
        # and accumulated stats is not required for correctness, but it helps
        # long notebook sessions avoid stale allocator telemetry and mirrors the
        # standalone FramePack notebook's reset_memory() practice.
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()    # extra fragmentation cleanup between model swaps
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _vram_free_gb() -> float:
    if _HAS_TORCH and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    return 0.0


def _gpu_has_fp8_tensor_cores() -> Optional[bool]:
    """True if the current GPU has native FP8 (e4m3) matmul tensor cores.

    FP8 tensor cores arrived with compute capability 8.9 (Ada Lovelace, RTX
    40xx) and are present on Hopper (9.0) and Blackwell (10.x+). Ampere
    (RTX 30xx / A100, 8.0-8.6) has NO native FP8 matmul — running a
    pre-quantized fp8 checkpoint there raises 'fp8 matmul not supported'.

    Returns None when it can't be determined (no torch / no CUDA), so callers
    can choose to proceed rather than block on missing information.
    """
    if not (_HAS_TORCH and torch.cuda.is_available()):
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)
    except Exception:
        return None


@contextlib.contextmanager
def _quiet_config_mismatch_warnings():
    """Suppress diffusers' benign "config attributes ... were not expected"
    warning during a single from_pretrained() call.

    This fires when a model repo's config.json (e.g. Wan2.2's VAE config)
    carries a field the installed diffusers' class doesn't know about yet
    (or vice versa, on an older repo snapshot vs. newer diffusers) — a pure
    library/repo version skew, not a real problem; the field is just ignored
    and the model loads and runs fine. Restored immediately after the load so
    genuinely useful warnings elsewhere are never hidden. The real fix for the
    underlying skew is keeping diffusers current — see the install notes.
    """
    try:
        from diffusers.utils import logging as _dlog
    except Exception:
        yield
        return
    prev = _dlog.get_verbosity()
    _dlog.set_verbosity_error()
    try:
        yield
    finally:
        _dlog.set_verbosity(prev)


_CUDA_DIAGNOSED = False     # module-level: print the NCCL/torch mismatch guidance only once


def _diagnose_torch_cuda_error(exc: Exception) -> None:
    """Print one clear, actionable message for a broken torch/NCCL CUDA stack.

    `undefined symbol: ncclCommResume` (or similar undefined-symbol errors out
    of libtorch_cuda.so) means the installed `nvidia-nccl-cu12` package doesn't
    match what this torch build was compiled against — usually because another
    package (deepspeed/vllm/xformers/a conda nccl) pulled in a different NCCL
    version, or torch.cuda.is_available() lazily loads a CUDA extension that
    can't actually resolve its symbols. torch.cuda.is_available() can still
    return True in this state, since the mismatch only surfaces once a CUDA
    op actually runs. This is an environment issue, not a pipeline bug — it
    will affect EVERY CUDA stage (TTS engines, KLEIN2, video engines), so it's
    worth fixing before a long render rather than letting each stage fall back.
    """
    global _CUDA_DIAGNOSED
    if _CUDA_DIAGNOSED:
        return
    _CUDA_DIAGNOSED = True
    msg = str(exc)
    if "undefined symbol" not in msg:
        return
    logger.warning(
        "\n"
        "════════════════════════════════════════════════════════════════\n"
        "  torch CUDA extension failed to load: %s\n"
        "  This is a version mismatch between torch and the installed NCCL\n"
        "  library (common after installing/upgrading deepspeed, vllm,\n"
        "  xformers, or a conda nccl package alongside pip torch). It will\n"
        "  likely affect every GPU stage in this pipeline, not just voices.\n"
        "  To fix, in this environment run:\n"
        "      python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
        "      pip show nvidia-nccl-cu12\n"
        "  then reinstall torch's CUDA stack as a matched set, e.g.:\n"
        "      pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12\n"
        "      pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu124\n"
        "  (swap cu124 for the CUDA build matching your driver). If a conda\n"
        "  nccl package is also installed (`conda list | grep -i nccl`),\n"
        "  remove it or unset LD_LIBRARY_PATH so it doesn't shadow torch's own.\n"
        "  Until this is fixed, GPU stages will fall back where possible\n"
        "  (e.g. engine-default voices) rather than crash the run.\n"
        "════════════════════════════════════════════════════════════════",
        msg.splitlines()[-1].strip() if msg else exc)


_DEVICE_NOT_READY_DIAGNOSED = False   # module-level: print this guidance only once


def _diagnose_device_not_ready(exc: Exception) -> None:
    """Print one clear, actionable message for a "CUDA driver error: device
    not ready" failure during a heavy model load.

    This error message is misleading on its own — it sounds like a driver/
    hardware problem, but in this pipeline it's almost always the SECOND
    symptom of a near-OOM that happened a moment earlier: a 14B+ DiT loading
    unquantized (e.g. the bnb4 patch failing to apply — see
    _log_dit_quant_status's log line) overshoots a 24GB card mid-shard-copy,
    which corrupts the CUDA context rather than raising a clean "out of
    memory" message. The corrupted context generally can't recover within
    the same process, which is why an automatic in-process retry tends to
    fail the same way again.
    """
    global _DEVICE_NOT_READY_DIAGNOSED
    if _DEVICE_NOT_READY_DIAGNOSED:
        return
    msg = str(exc)
    if "device not ready" not in msg.lower() and "not ready" not in msg.lower():
        return
    _DEVICE_NOT_READY_DIAGNOSED = True
    logger.warning(
        "\n"
        "════════════════════════════════════════════════════════════════\n"
        "  'device not ready' usually means a near-OOM corrupted the CUDA\n"
        "  context a moment earlier — not a driver/hardware fault. The\n"
        "  most common cause here: the bnb4 quantization patch didn't\n"
        "  actually apply, so a 14B+ DiT tried to land on the GPU\n"
        "  unquantized at full bf16 (~28GB), well over a 24GB card. Check\n"
        "  the log lines just above the failure for:\n"
        "    'WanModel_S2V.from_pretrained found (...)' — confirms the\n"
        "       patch applied (and whether the method was inherited).\n"
        "    'Wan2.2-S2V DiT confirmed 4-bit (...)' vs 'NO 4-bit params\n"
        "       were found' — confirms whether quantization actually took.\n"
        "  A corrupted CUDA context generally can't be recovered within\n"
        "  the same process, so if the automatic retry fails the same way,\n"
        "  this pipeline's wan_s2v_fallback_engine will take over for those\n"
        "  shots (wan_i2v + a lipsync pass) rather than dropping them. To\n"
        "  get a genuine second attempt at real lip-synced S2V instead of\n"
        "  the fallback, restart the Python process/kernel — this pipeline\n"
        "  is idempotent via resume=True, so nothing already completed is\n"
        "  lost.\n"
        "════════════════════════════════════════════════════════════════")


# Signatures of an UNRECOVERABLE, process-wide GPU fault. Once one of these
# fires mid-render, the CUDA context/allocator is corrupted: retrying the same
# shot fails identically, and every *later* shot on the same loaded engine fails
# the same way too. The only cure is to tear the engine down and hand its
# remaining shots to a fallback engine (a fresh load reinitialises the context).
# So when we see one we stop calling the model for this engine immediately
# instead of burning ~one full render + one retry on every remaining shot.
_FATAL_GPU_SIGNATURES = (
    "cudacachingallocator",          # the exact allocator INTERNAL ASSERT here
    "!handles_.at",                  # allocator handle table corruption
    "internal assert failed",        # torch INTERNAL ASSERT (allocator/context)
    "device-side assert",            # device-side assertion → context dead
    "an illegal memory access",      # illegal access → context dead
    "misaligned address",
    "unspecified launch failure",
    "uncorrectable ecc",
)
# Broader "this was a GPU/CUDA failure" signal. Used as a secondary trigger:
# if a shot fails BOTH attempts and the failure is CUDA-related, the context is
# almost certainly gone, so we also bail rather than grind through the rest.
_CUDA_ERROR_SIGNATURES = (
    "cuda", "cudnn", "cublas", "device not ready", "out of memory",
    "device-side", "nvrtc", "hip error", "gpu",
)


def _is_fatal_gpu_error(err: object) -> bool:
    """True for an unrecoverable, context-corrupting GPU fault (bail now)."""
    msg = str(err).lower()
    return any(s in msg for s in _FATAL_GPU_SIGNATURES)


def _is_cuda_error(err: object) -> bool:
    """True when an exception looks GPU/CUDA-related at all."""
    if err is None:
        return False
    msg = str(err).lower()
    return any(s in msg for s in _CUDA_ERROR_SIGNATURES)


def _wav_duration(path: str) -> float:
    if _HAS_SF:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    # ffprobe fallback
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path]
        )
        return float(out.strip())
    except Exception:
        return 0.0


def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(max(0.0, seconds) * sr), dtype=np.float32)


def _normalize_loudness(wav: np.ndarray, target_rms: float = 0.12,
                        peak_ceiling: float = 0.97) -> np.ndarray:
    """Scale a clip to a consistent RMS so shots sit at the same perceived level,
    then hard-limit the peak so the gain-up never clips. Silence is left alone."""
    wav = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    if rms > 1e-5:
        wav = wav * (target_rms / rms)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > peak_ceiling:
        wav = wav * (peak_ceiling / peak)
    return wav.astype(np.float32)


def _apply_edge_fades(wav: np.ndarray, sr: int, ms: int = 12) -> np.ndarray:
    """Short linear fade in/out on the clip edges so concatenated shots don't
    click/pop at the cut."""
    n = int(sr * max(0, ms) / 1000)
    if n <= 0 or wav.size < 2 * n:
        return wav
    wav = wav.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


def _emotion_from_bubble(bubble_type: str, emphasis: str = "normal") -> str:
    """Map the comic bubble taxonomy → a delivery emotion the TTS layer understands."""
    bt = (bubble_type or "speech").lower()
    table = {
        "shout": "angry", "excited": "excited", "angry": "angry",
        "scared": "scared", "tender": "tender", "cold": "cold",
        "sarcastic": "sarcastic", "whisper": "whisper",
        "thought": "intimate", "caption": "narration", "speech": "neutral",
    }
    emo = table.get(bt, "neutral")
    if emphasis == "emphasized" and emo == "neutral":
        emo = "emphatic"
    return emo


def _run_ffmpeg(args: List[str]):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)


def _resolve_torchao_quant(mode: str):
    """Return a zero-arg callable producing a torchao weight-only quant config
    for `quantize_`, robust across torchao versions, or None if unavailable.

    torchao renamed the factory helpers (e.g. ``int8_weight_only``) to config
    classes (``Int8WeightOnlyConfig``) and shuffled them between
    ``torchao.quantization`` and ``torchao.quantization.quant_api``. We probe
    every known spelling so int8/fp8 keeps working instead of silently falling
    back to bf16 (which costs VRAM on a 4090).
    """
    mode = (mode or "").lower()
    if mode == "int8":
        names = ["int8_weight_only", "Int8WeightOnlyConfig"]
    elif mode == "fp8":
        names = ["float8_weight_only", "Float8WeightOnlyConfig"]
    else:
        return None
    import importlib
    for modpath in ("torchao.quantization", "torchao.quantization.quant_api"):
        try:
            m = importlib.import_module(modpath)
        except Exception:
            continue
        for nm in names:
            fn = getattr(m, nm, None)
            if fn is not None:
                return fn            # both factory fns and config classes are callable
    return None


def _torchao_quantize_fn():
    """Resolve torchao's ``quantize_`` across versions, or None."""
    import importlib
    for modpath in ("torchao.quantization", "torchao.quantization.quant_api"):
        try:
            m = importlib.import_module(modpath)
        except Exception:
            continue
        fn = getattr(m, "quantize_", None)
        if fn is not None:
            return fn
    return None


def _cap_resolution_for_4090(w: int, h: int, max_long: int = 1280, max_short: int = 720,
                             multiple: int = 16) -> Tuple[int, int]:
    """Scale (w, h) down — never up — to fit within a 720p-equivalent budget
    (long side <= 1280, short side <= 720, in whichever orientation the
    source has), then snap both dimensions to a multiple of 16 (required by
    every video/image model here: Wan, Z-Image, KLEIN2 VAEs).

    This is the "resolution should match the input image, capped to 720p on
    a 4090" rule: a smaller source image is left at its own resolution
    (just snapped to /16); a larger one is downscaled to fit, preserving
    aspect ratio.
    """
    w, h = max(1, int(w)), max(1, int(h))
    target_long, target_short = (max_long, max_short) if w >= h else (max_short, max_long)
    long_side, short_side = max(w, h), min(w, h)
    scale = min(target_long / long_side, target_short / short_side, 1.0)   # never upscale
    nw, nh = w * scale, h * scale
    nw = max(multiple, int(round(nw / multiple)) * multiple)
    nh = max(multiple, int(round(nh / multiple)) * multiple)
    return nw, nh


def _ensure_hf_snapshot(repo_id: str, local_dir: str, allow_patterns: Optional[List[str]] = None,
                        enabled: bool = True) -> str:
    """Download a model repo's files to `local_dir` if they're not already
    there, via huggingface_hub.snapshot_download — the auto-download path for
    models that need a literal local checkpoint DIRECTORY (not a plain
    diffusers/transformers .from_pretrained() call, which already
    auto-downloads and caches on its own).
    """
    p = Path(local_dir)
    if p.exists() and any(p.iterdir()):
        return str(p)                      # already there
    if not enabled:
        raise RuntimeError(
            f"{local_dir!r} doesn't exist or is empty, and auto_download_models "
            f"is off. Either download {repo_id} there yourself, or enable "
            f"auto_download_models.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download model files "
            "(pip install huggingface_hub).") from e
    logger.info("[DOWNLOAD] %s not found locally — downloading from %s "
               "(this can take a while the first time)…", local_dir, repo_id)
    p.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(p), allow_patterns=allow_patterns)
    logger.info("[DOWNLOAD] %s ready.", local_dir)
    return str(p)


def _ensure_local_repo(repo_url: str, local_dir: str, enabled: bool = True) -> str:
    """Clone a GitHub repo to `local_dir` if it's not already there (for code
    that isn't pip-installable as a package, e.g. Wan-Video/Wan2.2's `wan`
    package) via a plain `git clone` — still entirely local once cloned; the
    one-time clone itself needs network access to GitHub specifically (not a
    model-hosting cloud API).
    """
    p = Path(local_dir)
    if p.exists() and any(p.iterdir()):
        return str(p)
    if not enabled:
        raise RuntimeError(
            f"{local_dir!r} doesn't exist or is empty, and auto_download_models "
            f"is off. Clone {repo_url} there yourself, or enable "
            f"auto_download_models.")
    if shutil.which("git") is None:
        raise RuntimeError(
            f"git isn't available to clone {repo_url} automatically. Install "
            f"git, or clone it yourself to {local_dir!r}.")
    logger.info("[DOWNLOAD] cloning %s → %s …", repo_url, local_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(p)], check=True)
    logger.info("[DOWNLOAD] %s ready.", local_dir)
    return str(p)


def _probe_fps(path: str) -> float:
    """Native frame rate of a video file (ffprobe); 0.0 if it can't be read."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip()
        if "/" in out:
            num, den = out.split("/")
            den = float(den)
            return float(num) / den if den else 0.0
        return float(out)
    except Exception:
        return 0.0


def _probe_duration(path: str) -> float:
    """Duration of a media file in seconds (ffprobe); 0.0 if it can't be read."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path]).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def _clip_has_audio(path: str) -> bool:
    """Whether a media file has at least one audio stream (ffprobe)."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path]
        ).decode()
        return "audio" in out
    except Exception:
        return False


def _probe_stream_duration(path: str, stream: str = "v:0") -> float:
    """Duration for a specific stream (ffprobe); falls back to container duration.

    Container duration can hide small audio/video mismatches. Stream duration is
    what matters for sync QA and for deciding whether to pad video or audio before
    concatenation.
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", stream,
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path]
        ).decode().strip().splitlines()
        for val in out:
            val = val.strip()
            if val and val.upper() != "N/A":
                return float(val)
    except Exception:
        pass
    return _probe_duration(path)


def _normalize_silent_video_duration(video_in: str, video_out: str,
                                     fps: int, target_seconds: float) -> str:
    """Write a silent H.264 clip at exact target duration/fps.

    FramePack natively generates at its own 30 fps cadence and may over-generate
    a little because latent sections are discrete. This helper trims or pads by
    holding the last frame so downstream audio muxing and final concat receive a
    deterministic clip length.
    """
    src = str(video_in)
    dst = str(video_out)
    fps = int(max(1, fps))
    target = float(max(0.05, target_seconds))
    vdur = _probe_stream_duration(src, "v:0")
    vf = f"fps={fps}"
    if vdur > 0 and vdur < target - 0.02:
        extra = target - vdur + (1.0 / fps)
        vf = f"tpad=stop_mode=clone:stop_duration={extra:.3f},fps={fps}"
    _run_ffmpeg([
        "-i", src,
        "-vf", vf,
        "-t", f"{target:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-an", dst,
    ])
    return dst



def _dialogue_speakers(shot: "Shot") -> List[str]:
    """Distinct non-narrator speakers with spoken text in this shot."""
    speakers: List[str] = []
    for ln in shot.lines or []:
        if ln.text.strip() and ln.speaker.upper() != "NARRATOR" and ln.speaker not in speakers:
            speakers.append(ln.speaker)
    return speakers


def _has_multispeaker_dialogue(shot: "Shot") -> bool:
    return len(_dialogue_speakers(shot)) > 1


def _sync_delta(video_path: Optional[str], audio_path: Optional[str]) -> Optional[float]:
    if not (video_path and Path(video_path).exists() and audio_path and Path(audio_path).exists()):
        return None
    return _probe_stream_duration(video_path, "v:0") - _wav_duration(audio_path)


def _cached_clip_is_valid(path: str, sh: "Shot") -> bool:
    """Whether a previously-rendered shot clip can be trusted by resume=True.

    Guards against stale clips left over from an earlier version of this
    pipeline (e.g. before an audio-muxing fix): a shot that's SUPPOSED to
    carry audio (sh.audio_path set) must have an audio stream in the cached
    file, and the cached file must be at least as long as that audio — a
    plain "does the file exist" check isn't enough to trust it.
    """
    if not Path(path).exists():
        return False
    if sh.audio_path and Path(sh.audio_path).exists():
        if not _clip_has_audio(path):
            return False
        audio_dur = _wav_duration(sh.audio_path)
        if audio_dur > 0:
            vdur = _probe_stream_duration(path, "v:0")
            if vdur < audio_dur - 0.15:
                return False
            # Also reject stale clips with substantial drift. Small differences
            # are normal because frame counts land on whole-frame boundaries.
            if abs(vdur - audio_dur) > 0.75:
                return False
    return True


# =============================================================================
# TTS ENGINES  (pluggable; uniform interface)
# =============================================================================
#
#   load()                            -> load the model onto GPU
#   register(name, ref_wav, params)   -> prepare/cache a speaker from a ref clip
#   synth(text, name, emotion) -> np.ndarray (float32 mono @ self.sr)
#   unload()                          -> free VRAM
#
# Engines that don't natively expose an "emotion" lever fall back to neutral,
# text-only delivery; only Chatterbox maps emotion onto its own controls
# (exaggeration/cfg_weight).
# -----------------------------------------------------------------------------

class BaseTTS:
    name = "base"

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.device = cfg.device if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self._speakers: Dict[str, Any] = {}

    def load(self):  # pragma: no cover - heavy
        raise NotImplementedError

    def register(self, name: str, ref_wav: Optional[str], params: Dict[str, Any]):
        # Default: just remember the ref + params; engines override if they need
        # to precompute a conditioning latent.
        self._speakers[name] = {"ref_wav": ref_wav, "params": dict(params or {})}

    def synth(self, text: str, name: str, emotion: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def unload(self):
        for attr in ("model", "pipe", "tts"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        self._speakers.clear()
        _free_vram()


class ChatterboxTTS(BaseTTS):
    """Resemble AI Chatterbox — default. Native emotion-exaggeration control."""
    name = "chatterbox"

    # bubble emotion -> (exaggeration, cfg_weight). Higher exaggeration = hotter.
    EMO = {
        "neutral": (0.5, 0.5), "emphatic": (0.65, 0.5), "narration": (0.4, 0.5),
        "excited": (0.85, 0.4), "angry": (0.9, 0.35), "scared": (0.8, 0.4),
        "tender": (0.35, 0.6), "cold": (0.3, 0.6), "sarcastic": (0.6, 0.45),
        "whisper": (0.3, 0.65), "intimate": (0.4, 0.6),
    }

    def load(self):
        from chatterbox.tts import ChatterboxTTS as _CB
        self.model = _CB.from_pretrained(device=self.device)
        self.sr = self.model.sr
        self.cfg.sample_rate = self.sr

    def synth(self, text: str, name: str, emotion: str) -> np.ndarray:
        spk = self._speakers.get(name, {})
        ex, cfgw = self.EMO.get(emotion, self.EMO["neutral"])
        ex = float(spk.get("params", {}).get("exaggeration_bias", 0.0)) + ex
        ex = max(0.25, min(1.5, ex))
        kwargs = dict(exaggeration=ex, cfg_weight=cfgw)
        ref = spk.get("ref_wav")
        if ref:
            kwargs["audio_prompt_path"] = ref
        wav = self.model.generate(text, **kwargs)        # torch tensor [1, T]
        return wav.squeeze().detach().float().cpu().numpy()


class F5TTS(BaseTTS):
    """F5-TTS — flow-matching, fast, zero-shot cloning from a short ref clip."""
    name = "f5"

    def load(self):
        from f5_tts.api import F5TTS as _F5
        self.model = _F5(device=self.device)
        self.sr = getattr(self.model, "target_sample_rate", 24000)
        self.cfg.sample_rate = self.sr

    def register(self, name, ref_wav, params):
        super().register(name, ref_wav, params)
        # F5 wants a transcript of the reference; allow it via params["ref_text"].

    def synth(self, text, name, emotion):
        spk = self._speakers.get(name, {})
        ref = spk.get("ref_wav")
        ref_text = spk.get("params", {}).get("ref_text", "")
        wav, sr, _ = self.model.infer(ref_file=ref, ref_text=ref_text, gen_text=text)
        self.sr = sr
        return np.asarray(wav, dtype=np.float32)


class E2TTS(F5TTS):
    """E2-TTS — the E2-TTS checkpoint that ships in the same SWivid/F5-TTS repo.

    Flat-UNet (non-DiT) flow-matching TTS; very natural zero-shot cloning. Like
    F5-TTS it needs a *transcript* of the reference clip (auto-filled by Whisper
    in the voice-casting stage when you don't supply one).
    """
    name = "e2"

    def load(self):
        from f5_tts.api import F5TTS as _F5
        # The constructor arg has shifted across releases; try the known forms.
        for kwargs in (
            {"model": "E2TTS_Base"},
            {"model_type": "E2-TTS"},
            {"model": "E2-TTS"},
        ):
            try:
                self.model = _F5(device=self.device, **kwargs)
                break
            except TypeError:
                continue
        else:  # pragma: no cover
            raise RuntimeError("Could not select the E2-TTS model in this f5_tts build "
                               "— check the F5TTS() constructor's model argument.")
        self.sr = getattr(self.model, "target_sample_rate", 24000)
        self.cfg.sample_rate = self.sr


class HiggsAudioV3(BaseTTS):
    """bosonai/higgs-audio-v3-tts-4b, ported to run on plain transformers —
    multimodalart/higgs-audio-v3-tts-4b-transformers. Zero-shot TTS with
    voice cloning from a reference clip (+ optional reference transcript,
    auto-filled via Whisper if not supplied — see _NEEDS_REF_TEXT).

    Avoids the original bosonai release's custom `boson_multimodal` package
    (which pulls its own model-serving stack) in favor of the community port
    that loads through plain `AutoModelForCausalLM`/`AutoTokenizer`, the same
    transformers>=5 install every other engine here already needs.
    """
    name = "higgs"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = self.cfg.model_id_override or TTS_MODEL_IDS["higgs"]
        self.tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True, dtype=torch.bfloat16
        ).eval()
        if self.device == "cuda" and _HAS_TORCH and torch.cuda.is_available():
            self.model.to("cuda")
        try:
            self.model.get_audio_codec()           # preload the 24kHz codec
        except Exception:
            pass
        self.sr = int(getattr(self.model.config, "sample_rate", 24000))
        self.cfg.sample_rate = self.sr

    def synth(self, text, name, emotion):
        spk = self._speakers.get(name, {})
        ref = spk.get("ref_wav")
        params = spk.get("params", {}) or {}
        top_p = float(params.get("top_p", 0.95))
        top_k = int(params.get("top_k", 50))
        kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 0.7)),
            top_p=top_p if top_p < 1.0 else None,
            top_k=top_k if top_k > 0 else None,
        )
        if ref:
            try:
                import soundfile as _sf
                data, sr = _sf.read(ref, dtype="float32", always_2d=True)   # [L, C]
                wav = torch.from_numpy(data).mean(dim=1)                    # mono [L]
                kwargs["reference_audio"] = wav
                kwargs["reference_sample_rate"] = sr
                ref_text = params.get("ref_text")
                if ref_text:
                    kwargs["reference_text"] = ref_text
            except Exception as e:
                logger.warning("  higgs: failed to load reference clip %r: %s", ref, e)
        with torch.no_grad():
            audio = self.model.generate_speech(text, self.tokenizer, **kwargs)
        self.sr = int(getattr(self.model.config, "sample_rate", self.sr))
        arr = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else audio
        return np.asarray(arr, dtype=np.float32).squeeze()


_TTS_REGISTRY = {
    "chatterbox": ChatterboxTTS, "f5": F5TTS, "e2": E2TTS, "higgs": HiggsAudioV3,
}


def make_tts(cfg: TTSConfig) -> BaseTTS:
    key = cfg.engine.lower()
    if key not in _TTS_REGISTRY:
        raise ValueError(f"Unknown TTS engine {cfg.engine!r}. "
                         f"Choose from {sorted(_TTS_REGISTRY)}.")
    return _TTS_REGISTRY[key](cfg)


# Engines that consume a transcript of the reference clip for cloning.
_NEEDS_REF_TEXT = {"f5", "e2", "higgs"}


# =============================================================================
# REFERENCE & AUDIO HELPERS  (mined from VideoSynthesisWorkflow.ipynb)
# =============================================================================

class WhisperTranscriber:
    """Lazy Whisper pipeline used to auto-caption reference clips for F5/E2-TTS.

    F5-TTS and E2-TTS clone *and* condition on a transcript of the reference
    clip; supplying the wrong text degrades the clone. Rather than make you
    transcribe every voice sample by hand, we transcribe once and cache.
    """
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline
            dtype = torch.float16 if (_HAS_TORCH and torch.cuda.is_available()) else torch.float32
            self._pipe = pipeline("automatic-speech-recognition", model=self.model_id,
                                  torch_dtype=dtype, device=self.device)

    def transcribe(self, wav_path: str) -> str:
        try:
            self._ensure()
            out = self._pipe(wav_path, chunk_length_s=30, batch_size=24,
                             generate_kwargs={"task": "transcribe"}, return_timestamps=False)
            return (out.get("text") or "").strip()
        except Exception as e:
            logger.warning("  Whisper transcription failed for %s: %s", wav_path, e)
            return ""

    def unload(self):
        self._pipe = None
        _free_vram()


def prepare_reference(ref_path: str, out_dir: Path, max_ms: int = 15000,
                      remove_silence: bool = True) -> str:
    """Silence-split + clip a reference clip to <=15s (mirrors process_voice()).

    Cloning engines behave best on a clean <=15s sample; long or gappy refs hurt
    quality. Returns a path to the prepared wav (or the original on any failure).
    """
    try:
        from pydub import AudioSegment, silence
        seg = AudioSegment.from_file(ref_path)
        if remove_silence:
            parts = silence.split_on_silence(seg, min_silence_len=1000,
                                             silence_thresh=-50, keep_silence=1000)
            if parts:
                merged = AudioSegment.silent(duration=0)
                for p in parts:
                    merged += p
                seg = merged
        if len(seg) > max_ms:
            seg = seg[:max_ms]
        out = out_dir / (Path(ref_path).stem + "_ref.wav")
        seg.export(str(out), format="wav")
        return str(out)
    except Exception as e:
        logger.debug("  prepare_reference fell back to raw clip (%s)", e)
        return ref_path


def chunk_text(text: str, max_chars: int = 320) -> List[str]:
    """Sentence-aware chunking so long lines don't degrade flow-matching TTS."""
    import re as _re
    chunks, cur = [], ""
    for sent in _re.split(r'(?<=[;:,.!?])\s+', text.strip()):
        if not sent:
            continue
        if len(cur.encode()) + len(sent.encode()) <= max_chars:
            cur += (" " if cur else "") + sent
        else:
            if cur:
                chunks.append(cur.strip())
            cur = sent
    if cur:
        chunks.append(cur.strip())
    return chunks or [text.strip()]


def _crossfade_concat(waves: List[np.ndarray], sr: int, fade_ms: int = 150) -> np.ndarray:
    """Concatenate per-chunk waves with an equal-power cross-fade at the seams."""
    waves = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves if len(w)]
    if not waves:
        return np.zeros(0, dtype=np.float32)
    if len(waves) == 1:
        return waves[0]
    fade = max(0, int(sr * fade_ms / 1000))
    out = waves[0]
    for w in waves[1:]:
        n = min(fade, len(out), len(w))
        if n == 0:
            out = np.concatenate([out, w])
            continue
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        out[-n:] = out[-n:] * np.cos(t) + w[:n] * np.sin(t)
        out = np.concatenate([out, w[n:]])
    return out


def _trim_silence(wav: np.ndarray, sr: int, thresh_db: float = -45.0,
                  keep_ms: int = 60) -> np.ndarray:
    """Trim leading/trailing near-silence, keeping a small lead-in/out."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return wav
    amp = 10 ** (thresh_db / 20.0)
    mask = np.abs(wav) > amp
    if not mask.any():
        return wav
    keep = int(sr * keep_ms / 1000)
    a = max(0, int(np.argmax(mask)) - keep)
    b = min(len(wav), len(wav) - int(np.argmax(mask[::-1])) + keep)
    return wav[a:b]


def resolve_ref_texts(voices: Dict[str, "VoiceProfile"], cfg: TTSConfig,
                      workdir: Path) -> None:
    """Ensure every cloning ref that needs a transcript has one.

    Order per voice: explicit cfg.ref_texts[name] → already-set params → Whisper.
    Only runs Whisper for engines in _NEEDS_REF_TEXT and only when a ref exists.
    No-op (and no Whisper load) otherwise.
    """
    if cfg.engine.lower() not in _NEEDS_REF_TEXT:
        return
    need = [(n, vp) for n, vp in voices.items()
            if vp.ref_wav and not (cfg.ref_texts.get(n) or vp.params.get("ref_text"))]
    # apply explicit texts first
    for n, vp in voices.items():
        if cfg.ref_texts.get(n):
            vp.params["ref_text"] = cfg.ref_texts[n]
    if not (need and cfg.auto_transcribe_refs):
        return
    logger.info("[REF] transcribing %d reference clip(s) with Whisper…", len(need))
    w = WhisperTranscriber(cfg.whisper_model, cfg.device)
    rdir = workdir / "audio"
    for n, vp in need:
        ref = vp.ref_wav
        if cfg.prepare_refs:
            ref = prepare_reference(ref, rdir, remove_silence=True)
            vp.ref_wav = ref
        vp.params["ref_text"] = w.transcribe(ref)
        logger.info("  %-20s ← \"%s…\"", n, (vp.params["ref_text"] or "")[:50])
    w.unload()


# =============================================================================
# VOICE CASTING  (hybrid sourcing: your clips → voice bank → engine default)
# =============================================================================
#
# A "voice bank" is just a folder of reference wavs. An optional index.json
# describes each so we can auto-cast by gender / age / descriptor:
#
#   ./voice_bank/
#       index.json
#       gravel_old_man.wav
#       bright_young_woman.wav
#       ...
#
#   index.json = [
#     {"file": "gravel_old_man.wav",   "gender": "male",   "tags": ["old","low","rough"]},
#     {"file": "bright_young_woman.wav","gender": "female", "tags": ["young","bright"]},
#     ...
#   ]
#
# If there's no index.json we fall back to filename keywords.
# -----------------------------------------------------------------------------

def _load_voice_bank(bank_dir: str) -> List[Dict[str, Any]]:
    p = Path(bank_dir)
    if not p.is_dir():
        return []
    idx = p / "index.json"
    if idx.exists():
        try:
            entries = json.loads(idx.read_text())
            for e in entries:
                e["path"] = str(p / e["file"])
            return [e for e in entries if Path(e["path"]).exists()]
        except Exception as e:
            logger.warning("voice_bank index.json unreadable (%s); using filenames.", e)
    out = []
    for wav in sorted(p.glob("*.wav")):
        stem = wav.stem.lower()
        gender = ("female" if any(t in stem for t in ("female", "woman", "girl", "_f_"))
                  else "male" if any(t in stem for t in ("male", "man", "boy", "_m_"))
                  else "")
        out.append({"file": wav.name, "path": str(wav), "gender": gender,
                    "tags": stem.replace("-", "_").split("_")})
    return out


def _voice_description_from_folder_name(name: str) -> str:
    """Convert a folder name into a compact voice description."""
    s = re.sub(r"[_\\/\-]+", " ", str(name or ""))
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _infer_gender_from_voice_description(desc: str) -> str:
    d = (desc or "").lower()
    if any(w in d for w in (
        "female", "woman", "girl", "lady", "actress", "princess",
        "angie", "emma", "lily", "jane", "grace", "daws", "flower"
    )):
        return "female"
    if any(w in d for w in (
        "male", "man", "boy", "guy", "old man", "judge", "farmer",
        "king", "knight", "assassin", "daniel", "dortice", "gerat",
        "dutch", "elon", "martin", "bear", "elder"
    )):
        return "male"
    if any(w in d for w in ("nonbinary", "non binary", "androgynous")):
        return "nonbinary"
    return ""


def _first_audio_file(folder: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    exts_l = {e.lower() for e in exts}
    try:
        files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts_l]
    except Exception:
        return None
    if files:
        return files[0]

    # Allow one shallow nested level for exported voice folders that place audio
    # inside a child folder.
    nested: List[Path] = []
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            try:
                nested.extend([p for p in sorted(child.iterdir()) if p.is_file() and p.suffix.lower() in exts_l])
            except Exception:
                pass
    return nested[0] if nested else None


def _load_character_voice_folders(cfg: "TTSConfig") -> List[Dict[str, Any]]:
    """Scan character_voice_dir for one reference audio file per voice folder."""
    root = _resolve_project_path(cfg.character_voice_dir)
    if not root.is_dir():
        logger.warning("[VOICE] character_voice_dir not found: %s", root)
        return []

    entries: List[Dict[str, Any]] = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue
        audio = _first_audio_file(folder, tuple(cfg.character_voice_audio_exts))
        if not audio:
            continue
        desc = _voice_description_from_folder_name(folder.name)
        entries.append({
            "id": folder.name,
            "folder": str(folder),
            "file": audio.name,
            "path": str(audio),
            "description": desc,
            "gender": _infer_gender_from_voice_description(desc),
            "tags": [t.lower() for t in re.split(r"\s+", desc) if t.strip()],
        })
    logger.info("[VOICE] scanned %d folder voice baseline(s) from %s.", len(entries), root)
    return entries


def list_character_voice_folder_descriptions(
    character_voice_dir: str = "/mnt/d/data/audio/characters",
    audio_exts: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"),
) -> List[Dict[str, Any]]:
    """Notebook helper: list folder voice descriptions and selected first audio files."""
    tmp = TTSConfig(character_voice_dir=character_voice_dir, character_voice_audio_exts=audio_exts)
    entries = _load_character_voice_folders(tmp)
    return [
        {
            "id": e["id"],
            "description": e["description"],
            "gender": e.get("gender", ""),
            "audio": e["path"],
            "file": e["file"],
        }
        for e in entries
    ]


def _character_voice_match_text(ch: "Character") -> str:
    """Text used to match a character to a folder voice description."""
    bits = []
    for attr in (
        "name", "gender", "role", "archetype", "age", "physical_build",
        "appearance", "personality", "voice_profile", "speech_pattern",
        "cadence", "dialect", "vocabulary_level", "wound", "longing",
    ):
        v = getattr(ch, attr, "") or ""
        if v:
            bits.append(f"{attr}: {v}")
    return " | ".join(str(b) for b in bits)


def _score_folder_voice_entry(entry: Dict[str, Any], ch: "Character") -> int:
    """Deterministic fallback voice-folder scoring."""
    desc = (entry.get("description") or "").lower()
    tags = set(entry.get("tags") or [])
    char_text = _character_voice_match_text(ch).lower()
    score = 0

    g = (getattr(ch, "gender", "") or "").lower()
    eg = (entry.get("gender") or "").lower()
    if g and eg:
        if eg in g or g in eg:
            score += 12
        elif eg in ("male", "female") and g in ("male", "female") and eg != g:
            score -= 8

    # Direct token overlap, excluding filler.
    stop = {"voice", "character", "the", "and", "old", "young", "male", "female"}
    char_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", char_text) if len(t) > 2 and t not in stop}
    for t in tags:
        if len(t) > 2 and t not in stop and t in char_terms:
            score += 3

    # Useful archetype/age/timbre cues.
    cue_pairs = [
        (("teen", "teenager", "boy", "girl"), ("teen", "teenager", "youth", "young")),
        (("elder", "old", "grumpy", "older"), ("elder", "old", "aged", "ancient", "wise", "grumpy")),
        (("demon", "ghost", "creepy"), ("demon", "ghost", "evil", "haunting", "supernatural", "paranormal", "monster")),
        (("ai", "computer", "cyborg", "robot", "bluetooth"), ("ai", "computer", "cyborg", "robot", "machine", "synthetic", "android")),
        (("announcer", "book reader"), ("announcer", "narrator", "formal", "broadcast", "reader")),
        (("judge",), ("authority", "formal", "judge", "court")),
        (("farmer", "country"), ("rural", "farmer", "country", "southern")),
        (("knight", "medevil", "medieval"), ("knight", "warrior", "medieval", "ancient")),
        (("assassin",), ("assassin", "killer", "cold", "stealth")),
        (("spaceship", "space ship"), ("spaceship", "space", "sci fi", "science fiction", "computer")),
    ]
    for voice_terms, char_terms2 in cue_pairs:
        if any(vt in desc for vt in voice_terms) and any(ct in char_text for ct in char_terms2):
            score += 7

    # Prefer named exact-ish matches if the character name intentionally matches
    # a voice folder.
    name = (getattr(ch, "name", "") or "").lower().replace("_", " ")
    if name and len(name) > 2 and name in desc:
        score += 20

    return score


def _ai_pick_folder_voice(ch: "Character", candidates: List[Dict[str, Any]]) -> Optional[str]:
    """Ask the LLM to pick one candidate id. Returns folder id or None."""
    if not (_HAS_NG and candidates):
        return None

    options = [
        {
            "id": e["id"],
            "description": e["description"],
            "gender": e.get("gender", ""),
        }
        for e in candidates
    ]

    prompt = (
        "You are casting a character voice from a local folder of reference voices. "
        "Each option's folder name is the voice description. Pick exactly ONE unused "
        "voice id that best matches the character's gender, age, role, personality, "
        "and likely vocal tone. Do not pick randomly; prefer a good archetype match. "
        "Return ONLY JSON like {\"id\":\"Folder_Name\", \"reason\":\"brief reason\"}.\n\n"
        f"CHARACTER:\n{_character_voice_match_text(ch)}\n\n"
        f"AVAILABLE VOICES:\n{json.dumps(options, ensure_ascii=False, indent=2)}"
    )
    try:
        raw = ng.get_openai_prompt_response(
            prompt,
            temperature=0.2,
            openai_model=getattr(ng, "openai_model_large", None),
            use_grok=getattr(ng, "USE_GROK", True),
        )
        data = ng.parse_json_response(raw)
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            pick = str(data.get("id", "")).strip()
            if any(e["id"] == pick for e in candidates):
                return pick
    except Exception as e:
        logger.debug("  AI folder voice pick failed for %s (%s)", getattr(ch, "name", ch), e)
    return None


def _pick_folder_voice(ch: "Character", candidates: List[Dict[str, Any]],
                       cfg: "TTSConfig") -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    pick_id = _ai_pick_folder_voice(ch, candidates) if cfg.use_ai_folder_voice_matching else None
    if pick_id:
        for e in candidates:
            if e["id"] == pick_id:
                return e

    # Fallback: deterministic score, stable tie-break by folder name.
    return sorted(
        candidates,
        key=lambda e: (_score_folder_voice_entry(e, ch), e.get("id", "")),
        reverse=True,
    )[0]


class _NarratorVoiceMatchProxy:
    name = "NARRATOR"
    gender = ""
    role = "off-screen narrator"
    archetype = "narrator announcer book reader"
    age = ""
    physical_build = ""
    appearance = ""
    personality = "clear, steady, cinematic narrator"
    voice_profile = "narrator, announcer, book reader, clear storytelling voice"
    speech_pattern = "measured narration"
    cadence = "clear and steady"
    dialect = ""
    vocabulary_level = ""
    wound = ""
    longing = ""


def _descriptor_for(char: "Character") -> str:
    """Pull a compact voice descriptor from the Character's rich voice fields."""
    bits = []
    for attr in ("gender", "physical_build", "cadence", "vocabulary_level",
                 "dialect", "voice_profile", "speech_pattern"):
        v = getattr(char, attr, "") or ""
        if v:
            bits.append(str(v))
    return " | ".join(bits)


def _score_bank_entry(entry: Dict[str, Any], char: "Character") -> int:
    desc = _descriptor_for(char).lower()
    score = 0
    g = (getattr(char, "gender", "") or "").lower()
    if g and entry.get("gender"):
        if entry["gender"] in g or g in entry["gender"]:
            score += 5
        else:
            score -= 4
    for tag in entry.get("tags", []):
        if tag and len(tag) > 2 and tag in desc:
            score += 1
    return score


def cast_voices(characters: List["Character"], cfg: TTSConfig,
                workdir: Optional[Path] = None) -> Dict[str, VoiceProfile]:
    """Assign every character (plus NARRATOR) a VoiceProfile.

    Modes (cfg.voice_mode):
      • random : a DISTINCT random voice per character where possible. A
                 supplied clip in cfg.character_refs always wins; otherwise
                 pick a distinct bank clip at random. With no bank, the
                 character is marked for BASELINE generation (Phase 2 will
                 synthesize a short reference clip via the engine itself and
                 clone from it for every line — see voice_baseline_template),
                 so the voice stays consistent even without a human reference.
      • match  : auto-cast from the bank by gender/descriptor (closest timbre).
      • folder_match : scan cfg.character_voice_dir; each subfolder name is a
                 voice description and its first audio file is the reference.
                 Pick the closest unused folder voice per character.
      • clone  : require a supplied ref per character; warn on any that's missing.

    Every assignment is written to the plan, so you can override any voice by
    editing its ``ref_wav`` (point it at your own clip → it becomes a clone),
    or its ``baseline_text``/``regenerate_baseline`` to redo the auto baseline.
    """
    import random as _r
    mode = (cfg.voice_mode or "random").lower()
    bank = _load_voice_bank(cfg.voice_bank_dir)
    folder_bank = _load_character_voice_folders(cfg) if mode == "folder_match" else []
    used_bank: set = set()
    profiles: Dict[str, VoiceProfile] = {}
    n_baseline = 0

    def random_bank_pick(seed: int):
        avail = [e for e in bank if e["path"] not in used_bank]
        if not avail:
            return None
        e = avail[_r.Random(seed).randrange(len(avail))]
        used_bank.add(e["path"])
        return e

    def best_bank_match(ch):
        avail = sorted((e for e in bank if e["path"] not in used_bank),
                       key=lambda e: _score_bank_entry(e, ch), reverse=True)
        if not avail:
            return None
        used_bank.add(avail[0]["path"])
        return avail[0]

    def best_folder_voice_match(ch):
        avail = [e for e in folder_bank if e["path"] not in used_bank]
        picked = _pick_folder_voice(ch, avail, cfg)
        if picked:
            used_bank.add(picked["path"])
        return picked

    for ch in characters:
        name = getattr(ch, "name", str(ch))
        seed = abs(hash(("voice", name))) % (2**31)
        descriptor = _descriptor_for(ch)
        gender = getattr(ch, "gender", "") or ""
        params: Dict[str, Any] = {"exaggeration_bias": ((seed % 7) - 3) * 0.03,
                                  "seed": seed}
        ref = cfg.character_refs.get(name)
        source = "engine_default"

        if ref:
            source = "clone"
        elif mode == "clone":
            logger.warning("  voice(clone): no ref for %s — will use a generated baseline.", name)
        elif mode == "folder_match":
            picked = best_folder_voice_match(ch)
            if picked:
                ref, source = picked["path"], "folder_match"
                params["voice_folder"] = picked.get("folder", "")
                params["voice_folder_id"] = picked.get("id", "")
                params["voice_description"] = picked.get("description", "")
                params["voice_file"] = picked.get("file", "")
            else:
                logger.warning("  voice(folder_match): no usable folder voice for %s — will use generated baseline.", name)
        elif mode == "match" and bank:
            picked = best_bank_match(ch)
            if picked:
                ref, source = picked["path"], "bank"
        else:  # random (default)
            picked = random_bank_pick(seed) if bank else None
            if picked:
                ref, source = picked["path"], "bank"

        baseline_text = ""
        if not ref:
            # No real reference: Phase 2 will generate one short baseline clip
            # for this character and clone from it for every line, so the
            # voice stays fixed across the whole film instead of drifting
            # call-to-call.
            source = "baseline"
            baseline_text = cfg.voice_baseline_template.format(name=name)
            n_baseline += 1

        profiles[name] = VoiceProfile(name=name, ref_wav=ref, gender=gender,
                                      descriptor=descriptor, source=source, seed=seed,
                                      baseline_text=baseline_text, params=params)
        voice_desc = params.get("voice_description", "")
        logger.info("  voice: %-22s [%s] %s%s", name, source,
                    Path(ref).name if ref else "(baseline pending)",
                    f" — {voice_desc}" if voice_desc else "")
    if n_baseline:
        logger.info("  %d character(s) will get a generated baseline voice "
                    "in Phase 2 (no folder/bank/clone ref supplied).", n_baseline)

    # Narrator — own distinct voice.
    nseed = abs(hash(("voice", "NARRATOR"))) % (2**31)
    nref, nsource = cfg.narrator_ref, ("clone" if cfg.narrator_ref else "engine_default")
    nparams: Dict[str, Any] = {"seed": nseed}
    if not nref and mode == "folder_match":
        picked = best_folder_voice_match(_NarratorVoiceMatchProxy())
        if picked:
            nref, nsource = picked["path"], "folder_match"
            nparams["voice_folder"] = picked.get("folder", "")
            nparams["voice_folder_id"] = picked.get("id", "")
            nparams["voice_description"] = picked.get("description", "")
            nparams["voice_file"] = picked.get("file", "")
    if not nref and bank:
        picked = random_bank_pick(nseed)
        if picked:
            nref, nsource = picked["path"], "bank"
    nbaseline = ""
    if not nref:
        nsource = "baseline"
        nbaseline = cfg.voice_baseline_template.format(name="the narrator")
    profiles["NARRATOR"] = VoiceProfile(name="NARRATOR", ref_wav=nref,
                                        descriptor="measured, neutral narrator",
                                        source=nsource, seed=nseed,
                                        baseline_text=nbaseline, params=nparams)
    return profiles


# =============================================================================
# STAGE 1 · STORY  →  SHOTS
# =============================================================================

def _normalise_dialogue(raw: Any) -> List[Line]:
    out: List[Line] = []
    for d in (raw or []):
        if not isinstance(d, dict):
            if str(d).strip():
                out.append(Line(speaker="", text=str(d).strip()))
            continue
        text = (d.get("text") or d.get("line") or d.get("content") or "").strip()
        if not text:
            continue
        spk = (d.get("speaker") or "").strip() or "NARRATOR"
        emo = _emotion_from_bubble(d.get("bubble_type", "speech"),
                                   d.get("emphasis", "normal"))
        out.append(Line(speaker=spk, text=text, emotion=emo))
    return out


def shots_from_comic_script(script: List[Dict]) -> List[Shot]:
    """Flatten the comic page/panel script into a flat ordered list of Shots.

    Consumes both the legacy shape (pages→panels with a ``description`` field)
    and the current comic_book_generator manifest shape (pages→panels with
    separate ``prompt`` / ``action_sequence`` fields).

    Field mapping from CBG manifest:
      prompt           → shot.image_prompt  (static-image description, no motion)
      action_sequence  → shot.action_sequence (motion-only brief for animation)
      description      → shot.description   (raw panel description, kept for fallbacks)

    When ``prompt`` is present it is used as the pre-built image prompt so
    generate_image_prompts() can skip re-generating it.  When ``action_sequence``
    is present it seeds generate_motion_prompts() so the LLM motion-extraction
    pass is bypassed for that shot (CBG already did the work).
    """
    shots: List[Shot] = []
    i = 0
    for page in script:
        for panel in (page.get("panels") or []):
            # CBG writes the cleaned static prompt into 'prompt'; the raw
            # panel description (which still contains motion language before
            # _extract_action_sequence strips it) lives in 'description'.
            cbg_prompt = str(panel.get("prompt", "") or "").strip()
            cbg_action_seq = panel.get("action_sequence")   # None if absent (old format)
            if cbg_action_seq is not None:
                cbg_action_seq = str(cbg_action_seq).strip()

            shots.append(Shot(
                index=i,
                description=str(panel.get("description", "") or ""),
                setting=str(panel.get("setting", "") or ""),
                mood=str(panel.get("mood", "") or ""),
                composition=str(panel.get("composition", "medium_shot") or "medium_shot"),
                characters_in_frame=list(panel.get("characters_in_frame", []) or []),
                lines=_normalise_dialogue(panel.get("dialogue")),
                # Use CBG's static image prompt directly when available so
                # generate_image_prompts() skips re-generating it.
                image_prompt=cbg_prompt or None,
                # Store the pre-extracted motion brief so generate_motion_prompts()
                # can use it instead of re-extracting from description.
                action_sequence=cbg_action_seq,
            ))
            i += 1
    return shots


def shots_from_user_script(script: List[Dict], default_character: str) -> List[Shot]:
    """Build Shots from a hand-written script.

    Each entry may carry:  speaker, text, action/description, setting, mood,
    composition, characters (list), emotion/bubble_type, engine (per-shot tag).
    Consecutive lines with the same `scene` id (or no scene change) can be
    grouped; by default each entry is its own shot.
    """
    shots: List[Shot] = []
    for i, e in enumerate(script):
        spk = (e.get("speaker") or default_character or "").strip()
        text = (e.get("text") or "").strip()
        emo = e.get("emotion") or _emotion_from_bubble(
            e.get("bubble_type", "speech"), e.get("emphasis", "normal"))
        lines = [Line(speaker=spk, text=text, emotion=emo)] if text else []
        sh = Shot(
            index=i,
            description=str(e.get("action") or e.get("description") or "").strip(),
            setting=str(e.get("setting", "")).strip(),
            mood=str(e.get("mood", "")).strip(),
            composition=str(e.get("composition", "medium_shot")).strip() or "medium_shot",
            characters_in_frame=list(e.get("characters") or ([spk] if spk else [])),
            lines=lines,
        )
        if e.get("engine"):
            sh.engine = str(e["engine"]).lower()
        shots.append(sh)
    return shots


def enrich_dialogue_for_voice(shots: List[Shot],
                              characters: List["Character"]) -> None:
    """Rewrite each spoken line for the *ear*: natural rhythm, breath, subtext.

    Uses the character's voice_guide so each one keeps a distinct cadence,
    dialect, and vocabulary register. NARRATOR captions are smoothed into
    spoken narration. Runs in-place; silently leaves a line untouched on any
    LLM/parse failure so the pipeline never blocks on this nicety.
    """
    if not _HAS_NG:
        logger.info("  (dialogue enrichment skipped — novel_generator unavailable)")
        return
    guides = {}
    for ch in characters:
        try:
            guides[getattr(ch, "name", "")] = ch.build_voice_guide()
        except Exception:
            guides[getattr(ch, "name", "")] = ""

    for sh in shots:
        for ln in sh.lines:
            if not ln.text.strip():
                continue
            guide = guides.get(ln.speaker, "")
            is_narr = ln.speaker.upper() == "NARRATOR"
            prompt = (
                "Rewrite this line so it sounds natural spoken aloud in a film. "
                "Keep it the same meaning and roughly the same length; add only "
                "natural spoken rhythm (contractions, light hesitations where it "
                "fits the emotion). Do NOT add stage directions or quotation "
                "marks. Return ONLY the rewritten line.\n\n"
                f"{'NARRATION (spoken by an off-screen narrator).' if is_narr else guide}\n"
                f"Emotional delivery: {ln.emotion}.\n"
                f"Line: {ln.text}"
            )
            try:
                rewritten = ng.get_openai_prompt_response(
                    prompt, temperature=0.7, use_grok=getattr(ng, "USE_GROK", True)
                )
                rewritten = (rewritten or "").strip().strip('"').strip()
                if rewritten:
                    ln.text = rewritten
            except Exception as e:
                logger.debug("  enrich skipped for one line (%s)", e)


# =============================================================================
# EMOTIONAL-INTELLIGENCE DIALOGUE ENGINE
# =============================================================================
# Rewrites whole multi-character exchanges using the CharacterGraph that
# synthesize_comic_story() already produces — the part that carries the
# psychology (wounds, longings, defenses, voice signatures) AND the relationship
# edges between characters (trust/fear/resentment, unspoken truths, theory-of-
# mind beliefs, power, feared/hoped trajectories). Feeding that into the rewrite
# is what turns "people taking turns saying lines" into a scene with subtext:
# what they want vs. what they admit, what they hide from each other, the line
# that lands because of history the audience can feel but isn't told.
#
# Mirrors comic_book_generator._build_character_voice_guide for per-character
# voice, then adds the relational layer and rewrites scene-by-scene so lines
# answer each other instead of being enriched in isolation.
# =============================================================================

def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


class EIDialogueEngine:
    def __init__(self, graph, characters: List["Character"]):
        self.graph = graph
        self.characters = characters or []
        self.char_by_name = _char_lookup(characters)

    # ---- per-character voice (EI node + concrete speech mechanics) -----------
    def voice_guide(self, name: str) -> str:
        node = None
        if self.graph is not None:
            node = self.graph.get_node(name)
            if node is None:
                for nm, nd in self.graph.nodes.items():
                    if nm.lower() in name.lower() or name.lower() in nm.lower():
                        node = nd
                        break
        parts = [f"{name}:"]
        if node is not None:
            if node.archetype:
                parts.append(f"  archetype={node.archetype}")
            if node.core_wound:
                parts.append(f"  wound={node.core_wound[:120]}")
            if node.core_longing:
                parts.append(f"  longing={node.core_longing[:120]}")
            if node.defense_mechanism:
                parts.append(f"  defense={node.defense_mechanism[:90]}")
            if node.shadow:
                parts.append(f"  shadow={node.shadow[:90]}")
            if node.voice_signature:
                parts.append(f"  voice={node.voice_signature[:160]}")
            if node.lexical_habits:
                parts.append(f"  habits={', '.join(node.lexical_habits[:4])}")
            if getattr(node, "rhythm", ""):
                parts.append(f"  rhythm={node.rhythm}")
            if getattr(node, "metaphor_pool", None):
                parts.append(f"  imagery from={', '.join(node.metaphor_pool[:3])}")
        ch = self.char_by_name.get(name) or self.char_by_name.get(name.split()[0] if name else "")
        if ch is not None:
            mech = []
            for label, attr in (("dialect", "dialect"), ("cadence", "cadence"),
                                ("vocab", "vocabulary_level"), ("speech", "speech_pattern"),
                                ("humor", "humor_style")):
                v = getattr(ch, attr, "")
                if v:
                    mech.append(f"{label}: {str(v)[:80]}")
            if getattr(ch, "verbal_tics", None):
                mech.append(f"tics: {', '.join(ch.verbal_tics[:3])}")
            if getattr(ch, "catchphrases", None):
                mech.append(f"catchphrases: {'; '.join(ch.catchphrases[:2])}")
            if mech:
                parts.append("  SPEECH MECHANICS (obey these): " + " | ".join(mech))
        return "\n".join(parts)

    # ---- relationship subtext between two speakers ---------------------------
    def relationship_block(self, a: str, b: str) -> str:
        if self.graph is None:
            return ""
        e = self.graph.get_edge(a, b) or self.graph.get_edge(b, a)
        if e is None:
            return ""
        dims = [("trust", e.trust), ("affection", e.affection), ("respect", e.respect),
                ("attraction", e.attraction), ("fear", e.fear), ("resentment", e.resentment),
                ("envy", e.envy), ("empathy", e.empathy)]
        hot = sorted(dims, key=lambda kv: abs(_f(kv[1])), reverse=True)
        chosen = [f"{k} {_f(v):+.0f}" for k, v in hot if abs(_f(v)) >= 1.5][:4]
        out = [f"{a} → {b}: " + (", ".join(chosen) if chosen else "neutral")]
        if getattr(e, "power_type", "") or _f(getattr(e, "perceived_power", 0)):
            out.append(f"    power: {e.power_type or '—'} ({_f(e.perceived_power):+.0f})")
        if e.unspoken_truths:
            out.append(f"    unspoken: {'; '.join(e.unspoken_truths[:2])}")
        if e.grievances:
            out.append(f"    grievances: {'; '.join(e.grievances[:2])}")
        if getattr(e, "secrets_kept_from", None):
            out.append(f"    secrets kept: {'; '.join(e.secrets_kept_from[:2])}")
        # theory of mind — what A *believes* B feels (often wrong → dramatic irony)
        tom = []
        for k in ("trust", "affection", "respect", "fear"):
            tv = _f(getattr(e, f"tom_believes_{k}", 0))
            if abs(tv) >= 1.5:
                tom.append(f"{k} {tv:+.0f}")
        if tom:
            out.append(f"    {a} BELIEVES {b} feels: " + ", ".join(tom))
        if getattr(e, "feared_trajectory", ""):
            out.append(f"    {a} fears it becomes: {e.feared_trajectory[:80]}")
        if getattr(e, "hoped_trajectory", ""):
            out.append(f"    {a} hopes it becomes: {e.hoped_trajectory[:80]}")
        return "\n".join(out)

    # ---- group consecutive shots into scenes (by setting) --------------------
    @staticmethod
    def _scenes(shots: List[Shot]) -> List[List[Shot]]:
        scenes, cur, last = [], [], None
        for sh in shots:
            key = (sh.setting or "").strip().lower()
            if cur and key != last:
                scenes.append(cur)
                cur = []
            cur.append(sh)
            last = key
        if cur:
            scenes.append(cur)
        return scenes

    # ---- the rewrite ---------------------------------------------------------
    def enrich(self, shots: List[Shot], theme: str = "", max_lines_per_call: int = 12):
        if not _HAS_NG:
            logger.info("  (EI enrichment skipped — novel_generator unavailable)")
            return
        scenes = self._scenes(shots)
        logger.info("[EI] rewriting dialogue across %d scene(s)…", len(scenes))
        for scene in scenes:
            # flatten the scene's lines into an indexed, addressable list
            flat: List[Tuple[Shot, int, Line]] = []
            for sh in scene:
                for li, ln in enumerate(sh.lines):
                    if ln.text.strip():
                        flat.append((sh, li, ln))
            if not flat:
                continue
            speakers = [ln.speaker for _, _, ln in flat
                        if ln.speaker.upper() != "NARRATOR"]
            distinct = list(dict.fromkeys(speakers))
            # build context once per scene
            guides = "\n".join(self.voice_guide(s) for s in distinct) or "(no profiles)"
            rels = []
            for i in range(len(distinct)):
                for j in range(len(distinct)):
                    if i == j:
                        continue
                    blk = self.relationship_block(distinct[i], distinct[j])
                    if blk:
                        rels.append(blk)
            rel_text = "\n".join(rels) if rels else "(no relationship data)"
            sc = scene[0]
            scene_ctx = (f"Setting: {sc.setting or '—'} | Mood: {sc.mood or '—'} | "
                         f"Theme: {theme or '—'}")

            # rewrite in batches so very long scenes don't blow the token budget
            for start in range(0, len(flat), max_lines_per_call):
                batch = flat[start:start + max_lines_per_call]
                draft = "\n".join(
                    f"[{k}] {ln.speaker}{' (NARRATION)' if ln.speaker.upper()=='NARRATOR' else ''}"
                    f" <{ln.emotion}>: {ln.text}"
                    for k, (_, _, ln) in enumerate(batch))
                prompt = (
                    "You are a master screenwriter polishing a scene's spoken dialogue. "
                    "Rewrite EACH numbered line so the exchange crackles with subtext and "
                    "every character sounds unmistakably like themselves. Use the psychology "
                    "and the relationships below to decide what each line really means — what "
                    "the speaker wants, hides, or fears. Keep each line's intent and roughly "
                    "its length; this is spoken aloud, so favour natural rhythm and "
                    "contractions. Do NOT add stage directions or quotation marks. NARRATION "
                    "lines stay as narration (no relational subtext).\n\n"
                    f"SCENE: {scene_ctx}\n\n"
                    f"CHARACTER VOICES:\n{guides}\n\n"
                    f"RELATIONSHIPS (subtext to play under the words):\n{rel_text}\n\n"
                    f"DRAFT LINES:\n{draft}\n\n"
                    "Return ONLY a JSON array, one object per line, no prose:\n"
                    '[{"i": 0, "text": "<rewritten>", "emotion": "<neutral|emphatic|excited|'
                    'angry|scared|tender|cold|sarcastic|whisper|intimate|narration>"}]'
                )
                try:
                    raw = ng.get_openai_prompt_response(
                        prompt, temperature=0.8,
                        openai_model=getattr(ng, "openai_model_large", None),
                        use_grok=getattr(ng, "USE_GROK", True))
                    data = ng.parse_json_response(raw)
                    if isinstance(data, dict):
                        data = data.get("lines") or data.get("dialogue") or []
                    for obj in (data or []):
                        idx = int(obj.get("i", -1))
                        if 0 <= idx < len(batch):
                            _, _, ln = batch[idx]
                            new_t = (obj.get("text") or "").strip().strip('"')
                            if new_t:
                                ln.text = new_t
                            if obj.get("emotion"):
                                ln.emotion = str(obj["emotion"]).strip().lower()
                except Exception as e:
                    logger.debug("  EI rewrite batch skipped (%s)", e)
        logger.info("[EI] dialogue rewrite complete.")


def balance_scene_dialogue(shots: List[Shot], characters: List["Character"],
                           theme: str, cfg: "ProjectConfig") -> None:
    """Add reactive lines for present-but-silent characters in lopsided scenes.

    A scene with 2+ characters in frame but speech concentrated in one of
    them reads as a monologue with an audience. The upstream script generator
    decides the initial cast distribution and EIDialogueEngine.enrich() only
    rewrites EXISTING lines' text — neither can fix who actually speaks. This
    pass can: for each scene, it finds characters that are present
    (characters_in_frame) but never speak, and asks the LLM for a few short
    reactive lines — placed only into shots that ALREADY frame that character
    and currently have NO line, never appended alongside an existing line in
    the same shot (which would put two different voices in one audio track
    feeding a single lip-synced face). Runs BEFORE EIDialogueEngine.enrich()
    so the new lines get the same psychological polish as everything else.
    """
    if not cfg.balance_scene_dialogue or not _HAS_NG:
        return
    scenes = EIDialogueEngine._scenes(shots)
    n_added = 0
    for scene in scenes:
        present: List[str] = []
        for sh in scene:
            for nm in sh.characters_in_frame:
                if nm not in present:
                    present.append(nm)
        if len(present) < 2:
            continue
        speakers = [ln.speaker for sh in scene for ln in sh.lines
                   if ln.text.strip() and ln.speaker.upper() != "NARRATOR"]
        distinct_speaking = list(dict.fromkeys(speakers))
        silent_present = [p for p in present if p not in distinct_speaking]
        if not silent_present:
            continue                            # everyone present already gets to speak
        # Only shots that already frame a silent character AND have no line
        # yet — never double up a shot with two different speakers' audio.
        candidates = [(sh, p) for sh in scene for p in silent_present
                     if p in sh.characters_in_frame
                     and not any(l.text.strip() for l in sh.lines)]
        if not candidates:
            continue                            # no clean reaction shot available — skip, don't force it

        sc = scene[0]
        beats = "\n".join(
            f"[{i}] shot {sh.index} ({sh.composition}): {(sh.description or '')[:140]}"
            for i, (sh, p) in enumerate(candidates[:8]))
        names_silent = ", ".join(dict.fromkeys(p for _, p in candidates))
        prompt = (
            "This scene has multiple characters on screen, but dialogue is "
            f"concentrated in too few of them. These characters are PRESENT "
            f"but currently SILENT: {names_silent}. Suggest up to "
            f"{cfg.balance_max_new_lines_per_scene} short reactive lines (a "
            "few words to one sentence — an interjection, a reaction, a beat "
            "of pushback or agreement) for the silent characters, each placed "
            "at one of the numbered beats below, so the scene reads as a real "
            "exchange instead of a monologue with an audience. Don't "
            "summarize plot — react to the moment.\n\n"
            f"SCENE: {sc.setting or '—'} | mood: {sc.mood or '—'} | theme: {theme or '—'}\n\n"
            f"BEATS:\n{beats}\n\n"
            "Return ONLY JSON: "
            '[{"beat": 0, "speaker": "<name from the silent list>", '
            '"text": "<short reactive line>", "emotion": "<neutral|emphatic|'
            'excited|angry|scared|tender|cold|sarcastic|whisper|intimate>"}]'
        )
        try:
            raw = ng.get_openai_prompt_response(
                prompt, temperature=0.8,
                openai_model=getattr(ng, "openai_model_large", None),
                use_grok=getattr(ng, "USE_GROK", True))
            data = ng.parse_json_response(raw)
            if isinstance(data, dict):
                data = data.get("lines") or []
            for obj in (data or [])[:cfg.balance_max_new_lines_per_scene]:
                bi = int(obj.get("beat", -1))
                if not (0 <= bi < len(candidates)):
                    continue
                sh, speaker = candidates[bi]
                if any(l.text.strip() for l in sh.lines):
                    continue                    # got a line from elsewhere in this batch; skip
                text = (obj.get("text") or "").strip().strip('"')
                if not text or speaker not in sh.characters_in_frame:
                    continue
                sh.lines.append(Line(speaker=speaker, text=text,
                                     emotion=str(obj.get("emotion") or "neutral").lower()))
                n_added += 1
        except Exception as e:
            logger.debug("  scene-balance pass skipped for a scene (%s)", e)
    if n_added:
        logger.info("[DIALOGUE] added %d reactive line(s) so multi-character "
                    "scenes read as real exchanges.", n_added)


_NON_CLOSE_COMPOSITIONS = {"wide_shot", "medium_shot", "over_shoulder"}


def _mergeable(a: Shot, b: Shot) -> bool:
    """Whether shot b can be folded into a's group: same two-or-more people
    on screen, a non-close-up framing, both speaking, and ALTERNATING
    speakers (a genuine back-and-forth, not just adjacent monologue beats)."""
    return (
        a.composition in _NON_CLOSE_COMPOSITIONS and b.composition in _NON_CLOSE_COMPOSITIONS
        and len(a.characters_in_frame) >= 2
        and set(a.characters_in_frame) == set(b.characters_in_frame)
        and a.is_dialogue and b.is_dialogue
        and bool(a.speaking_character) and bool(b.speaking_character)
        and a.speaking_character != b.speaking_character
    )


def merge_dialogue_shots(shots: List[Shot], cfg: "ProjectConfig") -> List[Shot]:
    """Combine short, alternating exchanges into ONE shot covering both
    speakers, instead of hard-cutting to a new shot on every single line.

    Real conversations are often held in one two-shot/wide rather than cut on
    every line. This finds RUNS of consecutive same-scene dialogue shots that
    already frame the exact same 2+ people in a non-close-up composition and
    alternate speakers, and merges up to merge_max_lines_per_shot of them
    into the first shot's slot — concatenating their lines in order (the
    existing per-shot audio assembly already handles multiple lines/speakers
    per shot, so this needs no changes downstream). The merged shot is tagged
    engine="wan_i2v" explicitly, since its audio now carries more than one
    voice and Wan-S2V lip-syncs to a single face. Close-ups are NEVER
    touched — those stay one-speaker-per-shot so S2V can lip-sync them
    precisely. Shots are renumbered after merging.
    """
    if not cfg.merge_dialogue_shots or len(shots) < 2:
        return shots
    scenes = EIDialogueEngine._scenes(shots)
    out: List[Shot] = []
    n_groups = 0
    for scene in scenes:
        i = 0
        while i < len(scene):
            run = [scene[i]]
            while (len(run) < max(2, cfg.merge_max_lines_per_shot)
                  and i + len(run) < len(scene)
                  and _mergeable(run[-1], scene[i + len(run)])):
                run.append(scene[i + len(run)])
            if len(run) >= 2:
                merged = run[0]
                for extra in run[1:]:
                    merged.lines.extend(extra.lines)
                merged.engine = "wan_i2v"
                out.append(merged)
                n_groups += 1
                i += len(run)
            else:
                out.append(run[0])
                i += 1
    for idx, sh in enumerate(out):
        sh.index = idx
    if n_groups:
        logger.info("[DIRECT] merged %d short exchange(s) into shared two-shots "
                    "instead of cutting on every line.", n_groups)
    return out


# =============================================================================
# VOICE CUE TOKENS  ·  Higgs Audio v3's inline emotion/style/sfx/prosody tags
# =============================================================================
# These are read directly out of the spoken text by the model itself — they
# have to live IN the string passed to synth(), not just as metadata on the
# Line object (which is all the existing `emotion` field gives the engine).

_VOICE_CUE_ENGINES = {"higgs"}        # engines that actually understand this token vocabulary

_EMOTION_TOKENS = [
    "elation", "amusement", "enthusiasm", "determination", "pride",
    "contentment", "affection", "relief", "contemplation", "confusion",
    "surprise", "awe", "longing", "arousal", "anger", "fear", "disgust",
    "bitterness", "sadness", "shame", "helplessness",
]
_STYLE_TOKENS = ["singing", "shouting", "whispering"]
# sfx token -> suggested onomatopoeia (must be paired immediately after the token)
_SFX_TOKENS = {
    "cough": "Ahem", "laughter": "Haha", "crying": "Boohoo", "screaming": "Ahh",
    "burping": "Burp", "humming": "Hmm", "sigh": "Uh", "sniff": "Sff", "sneeze": "Achoo",
}
_PROSODY_TOKENS = [
    "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast",
    "pitch_low", "pitch_high", "pause", "long_pause",
    "expressive_high", "expressive_low",
]

# every literal token string this vocabulary defines, for validating LLM output
_ALL_VOICE_CUE_TOKENS = (
    {f"<|emotion:{e}|>" for e in _EMOTION_TOKENS}
    | {f"<|style:{s}|>" for s in _STYLE_TOKENS}
    | {f"<|sfx:{s}|>" for s in _SFX_TOKENS}
    | {f"<|prosody:{p}|>" for p in _PROSODY_TOKENS})

_VOICE_CUE_TOKEN_RE = re.compile(r"<\|[a-z]+:[a-zA-Z_]+\|>")

_VOICE_CUE_REFERENCE = """\
PLACEMENT (important):
- SENTENCE-LEVEL tags go at the START of a sentence and color that whole
  sentence only: emotion, style, and prosody speed_*/pitch_*/expressive_*.
  To keep a multi-sentence line in one feeling, repeat the tag at the start of
  each sentence it should cover.
- INLINE tags go at the exact spot they occur: sfx, and prosody pause/long_pause.
- An sfx tag is written tag-then-onomatopoeia with NO space, then the words,
  e.g. "<|sfx:laughter|>Haha, so glad you made it." You may stack a leading
  emotion with an inline sfx: "<|emotion:elation|><|sfx:laughter|>Haha, welcome!"
- speed_very_slow only slows to ~5s; for a slower, more deliberate beat insert
  <|prosody:long_pause|> between phrases instead.

EMOTION (<|emotion:X|>, sentence-level): affection, amusement, anger, arousal,
  awe, bitterness, confusion, contemplation, contentment, determination, disgust,
  elation, enthusiasm, fear, helplessness, longing, pride, relief, sadness,
  shame, surprise
STYLE (<|style:X|>, sentence-level): singing, shouting, whispering
SFX (<|sfx:X|>, inline — MUST be immediately followed by its onomatopoeia):
  cough->Ahem, laughter->Haha, crying->Boohoo, screaming->Ahh, burping->Burp,
  humming->Hmm, sigh->Uh, sniff->Sff, sneeze->Achoo
PROSODY sentence-level (<|prosody:X|>): speed_very_slow, speed_slow, speed_fast,
  speed_very_fast, pitch_low, pitch_high, expressive_high, expressive_low
PROSODY inline (<|prosody:X|>): pause, long_pause"""


def _strip_invalid_voice_cue_tokens(text: str) -> str:
    """Drop any <|...|>-shaped token the LLM emitted that isn't in our known
    vocabulary (a typo/hallucination), leaving the rest of the text intact."""
    def _check(m):
        return m.group(0) if m.group(0) in _ALL_VOICE_CUE_TOKENS else ""
    return _VOICE_CUE_TOKEN_RE.sub(_check, text)


def _voice_cue_edit_is_sane(original: str, edited: str) -> bool:
    """Guard against the LLM rewriting content instead of just inserting
    tokens: the spoken WORDS (tokens stripped out) must still be mostly the
    same text, not a paraphrase or new line."""
    strip_tokens = lambda s: _VOICE_CUE_TOKEN_RE.sub(" ", s)
    orig_words = strip_tokens(original).lower().split()
    edit_words = strip_tokens(edited).lower().split()
    if not orig_words:
        return True
    overlap = len(set(orig_words) & set(edit_words))
    return overlap / max(1, len(set(orig_words))) >= 0.6


_HIGGS_EMOTION_ALIASES = {
    "joy": "elation", "happy": "elation", "happiness": "elation", "elated": "elation",
    "amused": "amusement", "playful": "amusement", "sarcastic": "bitterness",
    "excited": "enthusiasm", "excitement": "enthusiasm", "eager": "enthusiasm",
    "emphatic": "determination", "firm": "determination", "resolved": "determination",
    "determined": "determination", "confident": "pride", "proud": "pride",
    "calm": "contentment", "neutral": "contentment", "peaceful": "contentment",
    "tender": "affection", "warm": "affection", "loving": "affection", "intimate": "affection",
    "relieved": "relief", "thoughtful": "contemplation", "reflective": "contemplation",
    "narration": "contemplation", "narrator": "contemplation", "confused": "confusion",
    "shocked": "surprise", "surprised": "surprise", "wonder": "awe", "wondering": "awe",
    "yearning": "longing", "romantic": "longing", "desire": "arousal",
    "sensual": "arousal", "angry": "anger", "mad": "anger", "furious": "anger",
    "scared": "fear", "afraid": "fear", "terrified": "fear", "disgusted": "disgust",
    "resentful": "bitterness", "bitter": "bitterness", "sad": "sadness",
    "grief": "sadness", "grieving": "sadness", "ashamed": "shame",
    "helpless": "helplessness", "despair": "helplessness",
}


def _has_voice_token(text: str, kind: str) -> bool:
    return re.search(rf"<\|{re.escape(kind)}:[a-zA-Z_]+\|>", text or "") is not None


def _choose_higgs_emotion(line: "Line", shot: Optional["Shot"] = None) -> str:
    """Map internal emotion/mood/text cues to the documented Higgs emotion set."""
    raw = " ".join([
        getattr(line, "emotion", "") or "",
        getattr(shot, "mood", "") if shot is not None else "",
        getattr(line, "text", "") or "",
    ]).lower()

    # Direct documented token names win.
    for emo in _EMOTION_TOKENS:
        if re.search(rf"\b{re.escape(emo)}\b", raw):
            return emo

    # Common project/internal aliases.
    for key, emo in _HIGGS_EMOTION_ALIASES.items():
        if re.search(rf"\b{re.escape(key)}\b", raw):
            return emo

    # Punctuation/context fallback.
    if "!" in (line.text or ""):
        return "enthusiasm"
    if "?" in (line.text or ""):
        return "contemplation"
    return "contentment"


def _choose_higgs_style(line: "Line", shot: Optional["Shot"] = None) -> str:
    raw = " ".join([getattr(line, "emotion", "") or "", getattr(line, "text", "") or ""]).lower()
    if re.search(r"\b(sing|sings|singing|sang|humming|hum)\b", raw):
        return "singing"
    if re.search(r"\b(shout|shouts|shouting|yell|yells|yelling|scream|screams|screaming)\b", raw):
        return "shouting"
    if re.search(r"\b(whisper|whispers|whispering|hushed|under his breath|under her breath)\b", raw):
        return "whispering"
    return ""


def _choose_higgs_prosody(emotion: str, line: "Line") -> str:
    raw = (line.text or "").lower()
    words = (line.text or "").split()
    # A shouted/emphasised word in ALL CAPS (2+ letters) reads as heightened.
    has_caps_emphasis = any(len(w.strip(".,!?;:'\"")) >= 2
                            and w.strip(".,!?;:'\"").isupper() for w in words)
    if re.search(r"\b(wait|stop|run|hurry|quick|now|no!|go!)\b", raw):
        return "speed_fast"
    if has_caps_emphasis or emotion in {"enthusiasm", "surprise", "fear", "anger", "elation"}:
        return "expressive_high"
    # A line that trails off (… / ...) wants a slower, lingering delivery.
    if (line.text or "").rstrip().endswith(("...", "…")):
        return "speed_slow"
    if emotion in {"sadness", "shame", "helplessness", "longing", "contemplation"}:
        return "speed_slow"
    if emotion in {"contentment", "relief"}:
        return "expressive_low"
    return ""


# Punctuation that signals a real dramatic beat mid-line — a caught breath, a
# hesitation, a landing pause. A documented <|prosody:pause|> just before it
# tells Higgs to actually hold there instead of reading straight through.
_DRAMATIC_PAUSE_RE = re.compile(r"(\.\.\.|…|—|--)")


def _insert_dramatic_pauses(text: str, max_pauses: int = 2) -> str:
    """Insert <|prosody:pause|> before em-dashes / ellipses (word-preserving).

    Skips entirely if the text already carries an explicit pause token, so it
    never stacks with the LLM cue pass. Only documented tokens are used.
    """
    if not text or "<|prosody:pause|>" in text or "<|prosody:long_pause|>" in text:
        return text
    count = {"n": 0}

    def repl(m):
        if count["n"] >= max_pauses:
            return m.group(0)
        count["n"] += 1
        return "<|prosody:pause|>" + m.group(0)

    return _DRAMATIC_PAUSE_RE.sub(repl, text)


def _choose_higgs_sfx(line: "Line") -> Tuple[str, str]:
    raw = (line.text or "").lower()
    # Prefer explicit onomatopoeia already present in the line.
    if re.search(r"\b(ha+ha+|he+he+|laughs?|chuckles?)\b", raw):
        return "laughter", _SFX_TOKENS["laughter"]
    if re.search(r"\b(boohoo|sob|sobs|crying|cries)\b", raw):
        return "crying", _SFX_TOKENS["crying"]
    if re.search(r"\b(a+h+|aa+h+|screams?)\b", raw):
        return "screaming", _SFX_TOKENS["screaming"]
    if re.search(r"\b(ahem|coughs?)\b", raw):
        return "cough", _SFX_TOKENS["cough"]
    if re.search(r"\b(hmm+|mmm+|humming|hums?)\b", raw):
        return "humming", _SFX_TOKENS["humming"]
    if re.search(r"\b(sighs?|ahh|uh)\b", raw):
        return "sigh", _SFX_TOKENS["sigh"]
    if re.search(r"\b(sniff|sniffs|sff)\b", raw):
        return "sniff", _SFX_TOKENS["sniff"]
    if re.search(r"\b(achoo|sneeze|sneezes)\b", raw):
        return "sneeze", _SFX_TOKENS["sneeze"]
    if re.search(r"\b(burp|burps)\b", raw):
        return "burping", _SFX_TOKENS["burping"]
    return "", ""


# Where each sfx's onomatopoeia / trigger word tends to appear in a line, so we
# can tag an EXISTING onomatopoeia in place instead of injecting a duplicate
# (e.g. tag the "Haha" the writer already wrote, not add a second one).
_SFX_TRIGGER_PATTERNS = {
    "laughter": r"ha+ha+|he+he+|laughs?|chuckles?",
    "crying":   r"boo+hoo+|sob\w*|cries|crying",
    "screaming": r"a{2,}h+|screams?",
    "cough":    r"ahem|coughs?",
    "humming":  r"h+mm+|hums?|humming",
    "sigh":     r"sighs?|ahh|uh",
    "sniff":    r"sniffs?|sff",
    "sneeze":   r"achoo|sneezes?",
    "burping":  r"burps?",
}


def _insert_sfx_tag(sentence: str, sfx: str, ono: str) -> str:
    """Place an <|sfx|> tag in a sentence without duplicating onomatopoeia.

    If the onomatopoeia (or its trigger word) already appears, the bare tag is
    inserted immediately before it (catalog form: tag then onomatopoeia). If
    not, the tag + onomatopoeia are prepended to the sentence.
    """
    pat = _SFX_TRIGGER_PATTERNS.get(sfx)
    if pat:
        m = re.search(pat, sentence, re.I)
        if m:
            return sentence[:m.start()] + f"<|sfx:{sfx}|>" + sentence[m.start():]
    return f"<|sfx:{sfx}|>{ono} " + sentence


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """Split a line into sentences, keeping end punctuation on each piece."""
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _category_present(sentence: str, category: str) -> bool:
    """True if the sentence already carries any token of this category."""
    return re.search(rf"<\|{re.escape(category)}:[a-zA-Z_]+\|>", sentence) is not None


def _ensure_higgs_voice_cues_for_line(line: "Line", tts: "TTSConfig",
                                      shot: Optional["Shot"] = None) -> bool:
    """Ensure a single line has valid Higgs control tokens for audio synthesis.

    This is deterministic and content-preserving: it only prepends documented
    tokens/onomatopoeia. It never changes the spoken words.
    """
    if not (line.text or "").strip():
        return False
    original = line.text
    text = _strip_invalid_voice_cue_tokens(original).strip()

    # Expressive/dramatic layer: hold on ellipses and em-dashes. Word-preserving
    # and documented-token-only; done before the leading prefix so the prefix
    # stays at the very front of the line.
    if (getattr(tts, "expressive_delivery", True)
            and getattr(tts, "add_style_sfx_prosody_tokens", True)):
        text = _insert_dramatic_pauses(text)

    prefix: List[str] = []
    emotion = _choose_higgs_emotion(line, shot)

    # Per the Higgs tag catalog, emotion / style / speed-pitch-expressive prosody
    # are SENTENCE-LEVEL: each tag only colors the sentence it starts. So to keep
    # a whole multi-sentence line in one feeling, the tag has to lead EVERY
    # sentence, not just the line. sfx and pause/long_pause are inline and stay
    # where they occur (dramatic pauses were already inserted above).
    sentence_tags: List[str] = []
    if getattr(tts, "require_emotion_cue_tokens", True) and emotion:
        sentence_tags.append(f"<|emotion:{emotion}|>")
    sfx = ono = ""
    if getattr(tts, "add_style_sfx_prosody_tokens", True):
        style = _choose_higgs_style(line, shot)
        if style:
            sentence_tags.append(f"<|style:{style}|>")
        prosody = _choose_higgs_prosody(emotion, line)
        if prosody:
            sentence_tags.append(f"<|prosody:{prosody}|>")
        sfx, ono = _choose_higgs_sfx(line)

    sentences = _split_sentences(text) or [text]
    rebuilt: List[str] = []
    for i, s in enumerate(sentences):
        if not s.strip():
            continue
        lead = ""
        for t in sentence_tags:
            category = t[2:t.index(":")]        # emotion | style | prosody
            if not _category_present(s, category):
                lead += t
        # One sound effect, on the first sentence only, placed on an existing
        # onomatopoeia where possible (no duplication), after the sentence-level
        # tags in lead.
        if i == 0 and sfx and not _has_voice_token(text, "sfx"):
            s = _insert_sfx_tag(s, sfx, ono)
        rebuilt.append(lead + s if lead else s)

    text = re.sub(r"\s+", " ", " ".join(rebuilt)).strip()
    line.text = text
    return line.text != original


def ensure_required_voice_cue_tokens(shots: List["Shot"], tts: "TTSConfig") -> int:
    """Final audio-only guardrail: every Higgs line gets a valid emotion cue.

    Runs after the optional LLM cue pass and again immediately before TTS
    synthesis, so edited/older plans are safe. Visual prompts remove these
    tokens before image/video generation.
    """
    if not getattr(tts, "voice_cue_tokens", True):
        return 0
    if (getattr(tts, "engine", "") or "").lower() not in _VOICE_CUE_ENGINES:
        return 0

    n = 0
    for sh in shots:
        for ln in sh.lines:
            if _ensure_higgs_voice_cues_for_line(ln, tts, sh):
                n += 1
    if n:
        logger.info("[DIALOGUE] ensured Higgs voice-cue tokens on %d line(s).", n)
    return n


def add_voice_cue_tokens(shots: List[Shot], tts: TTSConfig, batch_size: int = 15) -> None:
    """Insert Higgs Audio v3's inline emotion/style/sfx/prosody tokens into
    spoken lines for audio synthesis. An optional LLM pass adds nuanced cues;
    a deterministic guardrail then ensures every Higgs line has at least one
    documented <|emotion:...|> token, with obvious style/sfx/prosody cues when
    appropriate. Only runs for engines whose vocabulary this actually is
    (currently Higgs); on any other engine these tokens would just be read
    aloud as literal text.

    Runs LAST among the text-mutating passes (after EI enrich,
    balance_scene_dialogue, merge_dialogue_shots) so it edits final text, not
    something a later rewrite would clobber. Each edit is sanity-checked
    (token-stripped word overlap with the original) before being accepted —
    a low-overlap "edit" looks like a paraphrase, not a token insertion, and
    is rejected, keeping the original line untouched.
    """
    if not tts.voice_cue_tokens or (tts.engine or "").lower() not in _VOICE_CUE_ENGINES:
        return
    if not _HAS_NG:
        ensure_required_voice_cue_tokens(shots, tts)
        return

    # Flatten every line across every shot, remembering where it came from.
    flat: List[Tuple[Shot, Line]] = [
        (sh, ln) for sh in shots for ln in sh.lines if ln.text.strip()]
    if not flat:
        return

    n_tagged = 0
    expressive = getattr(tts, "expressive_delivery", True)
    if expressive:
        guidance = (
            "You add Higgs Audio v3's inline voice-cue tokens to spoken lines "
            "so the TTS engine PERFORMS the emotion and delivery instead of "
            "reading flat text. Aim for an expressive, dramatic performance: "
            "most emotionally-charged lines should carry a cue, and you may "
            "LAYER an <|emotion:...|> with a <|prosody:...|> (and a <|style:...|> "
            "when someone whispers/shouts/sings) where it sharpens the moment. "
            "Write in a real non-verbal beat — a sigh, a caught laugh, a sob, a "
            "gasp of breath — as an <|sfx:...|> where the moment genuinely earns "
            "it. Do NOT tag flat, purely functional lines, and keep it to at "
            "most ~3 DIFFERENT effects on any single sentence so it stays "
            "believable rather than cartoonish. Follow the PLACEMENT rules below "
            "(sentence-level tags lead each sentence they color; sfx and pauses "
            "go inline). Tokens go exactly as spelled below — do not invent new "
            "ones.\n\n")
    else:
        guidance = (
            "You add Higgs Audio v3's inline voice-cue tokens to spoken lines "
            "so the TTS engine actually performs the emotion/delivery, not "
            "just speaks flat text. Use them SPARINGLY — most lines need "
            "zero or one token; only add where it genuinely sharpens the "
            "performance (a real emotional beat, a deliberate pause before "
            "something landing hard, an actual sigh/laugh/cry written into "
            "the moment). Never tag every line. Tokens go INLINE in the text, "
            "exactly as spelled below — do not invent new ones.\n\n")
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        blocks = "\n".join(
            f'[{k}] speaker={ln.speaker!r} emotion={ln.emotion!r}: "{ln.text}"'
            for k, (sh, ln) in enumerate(batch))
        prompt = (
            guidance +
            f"{_VOICE_CUE_REFERENCE}\n\n"
            "Rules: an sfx token MUST be immediately followed by its "
            "onomatopoeia (e.g. '<|sfx:sigh|>Uh, I suppose so.'). Place "
            "emotion/style/prosody tokens at the point in the line where the "
            "shift happens, not always at the very start — a single line can "
            "combine more than one tag where the delivery genuinely shifts "
            "partway through, e.g.:\n"
            '  "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, '
            "that was kind of hilarious. <|sfx:laughter|>Hehe, no, seriously, "
            'I was not ready for that."\n'
            "Do NOT change any of the actual words — only insert tokens.\n\n"
            f"LINES:\n{blocks}\n\n"
            "Return ONLY JSON, and ONLY for lines you actually changed (omit "
            'untouched lines): [{"i": 0, "text": "<|emotion:relief|> Thank '
            'god, you made it."}]')
        try:
            raw = ng.get_openai_prompt_response(
                prompt, temperature=0.5,
                openai_model=getattr(ng, "openai_model_large", None),
                use_grok=getattr(ng, "USE_GROK", True))
            data = ng.parse_json_response(raw)
            if isinstance(data, dict):
                data = data.get("lines") or []
        except Exception as e:
            logger.debug("  voice-cue token batch failed (%s) — leaving as-is.", e)
            continue
        for obj in (data or []):
            if not isinstance(obj, dict):
                continue
            bi = int(obj.get("i", -1))
            if not (0 <= bi < len(batch)):
                continue
            new_text = _strip_invalid_voice_cue_tokens(str(obj.get("text") or ""))
            _, ln = batch[bi]
            if new_text.strip() and _voice_cue_edit_is_sane(ln.text, new_text):
                ln.text = new_text
                n_tagged += 1
    if n_tagged:
        logger.info("[DIALOGUE] added voice-cue tokens to %d line(s) for Higgs.", n_tagged)
    ensure_required_voice_cue_tokens(shots, tts)


def build_ei_graph_from_characters(characters: List["Character"], story_idea) -> Optional[Any]:
    """Synthesize an EI graph for script-mode casts (so they also get subtext).

    Best-effort: uses novel_generator's own node/relationship builders. Returns
    None if the cast is trivial or anything goes wrong (caller falls back).
    """
    if not _HAS_NG or len(characters or []) < 1:
        return None
    try:
        graph = ng.CharacterGraph(story_idea)
        for ch in characters:
            try:
                node = ng.ai_build_character_node(ch, story_idea, characters)
            except Exception:
                node = ng._fallback_node(ch)
            graph.add_node(node)
        if len(characters) > 1:
            try:
                ng.ai_initialize_relationships(graph, story_idea)
            except Exception as e:
                logger.debug("  relationship init skipped (%s)", e)
        return graph
    except Exception as e:
        logger.warning("[EI] graph synthesis failed (%s) — using simple enrichment.", e)
        return None


def generate_voice_baselines(voices: Dict[str, VoiceProfile], engine: "BaseTTS",
                             cfg: TTSConfig, workdir: Path, resume: bool = True) -> None:
    """Generate a short, stable baseline clip for every voice marked source=="baseline".

    Synthesizes ``vp.baseline_text`` (the pangram-rich line naming the
    character) through the SAME engine that will voice the film, with no
    reference — i.e. however that engine's TTS sounds out of the box — then
    points the character's ref_wav at that clip. Every subsequent line for
    that character clones from this one fixed file, so the voice can't drift
    call-to-call the way an un-cloned "engine default" synthesis otherwise
    could. Idempotent (resume reuses an existing baseline clip) and
    per-character re-doable via VoiceProfile.regenerate_baseline.

    Skipped entirely for any voice that already has a real ref (clone/bank) —
    those are already a fixed file, already consistent.
    """
    if not _HAS_SF:
        return                              # synthesize_audio already raised on this
    bdir = workdir / "voice_baselines"
    bdir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for name, vp in voices.items():
        if vp.source != "baseline":
            continue
        text = (vp.baseline_text or "").strip() or cfg.voice_baseline_template.format(name=name)
        out = bdir / f"{_slug(name)}_baseline.wav"
        # If the plan's ref_wav already points somewhere ELSE (the user edited
        # it to their own clip without bothering to also flip `source`),
        # respect that override outright instead of clobbering it.
        if vp.ref_wav and Path(vp.ref_wav).resolve() != out.resolve():
            vp.source = "clone"
            continue
        if resume and not vp.regenerate_baseline and out.exists():
            vp.ref_wav = str(out)
            vp.baseline_path = str(out)
            n_done += 1
            continue
        try:
            engine.register(name, None, vp.params)     # no ref yet → engine's own voice
            wav = engine.synth(text, name, "neutral")
            sf.write(str(out), wav, engine.sr)
        except Exception as e:
            logger.warning(
                "  baseline generation failed for %s (%s) — that character's "
                "lines will fall back to a fresh engine-default synthesis per "
                "line (may vary slightly call-to-call) instead of a fixed clip.",
                name, e)
            continue
        vp.ref_wav = str(out)
        vp.baseline_path = str(out)
        vp.regenerate_baseline = False
        n_done += 1
        logger.info("  baseline voice ✓  %-22s → %s", name, out.name)
    if n_done:
        logger.info("[VOICE] %d baseline voice clip(s) ready (%s).",
                    n_done, bdir)


# =============================================================================
# STAGE 2 · VOICE  (TTS for every shot)
# =============================================================================

def _context_silence_ms(shots: List[Shot], i: int, cfg: TTSConfig) -> Tuple[int, int]:
    """(lead_ms, tail_ms) for shots[i], scaled by what's happening at its edit
    boundaries: a scene/setting change needs more breathing room than a
    same-scene speaker swap, which needs more than a straight continuation
    (which just gets the plain base silence).

    A special case fires when a dialogue shot opens a *new conversation* after a
    scene change — i.e. the previous shot was non-dialogue or a different scene
    entirely (not a continuation of the same exchange).  Viewers need extra time
    to orient to the new location before the first line lands, so
    dialogue_scene_entry_pad_ms is used there instead of the generic
    scene_transition_pad_ms.
    """
    sh = shots[i]
    lead, tail = cfg.lead_silence_ms, cfg.tail_silence_ms

    entry_pad = int(getattr(cfg, "dialogue_scene_entry_pad_ms", 1400))

    # Boundary breathing room (new-conversation scene-entry > scene change >
    # speaker change > plain continuation).
    def _boundary_pad(a: Shot, b: Shot) -> int:
        scene_changed = b.setting and (a.setting or "") != (b.setting or "")
        if scene_changed:
            # Larger pad when b opens a new conversation: b is dialogue and
            # a was either non-dialogue or a different set of characters, so
            # the viewer gets no warm-up from the previous shot's exchange.
            if b.is_dialogue and (
                not a.is_dialogue
                or set(a.characters_in_frame) != set(b.characters_in_frame)
            ):
                return entry_pad
            return cfg.scene_transition_pad_ms
        if (a.speaking_character and b.speaking_character
                and a.speaking_character != b.speaking_character):
            return cfg.speaker_change_pad_ms
        return 0

    lead_pad = _boundary_pad(shots[i - 1], sh) if i > 0 else 0
    tail_pad = _boundary_pad(sh, shots[i + 1]) if i < len(shots) - 1 else 0

    # For audio-driven dialogue shots, don't inflate the HEAD with the boundary
    # pad — that leading silence is exactly what risks an onset offset when the
    # mouth is generated from the waveform. Keep the head minimal and let the
    # previous shot's TAIL carry the pause (trailing silence is safe: the mouth
    # just settles). If the previous shot has no tail of its own, keep the pad on
    # the lead so the cut isn't abrupt.
    driven = bool(getattr(cfg, "trim_lead_for_audio_driven", True)) and sh.is_dialogue
    if driven:
        prev = shots[i - 1] if i > 0 else None
        prev_has_tail = bool(prev and getattr(prev, "audio_path", None)
                             and (getattr(prev, "duration", 0.0) or 0.0) > 0.0)
        lead = min(lead, int(getattr(cfg, "audio_driven_lead_ms", 60)))
        if not prev_has_tail:
            lead += lead_pad
    else:
        lead += lead_pad

    tail += tail_pad
    return lead, tail


def synthesize_audio(shots: List[Shot], voices: Dict[str, VoiceProfile],
                     cfg: TTSConfig, workdir: Path, voice_narration: bool = True,
                     resume: bool = True) -> None:
    """Generate cloned speech for every line, then a per-shot audio track.

    Loads the chosen TTS engine ONCE, processes all shots, then unloads — the
    single most important thing for fitting a 4090 alongside the video models
    (which we never co-resident with TTS).
    """
    # Ensure TTS-only delivery cues are present even for edited/older plans.
    # These tokens are for audio synthesis only; the visual prompt path strips
    # them and never sends them to image/video models.
    ensure_required_voice_cue_tokens(shots, cfg)

    # Resolve reference transcripts FIRST (Whisper loads + unloads on its own,
    # before the TTS model is on the GPU) for engines that condition on ref text.
    resolve_ref_texts(voices, cfg, workdir)

    if not _HAS_SF:
        # Without soundfile we can't write any .wav, which would silently leave
        # every shot audio-less (and the final film mute) with no other error.
        # Fail loud and early instead.
        raise RuntimeError(
            "soundfile is not installed, so no audio can be written and the "
            "film would be silent. Install it:  pip install soundfile")

    engine = make_tts(cfg)
    logger.info("[VOICE] loading TTS engine: %s", engine.name)
    try:
        engine.load()
    except OSError as e:
        # Surface the NCCL/torch mismatch clearly if it's what broke the load
        # (see _diagnose_torch_cuda_error) — this is the one real CUDA load
        # the voice stage makes, now that there's no separate bootstrap model.
        _diagnose_torch_cuda_error(e)
        raise
    except Exception as e:
        # Any other load-time failure is much easier to act on with the
        # engine name attached, instead of a bare exception surfacing from
        # deep inside its package.
        logger.error("  TTS engine %r failed to load: %s", engine.name, e)
        raise

    # Generate any pending per-character baseline clips (source=="baseline")
    # NOW — the engine is already loaded, so no extra model load is needed —
    # then register every speaker. Baselines become each character's fixed
    # cloning reference for the rest of the film.
    generate_voice_baselines(voices, engine, cfg, workdir, resume=resume)

    # Register every speaker we'll need (incl. NARRATOR) up front.
    for name, vp in voices.items():
        if name == "NARRATOR" and not voice_narration:
            continue
        engine.register(name, vp.ref_wav, vp.params)

    sr = cfg.sample_rate
    adir = workdir / "audio"

    for pos, sh in enumerate(shots):
        shot_path = str(adir / f"shot{sh.index:04d}.wav")
        if resume and Path(shot_path).exists():
            sh.audio_path = shot_path
            sh.duration = _wav_duration(shot_path)
            continue
        lead_ms, tail_ms = _context_silence_ms(shots, pos, cfg)
        chunks: List[np.ndarray] = [_silence(lead_ms / 1000, sr)]
        any_line = False
        for li, ln in enumerate(sh.lines):
            if not ln.text.strip():
                continue
            if ln.speaker.upper() == "NARRATOR" and not voice_narration:
                continue
            if ln.speaker not in voices:
                engine.register(ln.speaker, None, {})
                voices[ln.speaker] = VoiceProfile(name=ln.speaker, ref_wav=None)
            try:
                pieces = chunk_text(ln.text, cfg.max_chars_per_chunk)
                wavs = [np.asarray(engine.synth(pc, ln.speaker, ln.emotion),
                                   dtype=np.float32).reshape(-1) for pc in pieces]
                wav = _crossfade_concat(wavs, engine.sr, cfg.cross_fade_ms)
            except Exception as e:
                logger.warning("  TTS failed (%s / %r): %s", ln.speaker, ln.text[:40], e)
                continue
            if cfg.trim_line_silence:
                wav = _trim_silence(wav, engine.sr)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            line_path = str(adir / f"shot{sh.index:04d}_line{li:02d}.wav")
            if _HAS_SF:
                sf.write(line_path, wav, engine.sr)
            ln.audio_path = line_path
            ln.duration = len(wav) / float(engine.sr)
            chunks.append(wav)
            chunks.append(_silence(cfg.gap_ms / 1000, engine.sr))
            any_line = True

        if not any_line:
            sh.duration = 0.0
            sh.audio_path = None
            continue

        chunks.append(_silence(tail_ms / 1000, engine.sr))
        track = np.concatenate(chunks).astype(np.float32)
        # Sound polish: level every shot to a consistent loudness (so one voice
        # isn't louder than the next), then a tiny edge fade so concatenated
        # shots don't click at the cut.
        if cfg.loudness_normalize:
            track = _normalize_loudness(track, cfg.target_rms, cfg.peak_ceiling)
        else:
            peak = float(np.max(np.abs(track))) or 1.0
            track = (track / peak) * cfg.peak_ceiling
        track = _apply_edge_fades(track, engine.sr, cfg.edge_fade_ms)
        sf.write(shot_path, track, engine.sr)
        if Path(shot_path).exists():            # only claim audio that's really on disk
            sh.audio_path = shot_path
            sh.duration = len(track) / float(engine.sr)
            logger.info("  shot %04d  %5.2fs  %s", sh.index, sh.duration,
                        (sh.speaking_character or "(no dialogue)"))
        else:
            sh.audio_path = None
            logger.warning("  shot %04d  audio file not written (%s)", sh.index, shot_path)

    engine.unload()
    n_voiced = sum(1 for sh in shots if sh.audio_path)
    if n_voiced == 0:
        logger.warning(
            "[VOICE] no shot produced audio — the film will be SILENT. Likely "
            "causes: every shot is non-dialogue (narration off?), or the TTS "
            "engine failed on every line above. Check the warnings, or set "
            "ProjectConfig.voice_narration=True to voice NARRATOR captions.")
    else:
        logger.info("[VOICE] done; %d/%d shots voiced; TTS unloaded.", n_voiced, len(shots))


# =============================================================================
# STAGE 3 · STILL IMAGE  (KLEIN2, at video aspect ratio)
# =============================================================================

_COMP_HINT = {
    "close_up": "tight close-up", "extreme_close": "extreme close-up",
    "medium_shot": "medium shot", "wide_shot": "wide establishing shot",
    "over_shoulder": "over-the-shoulder shot", "dutch_angle": "dutch-angle shot",
    "bird_eye": "high bird's-eye angle", "worm_eye": "low worm's-eye angle",
}


def _char_lookup(characters: List["Character"]) -> Dict[str, "Character"]:
    out = {}
    for c in characters or []:
        nm = getattr(c, "name", "")
        if nm:
            out[nm] = c
            out[nm.split()[0]] = c        # first-name match
    return out


def _plain_spoken_text(text: str) -> str:
    """Remove engine control tokens and normalize spoken text for prompt filtering."""
    s = re.sub(r"<\|[^>]+?\|>", " ", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _remove_dialogue_leaks(text: str, shot: Optional["Shot"] = None) -> str:
    """Strip script/dialogue wording from a visual or motion prompt.

    The goal is not to remove the idea that someone is speaking; it is to
    remove the actual words, captions, subtitles, speech-bubble cues, and
    line fragments that cause image models to render text.
    """
    s = str(text or "")

    # Remove common script/caption lead-ins and everything after them on that
    # sentence/line. This catches phrases introduced by our own earlier helpers
    # such as "speaking this line: ...".
    leak_patterns = [
        r"(?i)\bspeaking this line\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\bsays\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\bsaying\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\bdialogue\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\bcaption\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\bsubtitle\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\btext on screen\s*:\s*[^.\n]*(?:[.\n]|$)",
        r"(?i)\bwords on screen\s*:\s*[^.\n]*(?:[.\n]|$)",
    ]
    for pat in leak_patterns:
        s = re.sub(pat, " ", s)

    # Remove quoted strings, which image models often interpret as visible text.
    s = re.sub(r"[\"“”‘’']([^\"“”‘’']{3,220})[\"“”‘’']", " ", s)

    # Remove exact/near-exact snippets from the actual shot lines.
    if shot is not None:
        for ln in getattr(shot, "lines", []) or []:
            spoken = _plain_spoken_text(getattr(ln, "text", ""))
            if not spoken:
                continue
            variants = {spoken}
            # Also remove the first clause/chunk when the prompt only copied a
            # shortened excerpt of the line.
            for sep in (".", "!", "?", ";", ","):
                if sep in spoken:
                    variants.add(spoken.split(sep)[0].strip())
            words = spoken.split()
            if len(words) >= 6:
                variants.add(" ".join(words[:12]))
                variants.add(" ".join(words[:8]))
            for v in sorted((x for x in variants if len(x) >= 12), key=len, reverse=True):
                s = re.sub(re.escape(v), " ", s, flags=re.IGNORECASE)

    # Remove explicit text-rendering artifacts from the prompt itself.
    s = re.sub(r"(?i)\b(?:speech|dialogue|caption|subtitle|text)\s+(?:bubble|box|caption|overlay)s?\b", " ", s)
    s = re.sub(r"(?i)\b(?:with|showing|displaying)\s+(?:words|text|captions|subtitles)\b[^.,;\n]*", " ", s)

    # Normalize punctuation/spaces.
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"(,\s*){2,}", ", ", s)
    return s.strip(" ,;:-")


def _visual_safe_description(shot: "Shot") -> str:
    """Description for image prompts: visible scene only, no spoken words."""
    raw = shot.description or shot.setting or "the scene"
    cleaned = _remove_dialogue_leaks(raw, shot)
    if not cleaned or len(cleaned) < 8:
        if shot.characters_in_frame:
            cleaned = "A cinematic view of " + ", ".join(shot.characters_in_frame[:2])
        elif shot.setting:
            cleaned = f"A cinematic view of {shot.setting}"
        else:
            cleaned = "A cinematic visual beat"
    return cleaned


_PEOPLE_WORD_RE = re.compile(
    r"(?i)\b("
    r"man|woman|person|people|character|face|mouth|eye|eyes|hand|hands|body|bodies|figure|figures|"
    r"he|she|they|him|her|his|hers|their|someone|speaker|singer|narrator|human|humans"
    r")\b"
)


def _shot_has_visible_people(shot: "Shot") -> bool:
    """Whether the shot's intended image contains visible people/characters.

    Primary source of truth is characters_in_frame. We intentionally do NOT try
    to infer people from story context, because that is exactly how animation
    prompts can hallucinate characters into scenic shots.
    """
    return bool(getattr(shot, "characters_in_frame", []) or [])


def _nonhuman_visual_description(shot: "Shot") -> str:
    """Visible-scene description for no-people shots only."""
    raw = (shot.setting or shot.description or "the scene")
    cleaned = _remove_dialogue_leaks(raw, shot)
    # Strip common human references if they slipped into description text.
    cleaned = _PEOPLE_WORD_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(?i)\b(speaks?|talks?|sings?|walks?|runs?|stands?|stares?|looks?)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    if not cleaned or len(cleaned) < 8:
        if shot.setting:
            cleaned = f"A cinematic view of {shot.setting}"
        else:
            cleaned = "A cinematic environmental beat"
    return cleaned


def _nonhuman_motion_prompt(shot: "Shot", vcfg: Optional["VideoConfig"] = None,
                            cinematic: bool = True) -> str:
    """Deterministic motion prompt for scenes with no visible people."""
    budget = int(vcfg.motion_prompt_max_chars) if vcfg is not None else 900
    desc = _nonhuman_visual_description(shot).rstrip(".")
    for sep in (". ", "; ", ", "):
        if sep in desc:
            desc = desc.split(sep)[0]
            break
    words = desc.split()
    if len(words) > 18:
        desc = " ".join(words[:18])

    bits: List[str] = []
    if desc:
        bits.append(desc)
    if cinematic:
        c = _cinematic_cues(shot, vcfg)
        env_motion = c["motion"]
        # Remove human-centric motion fragments if they slipped in.
        env_motion = _PEOPLE_WORD_RE.sub(" ", env_motion)
        env_motion = re.sub(r"(?i)\b(face|mouth|eyes?|expression|speaking|talking|singing|gesturing)\b", " ", env_motion)
        env_motion = re.sub(r"\s+", " ", env_motion).strip(" ,;:-")
        bits.append(f"Camera: {c['camera']}")
        if env_motion:
            bits.append(env_motion)
    bits.append("Only environmental, object, lighting, weather, or camera motion; no people appear.")
    return _clip_prompt(". ".join(b for b in bits if b) + ".", budget)


def _sanitize_motion_prompt_to_match_image(shot: "Shot", prompt: str,
                                           vcfg: Optional["VideoConfig"] = None) -> str:
    """Make the motion prompt consistent with the visible image content."""
    cleaned = _sanitize_motion_prompt_for_no_dialogue(shot, prompt)
    respect_no_people = (getattr(vcfg, "motion_prompts_respect_no_people_scenes", True)
                         if vcfg is not None else True)
    if respect_no_people and not _shot_has_visible_people(shot):
        # For no-people images, do not risk any human wording at all; replace with
        # a deterministic environment/object/camera motion prompt.
        return _nonhuman_motion_prompt(shot, vcfg, cinematic=True)
    # Visible-people path: keep motion faithful to what the still actually shows.
    cleaned = _strip_offimage_elements(shot, cleaned, vcfg)
    if (getattr(shot, "is_dialogue", False) and getattr(shot, "speaking_character", None)
            and (vcfg is None or getattr(vcfg, "keep_speaker_mouth_visible", True))):
        cleaned = _strip_mouth_occlusion_in_motion(cleaned, shot)
    return cleaned


def _sanitize_image_prompt_for_no_text(shot: "Shot", prompt: str) -> str:
    """Final image-prompt guardrail: visual description only, no dialogue text."""
    cleaned = _remove_dialogue_leaks(prompt, shot)
    # Do not append a positive "no text" sentence here; keep text avoidance in
    # ProjectConfig.image_negative so the positive prompt remains purely visual.
    return cleaned


def _sanitize_motion_prompt_for_no_dialogue(shot: "Shot", prompt: str) -> str:
    """Final motion-prompt guardrail: no speech acts, no mouth/jaw motion language.

    Removes from ALL motion prompts (dialogue or not):
      1. Quoted/embedded dialogue text that leaked from the script.
      2. Speaking/talking/singing verbs and any phrase that implies the character
         is vocally producing sound.
      3. Mouth-motion phrases — "mouth moving", "jaw shifting", "lips forming
         words", etc. — because LatentSync replaces the mouth region frame-by-
         frame in a separate pass, and any animator-generated mouth motion competes
         with that replacement and degrades the final output.

    Conservative on body language: head tilts, eye movement, eyebrow raises,
    breathing, and general facial expressions are intentionally kept — they read
    as emotion rather than speech and do not interfere with lip-sync.
    """
    if not prompt:
        return prompt

    cleaned = _remove_dialogue_leaks(prompt, shot)

    # --- speech-act verbs (the character is producing sound) ---
    cleaned = re.sub(
        r"(?i)\b(speak\w*|talk\w*|sing\w*|utter\w*|shout\w*|yell\w*|cry\s+out|"
        r"whisper\w*|mutter\w*|exclaim\w*|call\s+out|chant\w*|recit\w*|"
        r"narrat\w*|declar\w*|proclaim\w*|announc\w*)\b",
        " ", cleaned)

    # --- caption/subtitle language ---
    cleaned = re.sub(
        r"(?i)\b(?:dialogue|caption|subtitle|script|line|speech\s+bubble)\b",
        " ", cleaned)

    # --- mouth-motion phrases (full phrase, not just the noun) ---
    # These are longer patterns first so they're removed cleanly rather than
    # leaving behind partial matches.
    cleaned = re.sub(
        r"(?i)\bmouth\s+(?:mov\w*|open\w*|clos\w*|form\w*|shar\w*|speak\w*|"
        r"say\w*|utter\w*|work\w*|function\w*)[^,;.]*",
        " ", cleaned)
    cleaned = re.sub(
        r"(?i)\blips?\s+(?:mov\w*|form\w*|part\w*|press\w*|pursed?|work\w*|"
        r"speak\w*|sync\w*)[^,;.]*",
        " ", cleaned)
    # "jaw shifting / dropping / moving / working / clenching for speech"
    cleaned = re.sub(
        r"(?i)\bjaw\s+(?:shift\w*|drop\w*|mov\w*|work\w*|open\w*|lower\w*|"
        r"unclench\w*|clench\w*\s+(?:and\s+)?(?:releas\w*|open\w*))[^,;.]*",
        " ", cleaned)
    # "as he/she/they speaks/talks/says/utters"
    cleaned = re.sub(
        r"(?i)\bas\s+(?:he|she|they|the\s+\w+)\s+(?:speak\w*|talk\w*|say\w*|"
        r"utter\w*|sing\w*)\b[^,;.]*",
        " ", cleaned)
    # "naturally as he speaks", "while speaking", "while talking"
    cleaned = re.sub(
        r"(?i)\b(?:naturally\s+as\s+\w+\s+speak\w*|while\s+(?:speak\w*|talk\w*|"
        r"sing\w*)|in\s+mid[- ]?speech|mid[- ]?sentence)\b[^,;.]*",
        " ", cleaned)
    # "mouth clearly visible" — framing concern, not needed in motion prompt
    cleaned = re.sub(r"(?i),?\s*mouth\s+clearly\s+visible\b", " ", cleaned)

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:-")
    return cleaned


# ---------------------------------------------------------------------------
# Consistency guards (quality pass)
# ---------------------------------------------------------------------------
# Three problems these solve, all requested for better animation:
#   1. A speaking character must never have a hand / food / drink / object over
#      the mouth — it breaks lip-sync face detection and reads as "not talking".
#   2. The motion prompt must animate only what the still image actually shows;
#      it must not introduce a person, animal, vehicle, or object that is not in
#      the image (a very common motion-model hallucination).
#   3. Every still must carry the SAME canonical style string so the whole film
#      shares one look, instead of relying on the LLM to re-emit it per shot.

# Concrete, discrete things a motion model loves to invent. Weather / light /
# wind are deliberately EXCLUDED — the cinematic-cue layer adds those on purpose
# and they don't read as "an object that isn't there". Kept to nouns whose
# absence-from-image is unambiguous.
_HALLUCINATION_PRONE_ELEMENTS = {
    "train", "car", "truck", "bus", "van", "taxi", "motorcycle", "motorbike",
    "bicycle", "bike", "scooter", "boat", "ship", "plane", "airplane",
    "aircraft", "jet", "helicopter", "drone", "carriage", "wagon", "cart",
    "bird", "dog", "cat", "horse", "cow", "sheep", "deer", "wolf", "bear",
    "butterfly", "insect", "fish", "snake", "rabbit", "squirrel", "crowd",
    "dancer", "soldier", "child", "baby", "ball", "flag", "kite", "balloon",
    "confetti", "firework", "fireworks", "lantern", "umbrella",
}


def _image_content_text(shot: "Shot") -> str:
    """All text that describes what the still actually depicts, lower-cased."""
    return " ".join(
        t for t in (shot.image_prompt or "", shot.description or "", shot.setting or "")
        if t
    ).lower()


def _depicted_extract(shot: "Shot", limit: int = 240) -> str:
    """A short summary of what the still actually shows (Subject/Action/Objects).

    Fed to the motion-prompt LLM so it animates only what is in the image rather
    than re-imagining the beat. Falls back to the shot description.
    """
    ip = shot.image_prompt or ""
    wanted = []
    for ln in ip.split("\n"):
        low = ln.lower()
        if low.startswith(("subject:", "action:", "objects:", "environment:")):
            wanted.append(ln.strip())
    text = " ".join(wanted) if wanted else (shot.description or shot.setting or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _is_offimage_word(word: str, content: str) -> bool:
    """True if ``word`` names a discrete element that is NOT in the image."""
    w = word.strip(".,;:!?\"'()").lower()
    if not w:
        return False
    base = w[:-1] if (w.endswith("s") and len(w) > 3) else w
    if w not in _HALLUCINATION_PRONE_ELEMENTS and base not in _HALLUCINATION_PRONE_ELEMENTS:
        return False
    # Present in the image content (either form) → legitimately in frame, keep.
    return (w not in content) and (base not in content)


def _strip_offimage_elements(shot: "Shot", motion_prompt: str,
                             vcfg: Optional["VideoConfig"] = None) -> str:
    """Drop motion clauses that animate a discrete element not in the image.

    Conservative: only touches the curated ``_HALLUCINATION_PRONE_ELEMENTS`` set,
    only when the element is absent from the still's own description, and only
    the offending comma/sentence clause — never the whole prompt. Never returns
    empty (falls back to the original if everything would be stripped).
    """
    if not motion_prompt:
        return motion_prompt or ""
    if vcfg is not None and not getattr(vcfg, "motion_prompts_only_visible_elements", True):
        return motion_prompt
    content = _image_content_text(shot)
    kept_sentences: List[str] = []
    changed = False
    for sentence in re.split(r"(?<=[.;])\s+", motion_prompt):
        clauses = re.split(r"\s*,\s*", sentence)
        kept = []
        for c in clauses:
            if any(_is_offimage_word(w, content) for w in c.split()):
                changed = True
                continue
            if c.strip():
                kept.append(c.strip())
        if kept:
            kept_sentences.append(", ".join(kept))
    result = " ".join(s for s in kept_sentences if s).strip()
    if changed:
        logger.info("[MotionGuard] shot %s: removed off-image element(s) so motion "
                    "matches the still.", getattr(shot, "index", "?"))
    return result or motion_prompt


# Mouth-occlusion detection, applied clause-by-clause so we drop only the
# offending fragment and keep legitimate speaking cues ("mouth slightly open").
_OCC_MOUTH_WORDS = re.compile(r"(?i)\b(mouth|lips)\b")
_OCC_HIDE_WORDS = re.compile(
    r"(?i)\b(cover\w*|cup\w*|muffl\w*|stifl\w*|hid\w*|obscur\w*|conceal\w*|behind|over|across)\b")
_OCC_HAND_TO_FACE = re.compile(
    r"(?i)\bhands?\b[^,;.]*\b(mouth|lips|chin|jaw|face|nose)\b")
_OCC_CONSUMING = re.compile(
    r"(?i)\b(bit\w*|chew\w*|eat\w*|drink\w*|sip\w*|sipp\w*|swallow\w*|smok\w*|"
    r"vap\w*|puff\w*|munch\w*|mouthful|takes? a (?:bite|sip|drag|puff))\b")
_OCC_PROP_OVER = re.compile(
    r"(?i)\b(mask|scarf|veil|cloth|bandana|respirator|microphone|\bmic\b|megaphone|"
    r"cup|mug|glass|bottle|straw|cigarette|cigar|phone)\b[^,;.]*\b(mouth|lips|face)\b")


def _clause_occludes_mouth(clause: str) -> bool:
    """True if a clause would put something over / against a speaker's mouth."""
    if _OCC_CONSUMING.search(clause):
        return True
    if _OCC_HAND_TO_FACE.search(clause):
        return True
    if _OCC_PROP_OVER.search(clause):
        return True
    if _OCC_MOUTH_WORDS.search(clause) and _OCC_HIDE_WORDS.search(clause):
        return True
    return False


def _drop_occluding_clauses(value: str) -> Tuple[str, bool]:
    """Remove mouth-occluding comma/period clauses from a single field value."""
    changed = False
    kept = []
    for clause in re.split(r"\s*[,;]\s*|(?<=[.])\s+", value):
        c = clause.strip()
        if not c:
            continue
        if _clause_occludes_mouth(c):
            changed = True
            continue
        kept.append(c)
    return ", ".join(kept), changed


def _strip_mouth_occlusion_in_motion(motion_prompt: str, shot: "Shot") -> str:
    """For a speaking shot, drop motion that covers the mouth; never go empty.

    Also positively asserts that hands stay away from the face for the duration
    of the clip. FramePack (the default video engine) runs with real CFG=1.0, so
    its negative prompt is ignored — the motion prompt's positive text is the
    only lever that keeps a hand from drifting up to the mouth mid-clip, so we
    state the desired behaviour directly rather than relying on the negative.
    """
    if not motion_prompt:
        motion_prompt = ""
    cleaned, changed = _drop_occluding_clauses(motion_prompt)
    if changed:
        logger.info("[MouthGuard] shot %s: removed mouth-occluding motion clause "
                    "(hands-away framing preserved).", getattr(shot, "index", "?"))
    if not cleaned.strip():
        # Everything was occlusion — fall back to a neutral held pose. Do NOT
        # describe the mouth moving or the character speaking here: lip-sync is
        # handled by LatentSync in a separate pass, and any animator-generated
        # mouth motion competes with (and degrades) that pass.
        cleaned = "The character faces the camera, expression attentive, subtle head tilt; still posture"
    # Positive, negation-free steer so the video engine keeps hands down.
    if "away from the face" not in cleaned.lower():
        cleaned = cleaned.rstrip(" .") + ". " + _HANDS_AWAY_MOTION_CUE + "."
    return cleaned


# Affirmative, negation-free cues. These describe the DESIRED state so a
# CFG-free positive-only model (Z-Image-Turbo / FramePack) is steered correctly
# — see the long note in _lipsync_face_safety_instruction for why "no hand over
# mouth" phrasing is avoided in positive prompts.
_HANDS_AWAY_IMG_CUE = ("hands resting low and away from the face, "
                       "mouth and jaw fully visible and in sharp focus")
_HANDS_AWAY_MOTION_CUE = ("hands stay low and away from the face throughout")


def _strip_mouth_occlusion_in_image_prompt(prompt: str, shot: "Shot") -> str:
    """For a speaking shot, remove mouth-occluding gestures from the still prompt.

    Operates line-by-line so the labelled structure (Subject/Action/...) is
    preserved; only the Subject/Action/Objects field VALUES are cleaned of
    clauses that place objects in front of the mouth or describe eating/drinking/
    smoking/covering the face. Does NOT inject new action language or hands-away
    cues — those belong in the motion prompt, not the static image prompt.
    """
    if not prompt:
        return prompt or ""
    out_lines = []
    for ln in prompt.split("\n"):
        low = ln.lower()
        if low.startswith(("subject:", "action:", "objects:")) and ":" in ln:
            label, _, body = ln.partition(":")
            cleaned, changed = _drop_occluding_clauses(body.strip())
            if changed:
                if cleaned and not cleaned.endswith((".", "!", "?")):
                    cleaned += "."
                out_lines.append(f"{label}: {cleaned}".rstrip())
                continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def _canonical_style(theme: str) -> str:
    """The single style string every still must carry, for one consistent look."""
    return (theme or "cinematic, filmic").strip().rstrip(".")


def _enforce_style_anchor(image_prompt: str, theme: str) -> str:
    """Guarantee the identical canonical style appears in this still's prompt.

    The image LLM is told to fold the global style in, but it can paraphrase or
    drop it, causing shot-to-shot style drift. This deterministically ensures
    the exact same style tokens are present in every image prompt.
    """
    anchor = _canonical_style(theme)
    if not anchor or anchor.lower() in (image_prompt or "").lower():
        return image_prompt
    lines = (image_prompt or "").split("\n")
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("style details:") or low.startswith("style:"):
            label, _, body = ln.partition(":")
            body = body.strip().rstrip(".")
            merged = f"{body}; {anchor}" if body else anchor
            lines[i] = f"{label}: {merged}."
            return "\n".join(lines)
    lines.append(f"Style Details: {anchor}.")
    return "\n".join(lines)


def _finalize_image_prompt(shot: "Shot", prompt: str, theme: str) -> str:
    """One idempotent finalizer for every still prompt.

    Applies (in order): no-text/dialogue-leak sanitize, mouth-occlusion cleanup
    for speaking shots (strips bad clauses only — no action injection), a locked
    style anchor for a consistent look, and for dialogue shots a compact
    face-framing note folded into the Subject line. Safe to call more than once.

    Action content and hands-away cues are NOT added here; they belong in the
    motion prompt (see _lipsync_motion_framing_cue / generate_motion_prompts).
    """
    prompt = _sanitize_image_prompt_for_no_text(shot, prompt)
    if getattr(shot, "is_dialogue", False) and getattr(shot, "speaking_character", None):
        prompt = _strip_mouth_occlusion_in_image_prompt(prompt, shot)
    prompt = _enforce_style_anchor(prompt, theme)
    # Fold the face-framing note into the Subject line so it reads as a physical
    # description rather than a standalone technical instruction.
    face_note = _lipsync_face_safety_instruction(shot)
    if face_note and face_note not in prompt:
        lines = prompt.split("\n")
        for i, ln in enumerate(lines):
            if ln.lower().startswith("subject:") and ":" in ln:
                label, _, body = ln.partition(":")
                body = body.strip().rstrip(".")
                lines[i] = f"{label}: {body}; {face_note}"
                prompt = "\n".join(lines)
                break
    return prompt


# Generic but non-plain environment elaboration, keyed by rough mood/keyword,
# used ONLY by the emergency fallback builder below when the LLM authoring
# pass is unavailable/failed. These exist so the fallback path — which used
# to just echo the bare `shot.setting` string — still produces a layered,
# imaginative Environment field instead of a flat, empty backdrop. Each entry
# gives foreground/midground/background texture that can wrap around
# WHATEVER setting text the shot actually has, rather than replacing it.
_FALLBACK_ENV_TEXTURE = {
    "dark":   "Foreground catches scattered highlights against deep shadow. "
              "Midground recedes into gloom broken by a single light source. "
              "Background dissolves into near-darkness with a faint horizon line.",
    "night":  "Foreground is lit by whatever local light source is nearby. "
              "Midground falls into cool blue shadow. Background shows a "
              "night sky or distant lights receding into darkness.",
    "bright": "Foreground is crisply lit with clear detail. Midground carries "
              "warm ambient light. Background opens into a sunlit horizon "
              "with visible depth and atmosphere.",
    "storm":  "Foreground debris and texture caught mid-motion in the wind. "
              "Midground obscured by blowing rain or dust. Background shows "
              "roiling storm clouds and a churning sky.",
    "indoor": "Foreground shows nearby furniture or architectural detail in "
              "sharp focus. Midground carries the room's working light. "
              "Background recedes through a doorway or window into a further "
              "space, giving the room real depth rather than a flat wall.",
}
_DEFAULT_FALLBACK_ENV = (
    "Foreground carries specific, tangible detail near the subject. Midground "
    "shows the working space of the scene. Background recedes into an "
    "atmospheric, layered distance rather than a flat backdrop."
)


def _fallback_environment_block(shot: "Shot", theme: str) -> str:
    """Wrap whatever setting text exists with foreground/midground/background
    texture so the EMERGENCY fallback path never emits a bare, plain backdrop.

    Never invents a new location — it only adds spatial layering AROUND the
    setting the shot already specifies (or, if the shot has no setting at
    all, around the film's theme), which is the single biggest lever for
    avoiding the "single character, plain background" failure mode when the
    richer LLM-authored path is unavailable.
    """
    base = (shot.setting or "").strip()
    mood_key = (shot.mood or "").lower()
    texture = _DEFAULT_FALLBACK_ENV
    for key, block in _FALLBACK_ENV_TEXTURE.items():
        if key in mood_key or key in base.lower():
            texture = block
            break
    if base:
        return f"{base}. {texture}"
    # No setting at all — lean on the theme so it's still specific to this
    # film rather than a generic empty room.
    return f"{theme or 'The scene'}. {texture}"


def build_image_prompt(shot: Shot, char_idx: Dict[str, "Character"],
                       theme: str) -> str:
    """EMERGENCY FALLBACK ONLY. Assemble a prompt from the shot + canonical
    character looks when the LLM-authored structured pass (generate_image_
    prompts) is unavailable or failed for this shot.

    This used to be a bare-bones builder (Subject/Action/Environment/Camera/
    Mood/Style, with Environment being nothing more than the raw shot.setting
    string) that silently produced flat, plain-background, single-character
    images whenever the richer path didn't cover a shot. It now always emits
    all eight labelled sections — including Clothing and Objects, which were
    previously missing entirely from this path — and layers the Environment
    field with foreground/midground/background texture (see
    `_fallback_environment_block`) instead of echoing the bare setting.
    """
    people = []
    clothing_bits = []
    for nm in shot.characters_in_frame[:3]:
        ch = char_idx.get(nm) or char_idx.get(nm.split()[0] if nm else "")
        if ch:
            g = getattr(ch, "gender", "") or ""
            build = getattr(ch, "physical_build", "") or ""
            appear = getattr(ch, "appearance", "") or ""
            seg = ", ".join(x for x in (g, build, appear) if x)
            people.append(f"{nm} ({seg})" if seg else nm)
            outfit = getattr(ch, "costume", "") or getattr(ch, "outfit", "") or ""
            if outfit:
                clothing_bits.append(f"{nm}: {outfit}")
        elif nm:
            people.append(nm)
    parts = []
    if people:
        parts.append("Subject: " + "; ".join(people))
    if clothing_bits:
        parts.append("Clothing: " + "; ".join(clothing_bits))
    visual_desc = _visual_safe_description(shot)
    if visual_desc:
        parts.append("Action: " + visual_desc)
    # Environment ALWAYS gets spatial layering, even with no setting text —
    # this is the single change that prevents the plain-backdrop failure mode.
    parts.append("Environment: " + _fallback_environment_block(shot, theme))
    # A minimal, generic prop line so Objects is never silently empty; this is
    # deliberately soft ("if visible") since the fallback has no real prop
    # knowledge, but an empty Objects section is one of the clearest tells of
    # a plain, under-specified image.
    parts.append(
        "Objects: specific, tangible props consistent with the setting are "
        "visible nearby, grounding the scene in physical detail."
    )
    # Anchor the subject's frame position explicitly so the diffusion model
    # cannot fill empty wide-frame space by duplicating the subject — but
    # phrase it as a positive framing choice rather than bare "single figure,
    # plain" language, which read as an instruction to keep things minimal.
    cam_base = _COMP_HINT.get(shot.composition or "", "medium shot")
    anchor = (
        "one clearly composed subject anchored off-centre against a deep, "
        "layered background, no duplicate figures"
        if len(shot.characters_in_frame) <= 1
        else "subjects grouped with the environment visible around and "
             "behind them, no duplicates"
    )
    parts.append(f"Camera: {cam_base}; {anchor}.")
    if shot.mood:
        parts.append("Mood: " + shot.mood)
    parts.append("Style Details: " + (theme or "cinematic, filmic, richly detailed illustration"))
    return "\n".join(parts)


# Labelled sections, in the order the image model was tuned on (matches the
# comic generator's _ZIMAGE_SECTION_ORDER and the attached few-shot examples).
_STRUCT_SECTIONS = ("subject", "clothing", "action", "environment",
                    "objects", "lighting", "camera", "style")
_STRUCT_LABELS = {"subject": "Subject", "clothing": "Clothing", "action": "Action",
                  "environment": "Environment", "objects": "Objects",
                  "lighting": "Lighting", "camera": "Camera", "style": "Style Details"}

# Five exemplars drawn from real production prompts, chosen for variety of shot
# type (wide landscape, medium, close-up, action, environmental), rich layered
# Environment fields (foreground / midground / background all named), and a
# single unambiguous subject.  These teach the LLM the density and spatial
# specificity needed to prevent diffusion models from repeating the subject to
# fill empty wide-frame space.
_STRUCT_FEWSHOT = (
    # --- 1. Wide landscape, single silhouetted subject, strong env layering ---
    "Digital illustration of a woman in silhouette sitting on a small wooden boat facing away at sunset\n"
    "Subject: A young woman with long dark hair tied in a ponytail, gazing toward the horizon.\n"
    "Clothing: Loose-fitting traditional robes resembling a kimono, layered and draped naturally.\n"
    "Action: Seated cross-legged inside the boat, hands resting gently on her lap, body angled toward the setting sun.\n"
    "Environment: Foreground — calm dark water, the wooden hull of the boat occupying the lower third of frame. Midground — fiery orange reflections rippling outward from the subject. Background — distant rocky shores silhouetted under a vast open horizon, enormous red sun dominating the upper half of the sky, stylized cloud formations and bird silhouettes at the sun's edge.\n"
    "Objects: A sheathed katana rests beside her against the hull, handle visible.\n"
    "Lighting: Intense warm glow from the enormous red sun backlights the figure, casting dramatic reflections across the water and silhouetting the foreground hull.\n"
    "Camera: Wide shot, camera positioned low over the water surface, subject centered in silhouette against the sun; no other figures in frame.\n"
    "Style Details: Anime-inspired art style, bold color contrasts, painterly brushwork suggesting motion in ripples and atmosphere.\n"
    "---\n"
    # --- 2. Medium shot, night cityscape, strong foreground/bg separation ---
    "A cinematic digital painting of a lone figure on a skyscraper ledge at night overlooking a neon metropolis\n"
    "Subject: A single girl with short dark hair, seated sideways in profile on a rooftop edge, facing away from camera toward the skyline.\n"
    "Clothing: Orange cropped jacket, light-colored shorts, black knee-high socks, chunky black shoes.\n"
    "Action: Seated casually on the flat roof edge, knees bent slightly forward, expression contemplative as she gazes toward the distant horizon.\n"
    "Environment: Foreground — the concrete rooftop ledge and the subject's feet dangling above the drop. Midground — the immediate rooftop surface, ventilation units, and water tower silhouettes to the right. Background — a futuristic metropolis stretching to the horizon, glowing billboards, skyscraper grids, flying vehicle trails, and vibrant streets under a deep purple star-dotted sky.\n"
    "Lighting: Cool-toned ambient city glow dominates the scene; warm neon signs cast colored reflections on building surfaces; soft highlights define the figure's jacket edge without harsh shadows.\n"
    "Camera: High-angle wide shot capturing the solitary figure in the lower-left foreground and the expansive urban vista filling the right two-thirds of the frame; no duplicate figures.\n"
    "Style Details: Anime-inspired aesthetic, cel-shaded rendering, saturated purples, blues, oranges and pinks, clean lines with painterly brushstrokes.\n"
    "---\n"
    # --- 3. Rainy night medium shot, rich bokeh env, intimate single subject ---
    "Anime-style digital art of a girl sheltering under an umbrella on a rain-soaked urban night\n"
    "Subject: A young girl with long black hair and striking red eyes, crouching under a large dark umbrella; soft but distant expression gazing downward.\n"
    "Clothing: Traditional Japanese sailor-style school uniform — white short-sleeved shirt with navy collar, red necktie, pleated black skirt, knee-high socks, polished loafers; small black shoulder bag slung across her body.\n"
    "Action: Crouching low on wet pavement, one hand holding the umbrella handle, gaze directed downward at an orange tabby cat sitting nearby on the reflective ground.\n"
    "Environment: Foreground — wet pavement reflecting warm circular bokeh glows, the cat on the ground beside her. Midground — the umbrella canopy sheltering the subject. Background — rain-soaked urban night with blurred bokeh lights suggesting distant streetlamps and building windows; vertical rain streaks create atmospheric depth across the full frame.\n"
    "Lighting: Cool blue-toned ambient light dominates; warm out-of-focus city sources create glowing circles of contrast; subtle rim lighting on hair and shoulders silhouettes the figure against the dark backdrop.\n"
    "Camera: Medium close-up, slightly elevated angle, subject filling the left half of the frame; the cat anchors the lower-right foreground; no other human figures.\n"
    "Style Details: Anime-inspired digital art, smooth shading, detailed fabric folds and water droplets, cinematic composition, visual-novel keyframe aesthetic.\n"
    "---\n"
    # --- 4. Action wide shot, dynamic environment, single warrior subject ---
    "Oil painting of a silhouetted warrior standing on rocky ground facing an immense glowing sun\n"
    "Subject: A single silhouetted figure, back to the viewer, long hair whipping in the wind.\n"
    "Clothing: Dark tattered clothing with glowing red energy markings along the sleeves and hem.\n"
    "Action: Standing still on the rock, weight slightly forward, katana held low at the right side — a moment of stillness before action.\n"
    "Environment: Foreground — jagged rocky ground beneath the figure's feet, loose debris scattered left and right. Midground — a traditional Japanese wooden gate structure in silhouette to the left. Background — a misty mountain range receding into pale fog; a colossal white sun with swirling red energy rings fills the upper center of the sky.\n"
    "Objects: Katana held low at the figure's right side.\n"
    "Lighting: Soft atmospheric backlighting from the immense sun; high contrast between the bright sky and the deeply shadowed foreground rocks and figure.\n"
    "Camera: Wide ground-level shot, figure anchored in the lower-center of the frame, the sun and sky occupying the upper two-thirds; no other figures.\n"
    "Style Details: Loose expressive oil-painting brushstrokes, epic dramatic fantasy mood, warm-cool color contrast between glowing sky and dark earth.\n"
    "---\n"
    # --- 5. Close-up portrait, minimal env, zero risk of subject duplication ---
    "Close-up hyper-realistic portrait of a rugged warrior in a snowy environment\n"
    "Subject: A single weathered male face — deep-set amber eyes, prominent forehead wrinkles, thick dark eyebrows dusted with frost, sharp nose, full lips slightly parted; long white hair and beard framing the face.\n"
    "Clothing: Dark textured collar partially visible at the base of frame, blending into shadows.\n"
    "Action: Staring intensely forward, slight tension in the facial muscles suggesting determination or weariness.\n"
    "Environment: Foreground — falling snowflakes in sharp focus drifting across the face. Background — a blurred dark winter landscape, out-of-focus and minimal, keeping all attention on the single face.\n"
    "Lighting: Soft directional light from the upper left highlights the forehead and cheekbones; deeper shadows fall under the jawline and eye sockets.\n"
    "Camera: Tight close-up, face filling 80% of the frame, centered; no other figures or subjects.\n"
    "Style Details: Cinematic composition, hyper-realistic skin texture and frost detail, dramatic atmosphere emphasizing resilience.\n"
    "---\n"
    # --- 6. Multi-character group shot, layered beach environment, named props ---
    "Animated 3D still life of two companions relaxing together near the ocean\n"
    "Subject: A smiling young woman in casual clothes seated between her two travelling "
    "companions, all three at ease with one another.\n"
    "Clothing: She wears an oversized hoodie with matching shorts and sneakers; one "
    "companion wears a patterned wrap top and a woven skirt with fringes and a shell "
    "necklace; the other stands just behind holding a large carved hook-shaped tool.\n"
    "Action: All three seated or standing close together on a mossy rock, smiling at "
    "one another, bodies angled inward toward the group.\n"
    "Environment: Foreground — grass scattered with colourful flowers and a mossy rock "
    "seat. Midground — the companions grouped on the rock, a small animal at their feet. "
    "Background — a vibrant tropical shoreline under bright daylight: turquoise ocean "
    "water, distant palm trees, and pale sandy shoreline stretching to the horizon.\n"
    "Objects: A large intricately carved hook-shaped tool rests over one companion's "
    "shoulder; a small animal sits at their feet.\n"
    "Lighting: Bright natural sunlight from above and slightly in front, casting soft "
    "shadows on the grass and warm highlights across every face.\n"
    "Camera: Medium group shot, all three subjects filling the centre of frame, "
    "shoreline and horizon visible behind them; no extra figures.\n"
    "Style Details: Polished 3D animation style, smooth textures, expressive faces, "
    "richly saturated colours, detailed environmental foreground elements.\n"
    "---\n"
    # --- 7. Character-driven creature portrait, environment implied through palette ---
    "Digital art of a feral, chaotic figure with an aggressive presence\n"
    "Subject: A muscular humanoid figure with a wild pink mohawk styled into two high "
    "pigtails, pale skin marked with intricate tribal tattoos and bold geometric face "
    "paint.\n"
    "Clothing: A textured magenta outfit with jagged zig-zag trim at the hem, layered "
    "necklaces of bone-shard beads with a skull pendant at the centre chest.\n"
    "Action: Snarling with mouth open, fangs bared, eyes narrowed in an intense glare "
    "directed at something just off-frame.\n"
    "Environment: Foreground — cracked dry ground underfoot. Midground — drifting smoke "
    "haze at knee height. Background — a blurred, stormy blue sky suggesting an exposed "
    "outdoor battleground under harsh daylight.\n"
    "Lighting: Hard overhead daylight, deep contrast shadows carving the face and "
    "tattoos, a cool rim light separating the figure from the sky behind.\n"
    "Camera: Medium close-up, figure slightly off-centre, shoulders filling the lower "
    "frame; no other figures.\n"
    "Style Details: High-detail digital illustration, saturated tribal colour accents, "
    "gritty painterly rendering with sharp linework.\n"
    "---\n"
    # --- 8. Pure-environment establishing shot, no human figure, mythic scale ---
    "3D render of a majestic dragon-like presence emerging from stormy clouds\n"
    "Subject: A powerful eastern-style dragon head with layered coral-red horns, a "
    "voluminous white mane like flowing silk, and a single amber eye reflecting the sky.\n"
    "Environment: Foreground — swirling mist at the base of the frame. Midground — the "
    "dragon's mane and horns catching the light. Background — deep grey cumulus storm "
    "clouds contrasting soft illuminated cloud layers, small stylized birds flying near "
    "the snout for scale.\n"
    "Lighting: Soft directional light from the upper left casting gentle shadows across "
    "the scales, warm red tones on the face against cooler whites along the neck.\n"
    "Camera: Wide low-angle shot, the head and horns filling the upper two-thirds of the "
    "frame, clouds receding into deep background; no human figures.\n"
    "Style Details: Intricate traditional-dragon-art detailing blended with modern "
    "digital rendering, vibrant reds paired with clean whites, dramatic mythic scale."
)


def _render_structured_prompt(summary: str, comp: Dict[str, str]) -> str:
    """Summary line + labelled sections (exactly the attached example shape)."""
    lines = [summary.strip()] if summary and summary.strip() else []
    for key in _STRUCT_SECTIONS:
        body = str(comp.get(key, "") or "").strip().strip(" ,;")
        if body:
            if not body.endswith((".", "!", "?")):
                body += "."
            lines.append(f"{_STRUCT_LABELS[key]}: {body}")
    return "\n".join(lines)


# =============================================================================
# CREATIVE QUALITY PASSES
# =============================================================================
# Five optional passes that run during plan() to raise story and visual quality.
# All are gated by ProjectConfig flags so they can be disabled individually.
# Every looping LLM call uses cached_prefix to maximise Grok prefix-cache hits:
# the stable instruction block (identical across batches) sits at the head of
# every payload, and Grok's automatic prefix caching serves it at a discount
# after the first call in a conv-id session.
# =============================================================================

# ---------------------------------------------------------------------------
# Pass 1 — Story doctor
# ---------------------------------------------------------------------------
_STORY_DOCTOR_PREFIX = (
    "You are a story development consultant with deep knowledge of narrative "
    "craft, dramatic structure, and what makes stories emotionally resonant for "
    "general audiences. You evaluate premises against four core requirements: "
    "(1) a protagonist who wants something specific, (2) a meaningful obstacle "
    "that prevents easy achievement, (3) a real cost or risk if they fail or "
    "compromise, and (4) a change — the protagonist ends different from how "
    "they started, in a way that earns an emotional response. Stories that meet "
    "all four requirements consistently outperform those that don't, regardless "
    "of genre. You make targeted, minimal interventions — you strengthen what "
    "is weak without rewriting what is working."
)


def story_doctor_pass(story_idea: str, characters: list, pcfg: "ProjectConfig") -> str:
    """Pressure-test the premise and return a strengthened version if needed.

    Checks the four narrative requirements (want / obstacle / cost / change)
    and returns either the original idea (unchanged, if it already passes) or
    an enriched version that addresses the weaknesses. Never rewrites genre,
    setting, or character identities — only sharpens the dramatic stakes.

    Uses cached_prefix so the stable instruction block is served from Grok's
    prefix cache on every call in this conv-id session.
    """
    if not _HAS_NG or not getattr(pcfg, "story_doctor", True):
        return story_idea

    char_names = ", ".join(
        getattr(c, "name", str(c)) for c in (characters or [])
    ) or "(none yet)"

    variable_block = (
        f"STORY IDEA:\n{story_idea.strip()}\n\n"
        f"CAST: {char_names}\n\n"
        "TASK:\n"
        "Score this premise on each of the four requirements (1-5). For any "
        "requirement scoring below 3, write a single targeted fix — one or two "
        "sentences that add the missing element without altering genre, setting, "
        "or character identities. Then output the enriched premise.\n\n"
        "Return ONLY a JSON object:\n"
        '{"scores": {"want": N, "obstacle": N, "cost": N, "change": N}, '
        '"fixes": {"want": "...", "obstacle": "...", "cost": "...", "change": "..."}, '
        '"enriched_premise": "the full improved story idea (or the original if all scores >= 3)"}'
    )

    try:
        raw = ng.get_openai_prompt_response(
            variable_block,
            temperature=0.5,
            openai_model=getattr(ng, "openai_model_large", None),
            use_grok=getattr(ng, "USE_GROK", True),
            cached_prefix=_STORY_DOCTOR_PREFIX,
        )
        data = ng.parse_json_response(raw)
        if not isinstance(data, dict):
            return story_idea
        scores = data.get("scores", {})
        enriched = (data.get("enriched_premise") or "").strip()
        if not enriched:
            return story_idea
        low = [k for k, v in scores.items() if isinstance(v, (int, float)) and v < 3]
        if low:
            logger.info(
                "[StoryDoctor] Weak on: %s. Premise strengthened.", ", ".join(low)
            )
        else:
            logger.info("[StoryDoctor] Premise passes all four requirements.")
        return enriched if enriched != story_idea.strip() else story_idea
    except Exception as e:
        logger.warning("[StoryDoctor] Pass skipped (%s).", e)
        return story_idea


# ---------------------------------------------------------------------------
# Pass 2 — Visual treatment (per-film style bible)
# ---------------------------------------------------------------------------
_VISUAL_TREATMENT_PREFIX = (
    "You are a cinematographer and visual designer authoring a visual treatment "
    "for a short animated film. A visual treatment is a one-page document that "
    "gives the whole film a deliberately authored look: a signature palette, a "
    "recurring light quality, a compositional grammar, and a texture motif. "
    "Great films are visually coherent — the opening and the climax share DNA "
    "even while feeling different. Your treatment must be specific and concrete: "
    "not 'warm tones' but 'amber-and-teal with deep burgundy shadows'; not "
    "'dramatic lighting' but 'a single hard source raking from low-left, casting "
    "long diagonal shadows that compress the space'. Every choice should serve "
    "the story's emotional arc. The treatment is written as creative vocabulary "
    "that will be threaded into per-shot image prompts — it must be evocative "
    "and precise, not generic."
)


def generate_visual_treatment(
    story_idea: str,
    characters: list,
    theme: str,
    pcfg: "ProjectConfig",
) -> Dict[str, str]:
    """Generate a per-film visual treatment: palette, light, composition, texture.

    Returns a dict with keys: palette, light_quality, composition_grammar,
    texture_motif, atmosphere. Falls back to an empty dict on any failure so
    callers can treat it as optional enrichment without error handling.

    Uses cached_prefix for Grok prefix-cache efficiency.
    """
    empty: Dict[str, str] = {}
    if not _HAS_NG or not getattr(pcfg, "visual_treatment", True):
        return empty

    char_names = "; ".join(
        f"{getattr(c, 'name', '?')} ({getattr(c, 'role', '')})"
        for c in (characters or [])
    ) or "(none)"

    variable_block = (
        f"STORY:\n{story_idea.strip()[:600]}\n\n"
        f"CAST: {char_names}\n"
        f"BASE STYLE: {theme}\n\n"
        "Author a visual treatment for this film. Be ruthlessly specific.\n\n"
        "Return ONLY a JSON object:\n"
        '{"palette": "2-3 specific named colours + their emotional role", '
        '"light_quality": "exact light direction, hardness, colour temp, and what it does emotionally", '
        '"composition_grammar": "the recurring compositional idea — how subjects are placed, '
        'what negative space does, what leading lines the film returns to", '
        '"texture_motif": "2-3 specific materials or surfaces that recur and carry meaning", '
        '"atmosphere": "physical conditions — humidity, dust, temperature that should be felt in every frame"}'
    )

    try:
        raw = ng.get_openai_prompt_response(
            variable_block,
            temperature=0.72,
            openai_model=getattr(ng, "openai_model_large", None),
            use_grok=getattr(ng, "USE_GROK", True),
            cached_prefix=_VISUAL_TREATMENT_PREFIX,
        )
        data = ng.parse_json_response(raw)
        if not isinstance(data, dict):
            return empty
        treatment = {
            k: str(v).strip()
            for k, v in data.items()
            if k in ("palette", "light_quality", "composition_grammar",
                     "texture_motif", "atmosphere") and v
        }
        if treatment:
            logger.info(
                "[VisualTreatment] Generated: palette=%s | light=%s",
                treatment.get("palette", "—")[:60],
                treatment.get("light_quality", "—")[:60],
            )
        return treatment
    except Exception as e:
        logger.warning("[VisualTreatment] Pass skipped (%s).", e)
        return empty


def _format_visual_treatment_for_prompt(treatment: Dict[str, str]) -> str:
    """Format the visual treatment as a compact vocabulary block for image prompts."""
    if not treatment:
        return ""
    parts = []
    if treatment.get("palette"):
        parts.append(f"FILM PALETTE: {treatment['palette']}")
    if treatment.get("light_quality"):
        parts.append(f"LIGHT SIGNATURE: {treatment['light_quality']}")
    if treatment.get("composition_grammar"):
        parts.append(f"COMPOSITION: {treatment['composition_grammar']}")
    if treatment.get("texture_motif"):
        parts.append(f"TEXTURE MOTIF: {treatment['texture_motif']}")
    if treatment.get("atmosphere"):
        parts.append(f"ATMOSPHERE: {treatment['atmosphere']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pass 3 — Emotional arc map
# ---------------------------------------------------------------------------
_ARC_MAP_PREFIX = (
    "You are a story editor building an emotional arc map for a short animated "
    "film. An emotional arc map is a simple curve: it names the dominant feeling "
    "at each structural phase of the story (opening, rising action, mid-point "
    "shift, late complications, climax, resolution), assigns each phase an "
    "intensity level (1=quiet, 5=maximum), and specifies the visual language "
    "that should accompany that feeling — lighting weight, palette temperature, "
    "camera energy. This map is used to ensure that every shot's visual choices "
    "are calibrated to the story's actual emotional position, not just a generic "
    "interpretation of the shot's mood word."
)


def generate_emotional_arc_map(
    story_idea: str,
    n_shots: int,
    pcfg: "ProjectConfig",
) -> List[Dict]:
    """Build a list of arc-phase dicts, one per structural phase.

    Each dict has: phase, pct_start, pct_end, feeling, intensity (1-5),
    visual_language (lighting / palette / camera keywords).
    Returns [] on failure. Callers index by shot_index / n_shots.
    Uses cached_prefix for Grok cache efficiency.
    """
    if not _HAS_NG or not getattr(pcfg, "emotional_arc_map", True) or n_shots < 2:
        return []

    variable_block = (
        f"STORY:\n{story_idea.strip()[:600]}\n\n"
        f"TOTAL SHOTS: {n_shots}\n\n"
        "Build the emotional arc map. Use 5-7 phases. "
        "pct_start and pct_end are 0.0–1.0 fractions of the total shot count.\n\n"
        "Return ONLY a JSON array:\n"
        '[{"phase": "opening", "pct_start": 0.0, "pct_end": 0.12, '
        '"feeling": "curious unease", "intensity": 2, '
        '"visual_language": "cool desaturated palette, flat diffuse light, wide stable framing"}, ...]'
    )

    try:
        raw = ng.get_openai_prompt_response(
            variable_block,
            temperature=0.4,
            openai_model=getattr(ng, "openai_model_large", None),
            use_grok=getattr(ng, "USE_GROK", True),
            cached_prefix=_ARC_MAP_PREFIX,
        )
        data = ng.parse_json_response(raw)
        if not isinstance(data, list):
            return []
        arc = [
            d for d in data
            if isinstance(d, dict)
            and "pct_start" in d and "pct_end" in d and "feeling" in d
        ]
        if arc:
            logger.info(
                "[ArcMap] %d phases: %s",
                len(arc),
                " → ".join(d.get("phase", "?") for d in arc),
            )
        return arc
    except Exception as e:
        logger.warning("[ArcMap] Pass skipped (%s).", e)
        return []


def _arc_phase_for_shot(arc_map: List[Dict], shot_index: int, n_shots: int) -> Dict:
    """Return the arc-phase dict for this shot, or {} if map is empty."""
    if not arc_map or n_shots < 1:
        return {}
    pct = shot_index / max(1, n_shots - 1)
    for phase in arc_map:
        try:
            if float(phase["pct_start"]) <= pct <= float(phase["pct_end"]):
                return phase
        except (KeyError, TypeError, ValueError):
            continue
    # Clamp: return the last phase if past the end
    return arc_map[-1] if arc_map else {}


# ---------------------------------------------------------------------------
# Pass 4 — Prompt review-and-revise
# ---------------------------------------------------------------------------
_PROMPT_REVIEW_PREFIX = (
    "You are the art director reviewing a batch of image-generation prompts for "
    "a short animated film. Your job is quality control across four dimensions:\n"
    "1. STORY SERVICE: does this shot's visual description actually serve its "
    "narrative moment, or is it generic? A climax shot and an opening shot with "
    "the same mood word should look completely different.\n"
    "2. VISUAL VARIETY: are adjacent shots visually monotonous? If two consecutive "
    "shots have identical framing, identical lighting direction, and identical "
    "colour temperature, flag the second and suggest one specific change.\n"
    "3. LOCK CONSISTENCY: does anything in the prompt contradict the character's "
    "locked appearance, the established setting, or the film's visual treatment?\n"
    "4. IMAGINATIVE DEPTH — REJECT PLAIN SHOTS: flag any prompt whose Environment "
    "reads as a bare, empty, or generic backdrop (e.g. just a location name with no "
    "foreground/midground/background detail), or whose Objects field is empty or "
    "missing, or that would render as a single flat character with nothing else "
    "in frame. This is the single most important check — a technically correct "
    "but plain prompt still fails review. Revise it to add specific, concrete "
    "environmental and object detail consistent with the story world; never "
    "invent detail that contradicts the narrative or the locked setting.\n\n"
    "You return targeted, surgical revisions — change the minimum necessary to fix "
    "the problem. Do not rewrite prompts that already pass all four checks. "
    "Never alter: character gender/build/appearance locks, the style field, "
    "the camera anchoring, mouth-visibility constraints, or hands-away cues."
)


def review_and_revise_image_prompts(
    shots: List["Shot"],
    visual_treatment: Dict[str, str],
    pcfg: "ProjectConfig",
) -> None:
    """Critic pass: review image prompts in batches, revise any that fail QC.

    Modifies shot.image_prompt in-place. Uses cached_prefix so the stable
    art-director instruction block is served from Grok's prefix cache after
    the first batch call, making subsequent batch calls significantly cheaper.
    """
    if not _HAS_NG or not getattr(pcfg, "prompt_review", True):
        return

    batch_size = max(2, int(getattr(pcfg, "prompt_review_batch_size", 6)))
    shots_with_prompts = [s for s in shots if s.image_prompt]
    if not shots_with_prompts:
        return

    treatment_block = _format_visual_treatment_for_prompt(visual_treatment)
    n_shots = len(shots)
    n_revised = 0

    for start in range(0, len(shots_with_prompts), batch_size):
        batch = shots_with_prompts[start:start + batch_size]

        # Build the variable part: per-batch shot list with adjacent context
        shot_blocks = []
        for k, sh in enumerate(batch):
            # Include the previous shot's prompt as adjacent context
            prev_prompt = ""
            if sh.index > 0:
                prev = next(
                    (s for s in shots if s.index == sh.index - 1 and s.image_prompt),
                    None,
                )
                if prev:
                    prev_prompt = prev.image_prompt[:200]
            arc_phase = _arc_phase_for_shot(
                getattr(pcfg, "_arc_map_cache", []), sh.index, n_shots
            )
            arc_info = (
                f"arc: {arc_phase.get('phase', '?')} / {arc_phase.get('feeling', '?')} "
                f"(intensity {arc_phase.get('intensity', '?')})"
                if arc_phase else "arc: unknown"
            )
            shot_blocks.append(
                f"[{k}] shot_index={sh.index} | {arc_info}\n"
                f"  narrative: {(sh.description or sh.setting or '—')[:120]}\n"
                f"  prev_shot_prompt_summary: {prev_prompt[:120] or '—'}\n"
                f"  current_prompt:\n{sh.image_prompt}"
            )

        variable_block = (
            f"FILM VISUAL TREATMENT:\n{treatment_block or '(none)'}\n\n"
            "SHOTS TO REVIEW:\n" + "\n\n".join(shot_blocks) + "\n\n"
            "For each shot, decide: PASS (no changes needed) or REVISE (provide "
            "the corrected prompt).\n\n"
            "Return ONLY a JSON array:\n"
            '[{"i": 0, "verdict": "pass"}, '
            '{"i": 1, "verdict": "revise", "revised_prompt": "full corrected prompt here"}]'
        )

        try:
            raw = ng.get_openai_prompt_response(
                variable_block,
                temperature=0.4,
                openai_model=getattr(ng, "openai_model_large", None),
                use_grok=getattr(ng, "USE_GROK", True),
                cached_prefix=_PROMPT_REVIEW_PREFIX,
            )
            results = ng.parse_json_response(raw)
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                verdict = str(item.get("verdict", "pass")).lower()
                if verdict != "revise":
                    continue
                idx = item.get("i")
                if idx is None or not isinstance(idx, int) or idx >= len(batch):
                    continue
                new_prompt = str(item.get("revised_prompt", "")).strip()
                if new_prompt and new_prompt != batch[idx].image_prompt:
                    batch[idx].image_prompt = new_prompt
                    n_revised += 1
        except Exception as e:
            logger.debug("[PromptReview] Batch %d failed (%s).", start, e)

    if n_revised:
        logger.info("[PromptReview] Revised %d of %d prompts.", n_revised, len(shots_with_prompts))
    else:
        logger.info("[PromptReview] All prompts passed review unchanged.")


# ---------------------------------------------------------------------------
# Pass 5 — Shot-to-shot connective tissue (prev-shot summary)
# ---------------------------------------------------------------------------
_CONTINUITY_SUMMARY_PREFIX = (
    "You are a film editor writing one-line visual summaries of shots for a "
    "continuity sheet. Each summary captures only what the NEXT shot's image "
    "generation needs to know: dominant colour and light direction, where the "
    "subject is in the frame, and any compositional element worth echoing or "
    "deliberately breaking. Keep summaries to one short sentence — they are "
    "injected into the next shot's image prompt as a continuity cue, not "
    "described to a human reader."
)


def build_shot_continuity_summaries(
    shots: List["Shot"],
    pcfg: "ProjectConfig",
) -> None:
    """Generate a one-line continuity summary for each shot and store it on
    shot._prev_summary (a transient attribute set here and read by
    generate_image_prompts when shot_continuity_context is True).

    Batches the summaries in one LLM call per batch. Uses cached_prefix for
    Grok prefix-cache efficiency across batches.
    """
    if not _HAS_NG or not getattr(pcfg, "shot_continuity_context", True):
        return

    shots_with_prompts = [s for s in shots if s.image_prompt]
    if not shots_with_prompts:
        return

    batch_size = 10   # summaries are short — larger batches are fine
    summaries: Dict[int, str] = {}

    for start in range(0, len(shots_with_prompts), batch_size):
        batch = shots_with_prompts[start:start + batch_size]
        items = "\n".join(
            f'[{k}] shot {sh.index}: {sh.image_prompt[:300]}'
            for k, sh in enumerate(batch)
        )
        variable_block = (
            "Write a one-line continuity summary for each shot below.\n\n"
            f"{items}\n\n"
            "Return ONLY a JSON array:\n"
            '[{"i": 0, "summary": "one line here"}, ...]'
        )
        try:
            raw = ng.get_openai_prompt_response(
                variable_block,
                temperature=0.3,
                openai_model=getattr(ng, "openai_model_large", None),
                use_grok=getattr(ng, "USE_GROK", True),
                cached_prefix=_CONTINUITY_SUMMARY_PREFIX,
            )
            results = ng.parse_json_response(raw)
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        idx = item.get("i")
                        s = str(item.get("summary", "")).strip()
                        if isinstance(idx, int) and idx < len(batch) and s:
                            summaries[batch[idx].index] = s
        except Exception as e:
            logger.debug("[Continuity] Batch %d failed (%s).", start, e)

    # Attach summaries as transient attributes (not serialised to plan.json)
    for sh in shots:
        sh._prev_summary = summaries.get(sh.index - 1, "") if sh.index > 0 else ""

    logger.info(
        "[Continuity] Attached prev-shot summaries for %d shots.", len(summaries)
    )


# ── Creative vocabulary pools for image prompt generation ────────────────────
# These give the LLM a concrete palette of specific, evocative choices to draw
# from rather than defaulting to generic mood words. Injected into the prompt
# instruction so the model has reference material, not just abstract direction.
_TEXTURE_VOCAB = (
    "aged leather cracked at the creases, oxidised copper with patina blooms, "
    "rain-darkened concrete with mineral streaks, raw silk catching light at its "
    "weave, weathered oak with visible grain and knot eyes, tarnished brass "
    "fittings, frosted glass diffusing shapes behind it, wet asphalt reflecting "
    "neon in broken shards, rust bleeding through flaking paint, hand-stitched "
    "linen gone soft with wear, polished obsidian, bioluminescent membrane, "
    "matte ceramic glaze, burnished iron"
)
_LIGHT_VOCAB = (
    "caustic light rippling through shallow water, volumetric shafts cutting "
    "through dust, golden-hour raking light that stretches every shadow long, "
    "tungsten warmth against blue-hour exteriors, a single practical lamp "
    "casting hard fall-off, moonlight filtered through cloud layers, sodium "
    "street-lamp orange on wet stone, ember glow from below, rim light that "
    "separates a dark figure from a dark background, flickering fluorescent "
    "catching in the eyes, diffused overcast that kills shadows entirely, "
    "the blue ghost-light of a screen in a dark room"
)
_COMPOSITION_VOCAB = (
    "a figure dwarfed by architecture at the vanishing point, the rule of "
    "thirds with charged negative space on the dominant side, a low-angle that "
    "makes ordinary objects monumental, a frame-within-a-frame through a "
    "doorway or window, leading lines from foreground debris to the distant "
    "subject, a tightly cropped silhouette against a gradient sky, figure "
    "placed at the very edge of frame creating tension with empty space, "
    "layered depth with sharp foreground and blurred middle and sharp background"
)


# Extra guidance folded into the image-prompt instruction when expressive_detail
# is on: richer texture, atmosphere/depth, visible emotion, and dramatic framing.
_IMG_EXPRESSIVE_GUIDANCE = (
    "MAKE IT VIVID, IMAGINATIVE, AND DRAMATIC (without contradicting anything "
    "above):\n"
    "• SINGLE SUBJECT — CRITICAL: every prompt describes exactly ONE main "
    "subject. Never introduce a second instance of the same character, a "
    "mirror image, a reflection that reads as a duplicate figure, or any "
    "symmetrical arrangement that implies two of the same person. If the shot "
    "is a wide or landscape frame, the subject occupies ONE side or one defined "
    "region (e.g. 'lower-left foreground', 'center-right midground'); the rest "
    "of the frame is filled by the environment, not by repeating the subject.\n"
    "• Environment density: the Environment field MUST name at least three "
    "distinct spatial layers — foreground, midground, and background — each "
    "with specific, concrete elements (named objects, architectural features, "
    "atmospheric conditions, flora, debris, or props). A densely described "
    "environment leaves no empty space for the image model to fill by "
    "hallucinating a second figure. Sparse descriptions like 'dark background' "
    "or 'urban setting' are forbidden — replace them with specific detail.\n"
    "• Camera anchoring: the Camera field must state where in the frame the "
    "subject is positioned (e.g. 'subject anchored left-of-center', 'figure in "
    "lower-right foreground'). For wide and landscape shots this is mandatory — "
    "it pins the subject spatially and prevents the model from mirroring them "
    "across the frame.\n"
    "• Texture & material: every surface in the image has a specific material "
    "quality. Name it. Reach for the precise word — not 'rough wall' but "
    "'rain-darkened concrete with mineral streaks', not 'old jacket' but "
    "'aged leather cracked at the collar'. Draw from this palette when it fits: "
    f"{_TEXTURE_VOCAB}.\n"
    "• Light quality: name the exact quality, direction, and color temperature "
    "of every light source. Not 'dramatic lighting' but 'a single practical "
    "lamp casting a hard cone downward, leaving the upper face in shadow, warm "
    "tungsten against a cold blue window'. Draw from this palette: "
    f"{_LIGHT_VOCAB}.\n"
    "• Composition: choose one strong compositional idea that serves the "
    "emotional moment — don't center everything by default. Draw from: "
    f"{_COMPOSITION_VOCAB}.\n"
    "• Narrative specificity: every detail in the frame should belong to THIS "
    "story moment, not a generic version of it. A detective's desk has a "
    "specific case file, a specific cold coffee, a specific map with pins — "
    "not 'a cluttered desk'. A character's clothing has specific wear patterns "
    "that reveal their history. Small, telling props make images feel inhabited.\n"
    "• Visible emotion: render the character's feeling in the FACE and BODY. "
    "When an 'intended expression' is given, work it into the Subject/Action "
    "fields so the emotion is unmistakable — expressive and readable, not "
    "generic or neutral.\n"
    "• Atmosphere & depth: use atmosphere — haze, dust motes, moisture in the "
    "air, volumetric light, shallow depth of field — to give the shot physical "
    "presence. The viewer should feel the temperature and humidity of the space.\n"
    "• Imagination: a bold, specific, surprising choice that still serves the "
    "scene is always better than a competent cliché. Ask: what is the single "
    "most visually striking way to show this moment?\n\n"
)


def generate_image_prompts(shots: List[Shot], characters: List["Character"],
                           theme: str, batch_size: int = 5,
                           cinematic: bool = True, expressive: bool = True,
                           visual_treatment: Optional[Dict[str, str]] = None,
                           arc_map: Optional[List[Dict]] = None,
                           use_continuity_context: bool = True) -> None:
    """LLM-author a structured KLEIN2 prompt per shot, in the labelled format.

    Batches several shots per call for speed/cost. Injects each on-screen
    character's locked gender + build + appearance so Subject/Clothing stay
    consistent shot-to-shot. When ``cinematic`` is on, each shot also carries
    an intended lighting + camera cue mapped from its emotion. When
    ``expressive`` is on, each shot also carries an intended expression cue,
    a narrative arc position, and a distilled story-moment line. When
    ``visual_treatment`` is provided, the film's authored palette/light/texture
    vocabulary is woven into every prompt. When ``arc_map`` is provided, each
    shot's arc phase drives its visual language. When ``use_continuity_context``
    is True, the previous shot's continuity summary is included as a cue.

    Uses cached_prefix so the stable instruction scaffold (identical across all
    batches in a run) is served from Grok's prefix cache after the first call.
    """
    idx = _char_lookup(characters)
    n_shots = max(1, len(shots))
    arc_map = arc_map or []
    visual_treatment = visual_treatment or {}
    treatment_block = _format_visual_treatment_for_prompt(visual_treatment)

    def char_brief(nm: str) -> str:
        ch = idx.get(nm) or idx.get(nm.split()[0] if nm else "")
        if not ch:
            return nm
        bits = [getattr(ch, a, "") for a in ("gender", "physical_build", "appearance")]
        return f"{nm}: " + ", ".join(b for b in bits if b)

    def _arc_label(i: int) -> str:
        """Fallback arc label when no arc_map is available."""
        pct = i / n_shots
        if pct < 0.12:  return "opening / world-establishment"
        if pct < 0.30:  return "early rising action"
        if pct < 0.55:  return "mid-story complication"
        if pct < 0.75:  return "late rising action / crisis build"
        if pct < 0.90:  return "climax / turning point"
        return "resolution / denouement"

    def _arc_info(sh: Shot) -> str:
        """Rich arc info from the arc map, or a fallback label."""
        if arc_map:
            phase = _arc_phase_for_shot(arc_map, sh.index, n_shots)
            if phase:
                vl = phase.get("visual_language", "")
                return (
                    f"{phase.get('phase','?')} / {phase.get('feeling','?')} "
                    f"(intensity {phase.get('intensity','?')})"
                    + (f" | visual language: {vl}" if vl else "")
                )
        return _arc_label(sh.index)

    def _story_moment(sh: Shot) -> str:
        parts = []
        if sh.description and sh.description.strip():
            parts.append(sh.description.strip().rstrip("."))
        added_line = False
        for ln in (sh.lines or []):
            if ln.text.strip() and ln.speaker.upper() != "NARRATOR":
                snippet = ln.text.strip()[:80].rstrip(".,!?")
                parts.append(f'({ln.speaker} says: "{snippet}…")')
                added_line = True
                break
        if not added_line and sh.action_sequence and sh.action_sequence.strip():
            parts.append(sh.action_sequence.strip().rstrip("."))
        return "; ".join(parts)[:200] if parts else ""

    # ── Stable cached prefix — identical across all batches in this run ──────
    # Grok's automatic prefix caching serves this block at a discount after the
    # first call in a conv-id session, making subsequent batch calls cheaper.
    _cached_prefix = (
        "You write image-generation prompts in a STRICT labelled-section format. "
        "IMPORTANT: image prompts describe ONLY the initial visible still image. "
        "Never include spoken words, quoted dialogue, captions, subtitles, text overlays, "
        "speech bubbles, dialogue bubbles, labels, title cards, or any wording that should "
        "appear in the image. The Action field means visible pose/physical action only, "
        "not what anyone says. For dialogue shots, the first non-narrator speaking character "
        "must be the clear visible subject of the image; narrator-only shots "
        "may be cinematic cutaways with no face. "
        "MOUTH VISIBILITY: for any shot with a speaking character, that "
        "speaker's mouth, jaw and lower face MUST be fully visible and "
        "unobstructed — never put a hand, finger, food, drink, cup, "
        "cigarette, microphone, mask, or any object over or in front of the "
        "mouth/chin/face, and the Action field must not describe eating, "
        "drinking, smoking, biting, or covering/touching the mouth. "
        "NO MOUTH OR SPEECH MOTION IN THE ACTION FIELD: this pipeline applies "
        "lip-sync to the audio after the animation is generated. Never write 'speaks', "
        "'talks', 'sings', 'mouth moving', 'jaw shifting', or any synonym in "
        "the Action field; describe only body pose, gesture, and physical action. "
        "SINGLE SUBJECT — CRITICAL: every prompt describes exactly ONE main subject. "
        "Never introduce a mirror image, a reflection that reads as a duplicate figure, "
        "or any symmetrical arrangement implying two of the same person. "
        "ENVIRONMENT DENSITY — REQUIRED: name at least three distinct spatial layers "
        "(foreground, midground, background) with specific concrete elements in each. "
        "CAMERA ANCHORING — REQUIRED: state the subject's position in the frame. "
        "NARRATIVE SPECIFICITY: visual choices must be specific to THAT beat of the story. "
        "An opening shot and a climax shot with the same mood word should look completely different. "
        "FILM VISUAL TREATMENT — honour the film's authored palette, light, and texture "
        "vocabulary in every shot. "
        "Keep named characters' gender/build/appearance EXACTLY as given. "
        "STYLE CONSISTENCY: reuse the global style wording verbatim in every 'style' field. "
        "Be specific, vivid, and surprising.\n\n"
        + (_IMG_EXPRESSIVE_GUIDANCE if expressive else "")
        + "FORMAT (study these examples):\n" + _STRUCT_FEWSHOT
    )

    for start in range(0, len(shots), batch_size):
        batch = shots[start:start + batch_size]
        blocks = []
        for k, sh in enumerate(batch):
            people = "; ".join(char_brief(n) for n in sh.characters_in_frame[:3]) or "(none)"
            cue = ""
            if cinematic:
                c = _cinematic_cues(sh)
                look = c["lighting"] or "(let the global style lead)"
                cue = f"\n    intended look: {look} | camera: {c['camera']}"
            expr = _expression_cue(sh) if (expressive and _shot_has_visible_people(sh)) else ""
            expr_line = f"\n    intended expression: {expr}" if expr else ""
            arc_line = f"\n    narrative arc: {_arc_info(sh)}" if expressive else ""
            moment = _story_moment(sh) if expressive else ""
            moment_line = f"\n    story moment: {moment}" if moment else ""
            # Shot-to-shot continuity: include the previous shot's summary when available
            prev_summary = getattr(sh, "_prev_summary", "") if use_continuity_context else ""
            continuity_line = f"\n    prev shot visual summary: {prev_summary}" if prev_summary else ""
            visual_desc = _visual_safe_description(sh)
            blocks.append(
                f"[{k}] visual description: {visual_desc or '—'}\n"
                f"    on-screen: {people}\n"
                f"    setting: {sh.setting or '—'} | shot: {sh.composition} | mood: {sh.mood or '—'}"
                f"{cue}{expr_line}{arc_line}{moment_line}{continuity_line}"
            )

        # Variable part per batch: treatment + style + shots
        variable_part = (
            f"\n\nFILM VISUAL TREATMENT (honour in every shot):\n{treatment_block}\n\n"
            f"GLOBAL STYLE to reuse verbatim in every 'style' field: {theme or 'cinematic, filmic'}\n\n"
            "SHOTS:\n" + "\n".join(blocks) + "\n\n"
            "Return ONLY a JSON array, one object per shot:\n"
            '[{"i":0,"summary":"<one line>","subject":"...","clothing":"...",'
            '"action":"...","environment":"...","objects":"...","lighting":"...",'
            '"camera":"...","style":"..."}]  '
            "Omit a field by leaving it an empty string. No prose outside the JSON."
        )
        data = None
        if _HAS_NG:
            # Up to two attempts: a transient network hiccup or a single
            # malformed-JSON response used to silently degrade the WHOLE
            # batch to the plain emergency fallback (see build_image_prompt).
            # One retry with a stricter reminder recovers the large majority
            # of those cases before we ever fall back.
            for attempt in range(2):
                try:
                    # cached_prefix holds the stable instruction scaffold
                    # (identical across all batches); variable_part holds only
                    # what changes per batch. Grok serves the prefix from
                    # cache after the first call.
                    _part = variable_part
                    if attempt == 1:
                        _part += (
                            "\n\nREMINDER: return ONLY the JSON array, no prose, "
                            "no markdown fences, one object per shot index."
                        )
                    raw = ng.get_openai_prompt_response(
                        _part,
                        temperature=0.88 if attempt == 0 else 0.6,
                        openai_model=getattr(ng, "openai_model_large", None),
                        use_grok=getattr(ng, "USE_GROK", True),
                        cached_prefix=_cached_prefix,
                    )
                    parsed = ng.parse_json_response(raw)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("shots") or parsed.get("prompts") or []
                    if isinstance(parsed, list) and parsed:
                        data = parsed
                        break
                except Exception as e:
                    if attempt == 0:
                        logger.debug(
                            "  image-prompt batch (shots %s) attempt 1 failed "
                            "(%s) — retrying once.",
                            [s.index for s in batch], e,
                        )
                    else:
                        logger.warning(
                            "  [PROMPTS] image-prompt batch (shots %s) failed "
                            "after retry (%s) — these shots will use the "
                            "plain emergency fallback builder. If this "
                            "happens often, check API rate limits / JSON "
                            "parsing for this model.",
                            [s.index for s in batch], e,
                        )
        by_i = {int(o.get("i", -1)): o for o in (data or []) if isinstance(o, dict)}
        _fell_back = []
        for k, sh in enumerate(batch):
            o = by_i.get(k)
            if o:
                sh.image_prompt = _render_structured_prompt(o.get("summary", ""), o)
            if not sh.image_prompt:
                sh.image_prompt = build_image_prompt(sh, idx, theme)
                _fell_back.append(sh.index)
            sh.image_prompt = _finalize_image_prompt(sh, sh.image_prompt, theme)
        if _fell_back:
            logger.warning(
                "  [PROMPTS] shots %s used the plain emergency fallback "
                "builder (no LLM-authored prompt available for these).",
                _fell_back,
            )
    logger.info("[PROMPTS] structured image prompts ready for %d shots.", len(shots))
    _ensure_prompt_richness(shots, theme, visual_treatment)


# ---------------------------------------------------------------------------
# Deterministic richness gate — catches "single character, plain background"
# ---------------------------------------------------------------------------
# review_and_revise_image_prompts() (below) is a pure LLM-judgment critic pass:
# it can miss things, silently no-ops on a batch failure, and costs an LLM call
# per batch. This gate is the cheap, zero-LLM-cost, deterministic backstop that
# runs on EVERY shot right after prompts are authored — whether they came from
# the rich LLM-authored path or the plain emergency fallback — and repairs the
# two structural tells of a plain image: a missing/short/unlayered Environment
# section, and an empty Objects section.
_RICHNESS_MIN_ENV_WORDS = 12
_RICHNESS_ENV_LAYER_WORDS = ("foreground", "midground", "background", "behind", "surrounding", "beyond")
_STRUCT_LABEL_KEYS = ("subject", "clothing", "action", "pose & placement", "pose",
                     "environment", "objects", "lighting", "camera", "style", "style details")


def _prompt_sections(prompt: str) -> Dict[str, str]:
    """Parse a labelled structured prompt into {lower_case_label: body_text}."""
    out: Dict[str, str] = {}
    for ln in (prompt or "").split("\n"):
        if ":" not in ln:
            continue
        label, _, body = ln.partition(":")
        label = label.strip().lower()
        if label in _STRUCT_LABEL_KEYS:
            out[label] = body.strip()
    return out


def _is_environment_thin(env_text: str) -> bool:
    """True if an Environment section is missing, short, or has no spatial
    layering language — the direct structural signature of a plain backdrop."""
    if not env_text or not env_text.strip():
        return True
    if len(env_text.split()) < _RICHNESS_MIN_ENV_WORDS:
        return True
    low = env_text.lower()
    return not any(w in low for w in _RICHNESS_ENV_LAYER_WORDS)


def _replace_or_append_section(prompt: str, label: str, body: str) -> str:
    """Set a labelled section's body, replacing it in place if present or
    appending a new line if absent. Preserves every other line untouched."""
    pattern = re.compile(rf"(?im)^{re.escape(label)}\s*:.*$")
    if pattern.search(prompt):
        return pattern.sub(f"{label}: {body}", prompt, count=1)
    return prompt.rstrip() + f"\n{label}: {body}"


def _ensure_prompt_richness(shots: List["Shot"], theme: str,
                            visual_treatment: Optional[Dict[str, str]] = None) -> None:
    """Deterministic quality gate: run once, right after every image prompt is
    authored (LLM-authored or emergency-fallback alike), before the LLM critic
    pass. Directly targets the "single character, plain background" failure
    mode with zero additional LLM calls:

      1. Environment thin/missing → layer it with foreground/midground/
         background texture (reuses `_fallback_environment_block`, merging
         with any real content already present rather than discarding it).
      2. Objects empty → fill with a concrete, setting-appropriate filler
         line (pulls the film's texture motif from the visual treatment when
         available) so a panel is never rendered with zero named props.
      3. When comic_book_generator is importable, also runs its own
         lighting/colour/camera/texture/mood-coda gap-filling
         (`_enrich_prompt_with_few_shot_patterns`) on every prompt — the same
         "art director" enrichment logic the comic-panel pipeline uses,
         reused here rather than reimplemented.

    Safe to call on an empty or partially-authored shot list; skips any shot
    with no image_prompt at all (nothing to enrich yet).
    """
    if not shots:
        return
    treatment = visual_treatment or {}
    n_enriched = 0
    for sh in shots:
        prompt = sh.image_prompt or ""
        if not prompt.strip():
            continue
        sections = _prompt_sections(prompt)
        env = sections.get("environment", "")
        objects = sections.get("objects", "")
        changed = False

        if _is_environment_thin(env):
            layered = _fallback_environment_block(sh, theme)
            merged = (
                f"{env}. {layered}"
                if env and env.strip().rstrip('.').lower() not in layered.lower()
                else layered
            )
            prompt = _replace_or_append_section(prompt, "Environment", merged)
            changed = True

        if not objects.strip():
            filler = (
                "specific, tangible props consistent with the setting are "
                "visible nearby, grounding the scene in physical detail."
            )
            motif = treatment.get("texture_motif", "")
            if motif:
                filler = f"{motif}; {filler}"
            prompt = _replace_or_append_section(prompt, "Objects", filler)
            changed = True

        if changed:
            sh.image_prompt = prompt
            n_enriched += 1

        # Reuse the comic pipeline's own art-director gap-filling (lighting /
        # colour / camera / texture / mood-coda) — same logic, no duplication.
        if _HAS_CBG and hasattr(cbg, "_enrich_prompt_with_few_shot_patterns"):
            try:
                panel_like = {
                    "composition": sh.composition or "",
                    "mood": sh.mood or "",
                    "characters_in_frame": sh.characters_in_frame or [],
                    "_arc_emotion": sh.mood or "",
                    "_arc_intensity": 0.5,
                }
                is_cosmic = not sh.characters_in_frame
                sh.image_prompt = cbg._enrich_prompt_with_few_shot_patterns(
                    sh.image_prompt, panel_like,
                    grammar_label=theme or "", is_cosmic=is_cosmic,
                )
            except Exception as e:
                logger.debug(
                    "[Richness] cbg enrichment skipped for shot %s (%s).",
                    sh.index, e,
                )

    if n_enriched:
        logger.info(
            "[Richness] Layered environment and/or filled objects for %d/%d "
            "shot prompts (were thin, unlayered, or missing).",
            n_enriched, len(shots),
        )


def _clip_prompt(text: str, max_chars: int) -> str:
    """Trim a prompt to a safe length on a sentence/word boundary (no mid-word cut)."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in (". ", "; ", ", ", " "):          # prefer a clean break
        i = cut.rfind(sep)
        if i >= int(max_chars * 0.6):
            return cut[:i + (len(sep) - 1)].rstrip(" ,;")
    return cut.rstrip()


def _lipsync_face_safety_instruction(shot: "Shot") -> str:
    """Framing constraint for the STILL IMAGE PROMPT of a dialogue shot.

    Returns a brief Subject-level note describing the desired face framing so
    the diffusion model generates a detectable, unobstructed face for LatentSync
    to lock onto. Phrased as a visible physical description (front-facing,
    medium close-up, lower face in view) rather than a technical instruction
    block — it belongs in the Subject field, not as a separate paragraph.

    IMPORTANT — why this is phrased affirmatively (no "no …" clauses):
      * The default image engine (Z-Image-Turbo, guidance_scale=0.0) and the
        default video engine (FramePack, real CFG=1.0) BOTH run classifier-free
        guidance disabled, so the negative prompt is entirely ignored on the
        default path. Anti-occlusion intent that lives only in the negative
        prompt does nothing for this pipeline's out-of-the-box config.
      * Diffusion/video text encoders (T5/CLIP/LLaMA) handle negation poorly:
        a positive-prompt phrase like "no hands covering the mouth" frequently
        INDUCES hands near the mouth (the classic pink-elephant failure).
    So we describe the desired state directly — full lower face visible, medium
    close-up — which is what actually steers a CFG-free positive-only model.

    NOT used in motion prompts (see _lipsync_motion_framing_cue for the
    motion-prompt equivalent).
    """
    if not getattr(shot, "is_dialogue", False):
        return ""
    if not _shot_has_visible_people(shot):
        return ""
    speaker = shot.speaking_character or "the speaking character"
    return (
        f"{speaker} front-facing or three-quarter view, medium close-up, "
        "full mouth and jaw clearly visible and in sharp focus."
    )


def _lipsync_motion_framing_cue(shot: "Shot") -> str:
    """Framing constraint for the MOTION PROMPT of a dialogue shot.

    Keeps the face in frame and hands down during animation — but deliberately
    says nothing about the mouth moving or the character speaking. Lip-sync
    (LatentSync) replaces the mouth region in a separate pass; any mouth motion
    generated by the animation model competes with that replacement and degrades
    the final result. This cue is strictly about keeping the face visible and
    the hands away — nothing more.
    """
    if not getattr(shot, "is_dialogue", False):
        return ""
    if not _shot_has_visible_people(shot):
        return ""
    speaker = shot.speaking_character or "the character"
    return (
        f"{speaker}'s face stays clearly in frame, front-facing or three-quarter "
        "view; both hands remain low and away from the face throughout."
    )


# Few-shot style references for the action-beat motion-prompt LLM call —
# real examples of the level of vivid, SPECIFIC physical/expressive detail
# that reads well to a video model: posture, gesture, facial expression,
# camera behavior, environment interaction. Deliberately chosen so none
# involve stairs or coordinated multi-person physical contact (the rooftop
# example's "hand on another person" is a static touch during a held
# conversation, not dynamic coordinated motion — exactly the kind of minimal-
# contact interaction _is_multi_person_contact_risk is meant to steer TOWARD,
# not away from). Ours must be much shorter (~30 words) — these illustrate
# the KIND of specificity to aim for, not the length.
#
# IMPORTANT: none of these examples include any speaking, talking, singing,
# or mouth-movement language. Lip-sync (LatentSync) is applied in a separate
# pass and replaces the mouth region; if the animation model also generates
# moving lips they compete with the lip-sync output and degrade quality.
# All motion prompts — for dialogue shots and non-dialogue alike — describe
# only body motion, gesture, posture, eye/head motion, and camera motion.
_MOTION_PROMPT_FEWSHOTS = """\
- "A man holds an apple and gestures with his free hand, expression animated, \
head tilting slightly as he leans forward with intent."
- "Heavy rain. A man's arms are wide open, head slightly raised, eyes upward, \
expression full of surprise and expectation — as if something important is \
about to happen."
- "On a rooftop, a bald man rests a hand on another person's shoulder, \
expression serious, brow furrowed. Wind is strong; the lens shakes slightly \
and pulls closer — tense, like a movie scene."
- "A man in a suit sits on a sofa, leans forward toward the person across from \
him, jaw set, eyes focused — posture of someone trying hard to persuade."
- "A man lies on a sofa, hands folded on his legs, one foot slowly swaying. \
A lamp flickers. The camera slowly circles, like a movie scene."
- "Selfie POV: a man glides through the sky on a parachute, arms spread wide, \
expression elated, scenery passing around him."
- "A man walks beside railway tracks, arms swinging, gaze distant; a train \
slowly passes beside him."\
"""


def generate_motion_prompts(shots: List[Shot], theme: str, vcfg: VideoConfig,
                            batch_size: int = 6, cinematic: bool = True) -> None:
    """Author a SHORT, motion-focused prompt per shot for the animation engine.

    This is deliberately separate from the (long, detailed) image prompt: video
    models need a crisp statement of what MOVES and how the camera moves, not the
    full scene description — the still already conveys the composition.

    SOURCE PRIORITY for motion content:
      1. shot.action_sequence (non-None, non-empty)
         Pre-extracted by comic_book_generator._extract_action_sequence() —
         already stripped of static-scene language and ready to drive the
         animation model. Used directly, bypassing the LLM re-extraction pass
         for that shot. The cinematic-cue layer (camera move / lipsync safety)
         is still applied on top so all shots get consistent framing.
      2. shot.action_sequence is empty string
         CBG determined this is a purely static panel (no motion verbs found).
         Skip the LLM entirely and fall through to the deterministic fallback.
      3. shot.action_sequence is None
         No CBG extraction ran (user-script mode, old manifest).  The LLM
         motion-extraction pass runs as before.

    Each prompt is one or two short sentences, clamped to motion_prompt_max_chars.
    """
    budget = int(vcfg.motion_prompt_max_chars)

    def _apply_cinematic_and_safety(sh: Shot, base: str) -> str:
        """Append camera cues and face-framing constraint to a base motion string.

        Uses _lipsync_motion_framing_cue (not _lipsync_face_safety_instruction)
        so the motion prompt never tells the animator to move the mouth or speak.
        Lip-sync is handled by LatentSync in a separate pass.
        """
        parts = [base.rstrip(". ")]
        if cinematic:
            c = _cinematic_cues(sh, vcfg)
            parts.append(f"Camera: {c['camera']}")
            parts.append(c["motion"])
        framing = _lipsync_motion_framing_cue(sh)
        if framing:
            parts.append(framing)
        return ". ".join(p for p in parts if p).rstrip(". ") + "."

    def fallback(sh: Shot) -> str:
        if getattr(vcfg, "motion_prompts_respect_no_people_scenes", True) and not _shot_has_visible_people(sh):
            return _nonhuman_motion_prompt(sh, vcfg, cinematic=cinematic)
        desc = _visual_safe_description(sh).strip().rstrip(".")
        # keep only the first clause, then cap to ~20 words so it stays short
        for sep in (". ", "; ", ", "):
            if sep in desc:
                desc = desc.split(sep)[0]
                break
        words = desc.split()
        if len(words) > 20:
            desc = " ".join(words[:20])
        bits = [desc]
        if cinematic:
            c = _cinematic_cues(sh, vcfg)
            bits.append(f"Camera: {c['camera']}")
            bits.append(c["motion"])
        return _clip_prompt(". ".join(b for b in bits if b) + ".", budget)

    # Split into two buckets:
    #   cbg_shots: have a CBG action_sequence (including empty = static)
    #   llm_shots: need the LLM motion-extraction pass
    cbg_shots: List[Shot] = []
    llm_shots: List[Shot] = []
    for sh in shots:
        if sh.action_sequence is not None:
            cbg_shots.append(sh)
        else:
            llm_shots.append(sh)

    # --- Bucket 1: CBG action_sequence is authoritative ---
    for sh in cbg_shots:
        seq = sh.action_sequence.strip()
        if not seq:
            # Static panel — deterministic fallback (no motion to extract)
            motion = fallback(sh)
        elif not _shot_has_visible_people(sh):
            # Non-human scene with an action_sequence: keep the seq but filter
            # any human-language that doesn't belong in a no-people prompt.
            motion = _clip_prompt(_nonhuman_motion_prompt(sh, vcfg, cinematic=cinematic), budget)
        else:
            # Character action — use CBG's pre-extracted motion brief directly.
            motion = _apply_cinematic_and_safety(sh, seq)
            motion = _strip_offimage_elements(sh, motion, vcfg)
            if sh.is_dialogue and sh.speaking_character and getattr(vcfg, "keep_speaker_mouth_visible", True):
                motion = _strip_mouth_occlusion_in_motion(motion, sh)
        # Sanitize speaking/mouth language at plan time so plan.json is clean.
        motion = _sanitize_motion_prompt_for_no_dialogue(sh, motion)
        sh.motion_prompt = _clip_prompt(motion, budget)

    # --- Bucket 2: LLM extraction (no CBG action_sequence) ---
    for start in range(0, len(llm_shots), batch_size):
        batch = llm_shots[start:start + batch_size]
        data = None
        if _HAS_NG:
            blocks = []
            for k, sh in enumerate(batch):
                cam = _cinematic_cues(sh, vcfg)["camera"] if cinematic else ""
                action = ", ".join(_matched_action_verbs(sh)) if _is_action_shot(sh) else ""
                hazard = ("stairs" if vcfg.avoid_complex_motion and _has_stair_motion(sh)
                         else "multi-person contact" if vcfg.avoid_complex_motion
                         and _is_multi_person_contact_risk(sh) else "")
                tag = (" [SIMPLIFY: " + hazard + "]" if hazard else
                      " [ACTION BEAT]" if action else (" [WIDE/DIALOGUE-HEAVY]"
                      if sh.composition == "wide_shot" and sh.is_dialogue else ""))
                visual_desc = _nonhuman_visual_description(sh) if not _shot_has_visible_people(sh) else _visual_safe_description(sh)
                ppl_tag = " [NO PEOPLE]" if not _shot_has_visible_people(sh) else ""
                speak_tag = " [SPEAKING]" if (sh.is_dialogue and sh.speaking_character) else ""
                depicted = _depicted_extract(sh)
                blocks.append(f"[{k}]{tag}{ppl_tag}{speak_tag} visual scene: {visual_desc[:160] if visual_desc else '—'} | "
                              f"depicted in still: {depicted or '—'} | "
                              f"feeling: {sh.mood or '—'} | suggested camera: {cam or '—'}"
                              + (f" | physical action cue: {action}" if action and _shot_has_visible_people(sh) else ""))
            prompt = (
                "For each shot, write a SHORT video-animation prompt (max ~30 "
                "words, one or two sentences). Describe ONLY intended physical "
                "movement and CAMERA movement. Do NOT include spoken words, quoted "
                "dialogue, captions, subtitles, script lines, title cards, speech "
                "bubbles, dialogue bubbles, labels, or text overlays. Do not describe "
                "what anyone says; describe only body motion, facial motion, environmental "
                "motion, and camera motion.\n\n"
                "GLOBAL RULE — NO MOUTH, JAW, OR SPEECH MOTION IN ANY PROMPT:\n"
                "This pipeline applies lip-sync to the audio AFTER the animation is "
                "generated. The lip-sync process (LatentSync) replaces the mouth region "
                "frame-by-frame with phoneme-driven motion derived from the voice track. "
                "If the animation already contains mouth movement, jaw shifting, or any "
                "implication that the character is speaking or singing, the two mouth "
                "signals compete and the lip-sync output degrades visibly. Therefore:\n"
                "  • NEVER write: speaks, talks, sings, whispers, shouts, utters, "
                "mouth moves, mouth opens, jaw shifts, jaw drops, jaw working, lips "
                "forming words, mouth moving naturally, or any synonym.\n"
                "  • NEVER write 'as he/she/they speaks/talks/sings'.\n"
                "  • NEVER write 'mid-speech', 'while speaking', or 'while talking'.\n"
                "  • For ALL shots — dialogue or not — describe only posture, gesture, "
                "head orientation, eye movement, eyebrow expression, breathing, and "
                "body/camera motion. The mouth is handled entirely by the lip-sync pass.\n\n"
                "MATCH THE STILL: animate ONLY what is present in 'depicted in "
                "still' / 'visual scene'. NEVER introduce a person, character, "
                "animal, vehicle (car, train, bus, bike…), or object that is not "
                "already in the image — every subject that moves must already be "
                "visible in the still. Do not invent background action.\n"
                "SPEAKING characters (shots tagged [SPEAKING]): keep the face "
                "clearly in frame, both hands away from the face. Describe only "
                "posture, head tilt, eye focus, and subtle body language — NOT the "
                "mouth or any speech act. Do NOT have them eat, drink, smoke, bite, "
                "or raise a hand or object to the mouth, chin, or face.\n"
                "CRITICAL: if a shot is marked [NO PEOPLE], "
                "the motion prompt must not mention any person, character, face, mouth, "
                "eyes, expression, speaking, singing, walking, gestures, or human action "
                "of any kind. For [NO PEOPLE] shots, describe only environment/object/"
                "lighting/weather/camera motion that is relevant to the image. No style "
                "adjectives, no character back-story.\n\n"
                "For shots tagged [SIMPLIFY: stairs]: current video models "
                "handle stepping motion up/down stairs badly (foot placement, "
                "perspective shift). Do NOT ask for actual stepping — describe "
                "the figure holding a steady, mostly-grounded pose near the "
                "stairs with only subtle weight shifts; let the camera's own "
                "movement carry the sense of progress instead.\n"
                "For shots tagged [SIMPLIFY: multi-person contact]: coordinated "
                "two-person physical contact is the worst failure mode (merged "
                "limbs, interpenetration). Give only ONE person simple, "
                "grounded motion; have any other person stay essentially "
                "still — imply the interaction through proximity/expression, "
                "not coordinated physical movement.\n"
                "For shots tagged [ACTION BEAT]: name the SPECIFIC physical "
                "action (walking, climbing, struggling, reaching, etc.) and how "
                "the character interacts with the environment — favor dynamic, "
                "purposeful movement over generic motion (this tag never "
                "co-occurs with a [SIMPLIFY] tag). Study these for the LEVEL "
                "of vivid, specific physical/expressive detail to aim for — "
                "posture, gesture, facial expression, camera behavior — NOT "
                "their length; yours must stay far shorter (~30 words):\n"
                f"{_MOTION_PROMPT_FEWSHOTS}\n\n"
                "(In the examples above, elements like the apple, the train, or "
                "the flickering lamp are only appropriate when they actually "
                "appear in that shot's still — never add such an element yourself "
                "if it is not in 'depicted in still'.)\n\n"
                "For shots tagged "
                "[WIDE/DIALOGUE-HEAVY]: the face is small in frame, so keep "
                "motion ambient and ATMOSPHERIC (wind, light, distant movement) "
                "rather than focused on any character. Everything else: favor "
                "subtle, physically-plausible motion.\n\n"
                "SHOTS:\n" + "\n".join(blocks) + "\n\n"
                'Return ONLY JSON: [{"i":0,"motion":"<short motion prompt>"}]')
            try:
                raw = ng.get_openai_prompt_response(
                    prompt, temperature=0.6,
                    openai_model=getattr(ng, "openai_model_large", None),
                    use_grok=getattr(ng, "USE_GROK", True))
                data = ng.parse_json_response(raw)
                if isinstance(data, dict):
                    data = data.get("shots") or data.get("prompts") or []
            except Exception as e:
                logger.debug("  motion-prompt batch failed (%s) — using fallback.", e)
        by_i = {int(o.get("i", -1)): o for o in (data or []) if isinstance(o, dict)}
        for k, sh in enumerate(batch):
            o = by_i.get(k)
            motion = (o or {}).get("motion", "").strip() if o else ""
            motion = _clip_prompt(motion, budget) if motion else fallback(sh)
            # Deterministic backstops so the stored prompt already matches the
            # still and keeps speakers lip-syncable, regardless of LLM compliance.
            if _shot_has_visible_people(sh):
                motion = _strip_offimage_elements(sh, motion, vcfg)
                if (sh.is_dialogue and sh.speaking_character
                        and getattr(vcfg, "keep_speaker_mouth_visible", True)):
                    motion = _strip_mouth_occlusion_in_motion(motion, sh)
            # Sanitize speaking/mouth language at plan time so plan.json is clean.
            motion = _sanitize_motion_prompt_for_no_dialogue(sh, motion)
            sh.motion_prompt = _clip_prompt(motion, budget)

    cbg_count = len(cbg_shots)
    llm_count = len(llm_shots)
    logger.info("[PROMPTS] short motion prompts ready for %d shots (≤%d chars) "
                "— %d from CBG action_sequence, %d via LLM extraction.",
                len(shots), budget, cbg_count, llm_count)


# =============================================================================
# CINEMATIC DIRECTION  ·  quality-oriented planning (Phase 1)
# =============================================================================
# These passes shape *how* the film reads — its opening, its pacing, and the
# visual/motion grammar of each shot — before any pixel is rendered. They cost
# nothing at render time and lift perceived production value far more than any
# pixel-level tweak: a gripping cold-open, durations that breathe with the
# emotion, and lighting/lens/camera-move cues mapped from each beat's feeling.

# emotion / mood keyword → energy bucket (drives pacing + camera + lighting)
_HIGH_ENERGY = {"tense", "fear", "afraid", "panic", "anger", "angry", "furious",
                "shock", "shocked", "terror", "urgent", "desperate", "frantic",
                "alarm", "rage", "dread", "threat", "danger", "chaos", "violent"}
_LOW_ENERGY  = {"tender", "intimate", "calm", "sad", "grief", "melancholy", "wistful",
                "somber", "reflective", "longing", "hushed", "quiet", "gentle",
                "weary", "tired", "lonely", "still", "peaceful", "solemn", "numb"}
_WONDER      = {"awe", "wonder", "joy", "hope", "elated", "triumphant", "warm", "love"}
_MYSTERY     = {"mystery", "suspense", "eerie", "ominous", "uneasy", "foreboding", "tense"}

# physical-action vocabulary — drives both composition (action needs room to
# show the body) and motion-prompt specificity (name the actual action
# instead of generic "natural movement").
_ACTION_VERBS = {
    "walk", "walks", "walking", "run", "runs", "running", "sprint", "sprints",
    "climb", "climbs", "climbing", "reach", "reaches", "reaching", "grab",
    "grabs", "grabbing", "turn", "turns", "turning", "push", "pushes",
    "pull", "pulls", "struggle", "struggles", "struggling", "fight", "fights",
    "fighting", "chase", "chases", "chasing", "flee", "flees", "fleeing",
    "stumble", "stumbles", "stumbling", "fall", "falls", "falling", "crawl",
    "crawls", "crawling", "embrace", "embraces", "embracing", "strike",
    "strikes", "throw", "throws", "throwing", "lift", "lifts", "lifting",
    "carry", "carries", "carrying", "drag", "drags", "dragging", "kneel",
    "kneels", "kneeling", "stand", "stands", "standing", "rise", "rises",
    "rising", "approach", "approaches", "approaching", "retreat", "retreats",
    "leap", "leaps", "leaping", "jump", "jumps", "jumping", "swing", "swings",
    "duck", "ducks", "ducking", "lunge", "lunges", "lunging", "wrestle",
    "wrestles", "wrestling", "shove", "shoves", "shoving",
}


def _matched_action_verbs(shot: Shot) -> List[str]:
    # Prefer action_sequence when present — CBG already extracted motion content
    # from the description into it, so scanning description for action verbs
    # after that extraction risks missing them.
    text = (shot.action_sequence if shot.action_sequence is not None
            else f"{shot.description or ''} {shot.mood or ''}").lower()
    words = {w.strip(".,!?;:\"'()") for w in text.split()}
    return sorted(words & _ACTION_VERBS)[:3]


def _is_action_shot(shot: Shot) -> bool:
    # A shot with a non-empty action_sequence is definitively an action shot;
    # an empty action_sequence means CBG found no motion (static panel).
    if shot.action_sequence is not None:
        return bool(shot.action_sequence.strip()) and not shot.is_dialogue
    return (not shot.is_dialogue) and bool(_matched_action_verbs(shot))


# Current video/animation models handle these two patterns badly — stepping
# motion up/down stairs (foot placement + perspective shift over many frames)
# and coordinated multi-person physical contact (limbs merge, bodies
# interpenetrate, motion desyncs between the two figures). Detected and
# steered toward a simpler equivalent rather than asked for outright.
_STAIR_KEYWORDS = {"stairs", "staircase", "stairway", "stairwell", "steps", "stoop"}
_CONTACT_RISK_VERBS = {
    "fight", "fights", "fighting", "wrestle", "wrestles", "wrestling",
    "embrace", "embraces", "embracing", "carry", "carries", "carrying",
    "drag", "drags", "dragging", "strike", "strikes", "shove", "shoves",
    "shoving", "chase", "chases", "chasing", "lift", "lifts", "lifting",
    "grab", "grabs", "grabbing", "push", "pushes", "pull", "pulls",
}


def _has_stair_motion(shot: Shot) -> bool:
    text = f"{shot.description or ''} {shot.mood or ''}".lower()
    words = {w.strip(".,!?;:\"'()") for w in text.split()}
    return bool(words & _STAIR_KEYWORDS)


def _is_multi_person_contact_risk(shot: Shot) -> bool:
    """2+ people in frame AND the action implies coordinated physical
    contact between them — the specific combination current models handle
    worst (one person alone climbing/reaching is comparatively fine)."""
    if len(shot.characters_in_frame) < 2:
        return False
    text = f"{shot.description or ''} {shot.mood or ''}".lower()
    words = {w.strip(".,!?;:\"'()") for w in text.split()}
    return bool(words & _CONTACT_RISK_VERBS)


def _shot_feeling(shot: Shot) -> str:
    words = " ".join([shot.mood or ""] + [l.emotion or "" for l in shot.lines]).lower()
    return words


# Feeling → a concrete, drawable facial-expression + body-language phrase, so a
# character's emotion is visible in the still (a frozen neutral face is the most
# common thing that flattens the drama). Keywords are matched against the shot's
# mood + line emotions; first match wins.
_EXPRESSION_BY_FEELING = [
    (("anger", "angry", "furious", "rage", "fury", "wrath"),
     "jaw clenched, brow furrowed hard, eyes burning, nostrils flared"),
    (("fear", "afraid", "terror", "terrified", "panic", "dread", "alarm"),
     "eyes wide, pupils tight, mouth parted in alarm, body tensed"),
    (("sad", "grief", "sorrow", "melancholy", "heartbroken", "mourning", "numb", "lonely"),
     "eyes glistening, brow drawn upward, mouth downturned, shoulders sunken"),
    (("joy", "elation", "happy", "delight", "triumphant", "elated", "gleeful"),
     "bright open smile, eyes crinkled, chin lifted, whole face radiant"),
    (("awe", "wonder", "amazed", "astonished", "marvel"),
     "eyes wide with wonder, lips slightly parted, face upturned and lit"),
    (("tender", "love", "intimate", "affection", "warm", "fond"),
     "soft gaze, gentle half-smile, relaxed brow, warmth in the eyes"),
    (("determination", "determined", "resolute", "defiant", "resolved", "steely"),
     "steady hard gaze, jaw set, chin level, unwavering"),
    (("shock", "surprise", "surprised", "stunned", "shocked"),
     "eyebrows shot up, eyes round, mouth agape"),
    (("suspense", "tense", "uneasy", "ominous", "foreboding", "wary", "nervous"),
     "guarded eyes, tight mouth, wary micro-tension across the face"),
    (("shame", "guilt", "embarrassed", "ashamed"),
     "gaze cast down, flushed, mouth pressed, head slightly bowed"),
]


def _expression_cue(shot: Shot) -> str:
    """A drawable expression phrase for the shot's dominant feeling ('' if none)."""
    w = _shot_feeling(shot)
    for keys, phrase in _EXPRESSION_BY_FEELING:
        if any(k in w for k in keys):
            return phrase
    return ""


def _energy_of(shot: Shot) -> str:
    w = _shot_feeling(shot)
    if any(k in w for k in _HIGH_ENERGY):
        return "high"
    if any(k in w for k in _LOW_ENERGY):
        return "low"
    return "mid"


_CAMERA_BY_COMP = {
    "extreme_close": "a slow creep closer",
    "close_up":      "a slow push-in",
    "medium_shot":   "subtle parallax with a gentle drift",
    "wide_shot":     "a slow, drifting establishing move",
    "over_shoulder": "a slow arc past the shoulder",
    "dutch_angle":   "an unsettled, faintly swaying frame",
}


def _cinematic_cues(shot: Shot, vcfg: Optional["VideoConfig"] = None) -> Dict[str, str]:
    """Map a shot's feeling + framing to lighting / camera-move / motion cues."""
    w = _shot_feeling(shot)
    energy = _energy_of(shot)
    camera = _CAMERA_BY_COMP.get(shot.composition, "a gentle, steady drift")
    if energy == "high":
        camera = "a tense, deliberate push-in" if "close" in shot.composition else "a slow, creeping move"
    elif energy == "low":
        camera = "an almost-still, breathing hold" if "close" in shot.composition else camera

    if any(k in w for k in ("anger", "angry", "furious", "rage")):
        lighting = "harsh high-contrast light, hard shadows, a warm-red cast"
    elif any(k in w for k in _HIGH_ENERGY):
        lighting = "low-key lighting, hard shadows, a cool desaturated palette"
    elif any(k in w for k in ("tender", "intimate", "love", "warm")):
        lighting = "soft warm light, golden rim light, shallow depth of field"
    elif any(k in w for k in ("sad", "grief", "melancholy", "somber", "lonely", "numb")):
        lighting = "muted overcast light, low saturation, soft falloff"
    elif any(k in w for k in _WONDER):
        lighting = "bright lifted light, a warm atmospheric glow"
    elif any(k in w for k in _MYSTERY):
        lighting = "chiaroscuro, pools of light in darkness, atmospheric haze"
    else:
        lighting = ""

    avoid_complex = vcfg is None or vcfg.avoid_complex_motion
    action_words = _matched_action_verbs(shot)
    if avoid_complex and _has_stair_motion(shot):
        # Current models handle stepping motion up/down stairs badly (foot
        # placement + perspective shift over many frames). Keep the figure
        # essentially grounded near the stairs rather than asking for actual
        # stepping; let the camera carry the sense of movement instead.
        motion = ("the figure holds a steady, mostly-grounded pose at the "
                  "stairs — minimal stepping motion, weight shifting subtly "
                  "rather than full strides; the camera's own slow movement "
                  "carries the sense of progress, physically coherent")
    elif avoid_complex and _is_multi_person_contact_risk(shot):
        # Coordinated two-person physical contact is the worst failure mode
        # (merged limbs, interpenetration, desynced motion). Keep ONE person's
        # motion simple and grounded; treat the other as essentially still —
        # implying the interaction through proximity/expression, not motion.
        motion = ("only the primary figure moves, with simple, grounded, "
                  "single-axis motion; any other person in frame stays "
                  "essentially still — imply the interaction through "
                  "proximity and expression rather than coordinated physical "
                  "contact, physically coherent, no merged or overlapping limbs")
    elif action_words:
        motion = (f"purposeful physical motion — {', '.join(action_words)} — "
                  "full-body engagement with the environment, dynamic but "
                  "physically coherent")
    else:
        motion = ("smooth cinematic motion, natural and subtle movement, "
                  "stable identity, physically plausible")

    # Dialogue shots are lip-synced downstream (Wan-S2V natively, or I2V + an
    # external LatentSync pass). Big camera moves and body motion make the face
    # harder to track and visibly hurt lip-sync, so steer talking shots toward a
    # near-locked frame with subtle head motion and natural micro-expressions.
    # Never override the stair / multi-person-contact safety motion.
    # IMPORTANT: do NOT mention speaking, talking, mouth motion, or lip movement
    # here — LatentSync replaces the mouth in a separate pass and any animated
    # mouth motion in the video competes with it.
    lipsync_friendly = (vcfg is None) or getattr(vcfg, "lipsync_friendly_motion", True)
    is_safety_motion = avoid_complex and (_has_stair_motion(shot)
                                          or _is_multi_person_contact_risk(shot))
    if (lipsync_friendly and getattr(shot, "is_dialogue", False)
            and _shot_has_visible_people(shot)):
        if "close" in shot.composition:
            camera = "an almost-imperceptible slow push-in on a locked-off frame"
        elif shot.composition == "wide_shot":
            camera = "a still, locked-off wide frame with only faint ambient drift"
        else:
            camera = "a steady, near-locked frame with a whisper of handheld life"
        if not is_safety_motion:
            motion = ("the character holds a stable, front-facing pose; subtle "
                      "eyebrow and eyelid movement, natural blinking, slight weight "
                      "shifts — minimal body and camera movement so the face stays "
                      "clearly in frame")
    return {"energy": energy, "camera": camera, "lighting": lighting, "motion": motion}


def shape_pacing(shots: List[Shot], pcfg: ProjectConfig) -> None:
    """Set intentional per-shot durations so the film breathes with its emotion.

    Dialogue shots are left alone — their length is the spoken audio. For the
    rest, high-energy beats get quick, punchy holds and quiet/searching beats
    get longer, lingering ones. Writes Shot.duration_hint (the renderer's
    target), which the user can still override per shot in the plan.
    """
    if not pcfg.shape_pacing:
        return
    BY_ENERGY = {"high": 2.2, "mid": 3.5, "low": 5.0}
    for sh in shots:
        if sh.is_dialogue:                      # spoken length wins; don't fight it
            continue
        if sh.duration_hint:                    # respect an explicit hint
            continue
        secs = BY_ENERGY[_energy_of(sh)]
        if sh.composition in ("wide_shot",):    # establishers hold a touch longer
            secs += 0.8
        sh.duration_hint = round(secs, 2)
    logger.info("[DIRECT] paced %d non-dialogue shots by emotional energy.", len(shots))


def enforce_shot_variety(shots: List[Shot], pcfg: ProjectConfig) -> None:
    """Break up runs of 3+ identical compositions so the cutting has rhythm.

    A wall of medium shots reads flat; varying scale (wide → medium → close)
    gives the edit visual rhythm. Dialogue shots are only swapped among
    face-friendly framings (never pushed to a wide), so lip-sync still has a
    face to work with. Runs before prompts so the new framing flows into them.
    """
    if not pcfg.shot_variety or len(shots) < 3:
        return
    ALL = ["wide_shot", "medium_shot", "close_up", "over_shoulder"]
    FACE = ["close_up", "medium_shot", "over_shoulder"]      # keep a face in frame
    changed = 0
    for i in range(2, len(shots)):
        a, b, c = shots[i - 2].composition, shots[i - 1].composition, shots[i].composition
        if a == b == c:
            palette = FACE if shots[i].is_dialogue else ALL
            for cand in palette:
                if cand != b:
                    shots[i].composition = cand
                    changed += 1
                    break
    if changed:
        logger.info("[DIRECT] varied %d shot(s) to avoid repetitive framing.", changed)


def _shot_word_count(shot: Shot) -> int:
    return sum(len(ln.text.split()) for ln in shot.lines
              if ln.text.strip() and ln.speaker.upper() != "NARRATOR")


def direct_shot_composition(shots: List[Shot], cfg: "ProjectConfig") -> None:
    """Frame each shot for what it actually needs.

      • A short, emotionally-charged single-speaker line → close-up. Drama
        reads best tight on one face.
      • A long dialogue block (>= heavy_dialogue_word_threshold words) →
        wide/landscape framing. Faces are small at that distance, which is
        easier to sustain convincingly over a long take and makes lip-sync
        precision much less load-bearing — exactly the shots where it would
        otherwise be hardest to keep convincing.
      • Physical action (walking, climbing, struggling, ...) → pulled OUT of
        close-up, since you can't see someone's actions in a tight shot on
        their face.

    Only ever nudges composition toward the better choice for the content —
    never overrides a shot already framed the same way for the same reason.
    Runs BEFORE merge_dialogue_shots (so pushing heavy dialogue to wide also
    creates more legitimate merge candidates) and enforce_shot_variety (which
    can then build variety on top of an already content-driven base).
    """
    if not cfg.direct_shot_composition:
        return
    n_drama = n_wide = n_action = 0
    for sh in shots:
        if sh.is_dialogue and sh.lines:
            wc = _shot_word_count(sh)
            if wc >= cfg.heavy_dialogue_word_threshold:
                if sh.composition != "wide_shot":
                    sh.composition = "wide_shot"
                    n_wide += 1
            elif _energy_of(sh) in ("high", "low") and len(sh.characters_in_frame) <= 1:
                if sh.composition not in ("close_up", "extreme_close"):
                    sh.composition = "close_up"
                    n_drama += 1
        elif _is_action_shot(sh) and sh.composition in ("close_up", "extreme_close"):
            sh.composition = "wide_shot"
            n_action += 1
    if n_drama or n_wide or n_action:
        logger.info("[DIRECT] composition: %d→close-up (drama), %d→wide (heavy "
                    "dialogue, lip-sync less load-bearing), %d→wide (action needs room).",
                    n_drama, n_wide, n_action)


def craft_opening_hook(shots: List[Shot], story_idea_str: str,
                       characters: List["Character"], pcfg: ProjectConfig) -> None:
    """Prepend a gripping ~3s cold-open engineered to hook the viewer fast.

    A film loses most drop-off in the first seconds, so we open on the single
    most arresting image of the story — mid-tension, in medias res — chosen to
    provoke an immediate emotional reaction (intrigue, dread, wonder). The hook
    is inserted as shot 0 (hard cut in, no fade) and given its own structured
    image prompt. LLM-authored when available, with a deterministic fallback
    that promotes the story's most charged existing beat.
    """
    if not pcfg.opening_hook or not shots:
        return

    hook: Optional[Shot] = None
    if _HAS_NG:
        try:
            beats = "\n".join(
                f"[{i}] {sh.composition} | {sh.mood or '—'} | {(sh.description or '')[:120]}"
                for i, sh in enumerate(shots[:12]))
            prompt = (
                "You are a film editor choosing a COLD OPEN — the first ~3 seconds "
                "of a short film, designed to grip a viewer instantly and provoke an "
                "immediate emotional reaction (intrigue, dread, wonder, tension). "
                "It should drop us into a striking, specific image mid-moment, NOT a "
                "calm establishing shot, and must not spoil the ending.\n\n"
                f"STORY:\n{story_idea_str[:1200]}\n\n"
                f"EXISTING OPENING BEATS:\n{beats}\n\n"
                "Return ONLY JSON:\n"
                '{"description":"<the striking cold-open image, one vivid sentence>",'
                '"setting":"<where>","composition":"<close_up|extreme_close|medium_shot|'
                'wide_shot|over_shoulder|dutch_angle>","mood":"<the intended feeling>",'
                '"emotional_target":"<the reaction to provoke>"}')
            raw = ng.get_openai_prompt_response(
                prompt, temperature=0.8,
                openai_model=getattr(ng, "openai_model_large", None),
                use_grok=getattr(ng, "USE_GROK", True))
            o = ng.parse_json_response(raw)
            if isinstance(o, dict) and o.get("description"):
                hook = Shot(index=0, description=o.get("description", ""),
                            setting=o.get("setting", "") or shots[0].setting,
                            composition=o.get("composition", "close_up") or "close_up",
                            mood=o.get("mood", "") or "tense, arresting",
                            characters_in_frame=list(shots[0].characters_in_frame[:1]))
                logger.info("[DIRECT] cold-open authored (target: %s).",
                            o.get("emotional_target", "attention"))
        except Exception as e:
            logger.debug("  hook LLM failed (%s) — using fallback.", e)

    if hook is None:
        # Fallback: build a tight, punchy hook from the most emotionally charged
        # of the early beats (prefer a close framing of a high-energy moment).
        scored = sorted(
            shots[:10],
            key=lambda s: (_energy_of(s) == "high", "close" in s.composition,
                           len(s.description or "")),
            reverse=True)
        src = scored[0] if scored else shots[0]
        hook = Shot(index=0,
                    description=(src.description or "A charged, decisive moment.").strip(),
                    setting=src.setting,
                    composition="close_up" if "close" not in src.composition else src.composition,
                    mood=(src.mood or "tense, arresting"),
                    characters_in_frame=list(src.characters_in_frame[:1]))
        logger.info("[DIRECT] cold-open synthesized from the strongest early beat.")

    hook.duration_hint = round(float(pcfg.hook_seconds), 2)   # short, punchy, silent
    shots.insert(0, hook)
    for i, sh in enumerate(shots):                            # renumber
        sh.index = i


class BaseImageGen:
    """Pluggable still-image backend: load() once → generate() per prompt → unload()."""
    name = "base"

    def __init__(self, pcfg: ProjectConfig, vcfg: VideoConfig):
        self.pcfg = pcfg
        self.vcfg = vcfg

    def load(self):
        raise NotImplementedError

    def generate(self, prompt: str, width: int, height: int, seed: int):
        raise NotImplementedError

    def unload(self):
        for attr in ("pipe",):
            if hasattr(self, attr):
                setattr(self, attr, None)
        _free_vram()


class ZImageGen(BaseImageGen):
    """Tongyi Z-Image-Turbo (6B DiT) via diffusers ZImagePipeline. Default.

    Photoreal, strong prompt adherence, bilingual text, ~8–9 steps, CFG-free
    (guidance 0.0). Fits a 4090 easily; we keep it resident on CUDA and only
    fall back to CPU offload on low VRAM. Needs a recent diffusers
    (`pip install git+https://github.com/huggingface/diffusers`).
    """
    name = "zimage"

    def load(self):
        from diffusers import ZImagePipeline
        self.pipe = ZImagePipeline.from_pretrained(
            self.pcfg.zimage_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        if _vram_free_gb() >= 18:                 # 6B model is comfortable on a 4090
            self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()

    def generate(self, prompt, width, height, seed):
        # Z-Image requires width/height divisible by 16.
        W, H = (width // 16) * 16, (height // 16) * 16
        neg = self.pcfg.image_negative or None
        out = self.pipe(
            prompt=prompt, negative_prompt=neg, width=W, height=H,
            num_inference_steps=int(self.pcfg.zimage_steps),
            guidance_scale=float(self.pcfg.zimage_guidance),
            generator=torch.Generator("cuda").manual_seed(int(seed)))
        return out.images[0]


class KLEIN2Gen(BaseImageGen):
    """KLEIN2 routed through novel_generator (the previous default). Option."""
    name = "klein2"

    def load(self):
        if not _HAS_NG:
            raise RuntimeError("novel_generator unavailable — cannot use the KLEIN2 image model.")
        self._prev = getattr(ng, "IMAGE_MODEL", "ZIMAGE")
        ng.IMAGE_MODEL = KLEIN2_IMAGE_MODEL
        self.pipe = ng.load_ImageZ_pipe()

    def generate(self, prompt, width, height, seed):
        W, H = (width // 16) * 16, (height // 16) * 16
        image = ng.gen_ImageZ_image(self.pipe, prompt, height=H, width=W, seed=int(seed),
                                    extra_negative=self.pcfg.image_negative)
        try:
            return image if hasattr(image, "save") else Image.fromarray(np.asarray(image))
        except Exception:
            return image

    def unload(self):
        self.pipe = None
        if _HAS_NG:
            ng.IMAGE_MODEL = getattr(self, "_prev", "ZIMAGE")
        _free_vram()


_IMAGE_REGISTRY = {"zimage": ZImageGen, "klein2": KLEIN2Gen}


def _safe_image_generation_resolution(width: int, height: int, vcfg: "VideoConfig") -> Tuple[int, int]:
    """Return the still-image generation size.

    Uses a 64-pixel multiple by default so freshly generated stills are valid
    constraints. This affects newly generated PNGs only; downstream compose can
    still normalize the final film to vcfg.width/vcfg.height if those differ.

    When ``prefer_ltx2`` is active the still is the seed frame for the first
    LTX-2 generation segment, so it **must** be the same size that
    ``LTX2._resolution()`` will use.  The GPU presets now set ``vcfg.width``
    and ``vcfg.height`` to the exact target generation resolution (e.g. 960×576
    for a 3090, 1280×720 for a 4090), so this branch simply snaps those values
    to the LTX alignment multiple (min 16) and applies the ltx2_max ceiling —
    resulting in the exact preset resolution with no further scaling.
    """
    mult = int(getattr(vcfg, "image_resolution_multiple", 64) or 64)
    mode = getattr(vcfg, "image_resolution_rounding", "ceil")
    max_long = int(getattr(vcfg, "image_max_long_side", 1024) or 0)

    w = _round_to_multiple(width, mult, mode)
    h = _round_to_multiple(height, mult, mode)

    if max_long > 0 and max(w, h) > max_long:
        scale = max_long / float(max(w, h))
        w = _round_to_multiple(max(mult, int(w * scale)), mult, "floor")
        h = _round_to_multiple(max(mult, int(h * scale)), mult, "floor")

    # ── LTX-2 seed-frame alignment ────────────────────────────────────────────
    # When LTX-2 is the active engine (prefer_ltx2=True), the still is the seed
    # frame for the first video segment and must match exactly what
    # LTX2._resolution() will produce.  The GPU presets now write the target
    # generation resolution into vcfg.width/height (e.g. 960×576 for 3090,
    # 1280×720 for 4090) AND into ltx2_max_long/short, so this path just
    # re-snaps those values with the same mult=max(16, ltx2_resolution_multiple)
    # used by LTX2._resolution() and applies the same cap — producing the
    # identical pixel dimensions with no lossy rescaling.
    if getattr(vcfg, "prefer_ltx2", False):
        ltx_mult  = max(16, int(getattr(vcfg, "ltx2_resolution_multiple", 64)))
        ltx_long  = int(getattr(vcfg, "ltx2_max_long",  960))
        ltx_short = int(getattr(vcfg, "ltx2_max_short", 576))
        w = _round_to_multiple(width,  ltx_mult, "nearest")
        h = _round_to_multiple(height, ltx_mult, "nearest")
        w, h = _cap_resolution_for_4090(w, h,
                                        max_long=ltx_long,
                                        max_short=ltx_short,
                                        multiple=ltx_mult)

    return int(w), int(h)


def _image_seed_for(shot: "Shot", pcfg: "ProjectConfig",
                    vcfg: "VideoConfig") -> int:
    """Seed for a shot's still.

    A fixed global VideoConfig.seed (>=0) pins every shot and wins outright.
    Otherwise, when consistent_character_seed is on, all stills whose primary
    subject is the same character share ONE seed keyed on that character's name,
    so the character renders consistently shot-to-shot (the per-shot prompt still
    varies pose/framing/setting). No-character scenic shots fall back to a
    per-shot seed so scenery stays varied.
    """
    if getattr(vcfg, "seed", -1) is not None and getattr(vcfg, "seed", -1) >= 0:
        return int(vcfg.seed)
    if getattr(pcfg, "consistent_character_seed", True):
        primary = (getattr(shot, "speaking_character", None)
                   or (shot.characters_in_frame[0] if shot.characters_in_frame else ""))
        if primary:
            return abs(hash(("char-img", primary.strip().lower()))) % (2**31)
    return abs(hash(("img", shot.index))) % (2**31)


# ---------------------------------------------------------------------------
# Mouth-visibility verification for dialogue stills (deterministic backstop).
#
# NARRATOR SHOTS NEVER ENTER THIS PATH.
# The caller (_gen_to) gates on shot.is_dialogue AND shot.speaking_character.
# Both properties exclude lines whose speaker == "NARRATOR", so narration-only
# shots (voice-over, captions) are never sent here regardless of audio content.
#
# STYLIZED / CARTOON / NON-HUMAN CHARACTERS.
# mediapipe FaceMesh is trained on photos; it handles anime, painted, and
# non-human characters at lower confidence than photoreal faces. We therefore
# run with relaxed thresholds when the theme signals stylized art (or when the
# caller explicitly sets mouth_visibility_stylized_art=True). LatentSync itself
# is trained on cartoons and anime, so if LatentSync can sync it, we should
# accept it — the check is a sanity filter (hands-over-mouth, silhouette,
# back-of-head), not a photorealism gate.
# ---------------------------------------------------------------------------

_FACEMESH = None
_FACEMESH_CONFIDENCE = None   # confidence the current instance was built at
_CV2_FACE = None
_CV2_PROFILE = None
_CV2_TRIED = False
_MOUTHVIS_WARNED = False

# Art-style keywords that trigger the relaxed thresholds automatically.
_STYLIZED_ART_KEYWORDS = frozenset((
    "anime", "cartoon", "animated", "cel shad", "graphic novel", "comic",
    "toon", "painterly", "airbrush", "illustrated", "illustration",
    "watercolor", "oil paint", "stylized", "stylised", "hand drawn",
    "hand-drawn", "sketch", "inked", "manga",
))


def _is_stylized_theme(theme: str) -> bool:
    """True when the project theme string signals non-photorealistic art."""
    low = (theme or "").lower()
    return any(kw in low for kw in _STYLIZED_ART_KEYWORDS)


def _get_facemesh(min_confidence: float = 0.5):
    """Return a mediapipe FaceMesh built at *min_confidence*, rebuilding if the
    confidence changed (e.g. first call was photoreal, now we need stylized).
    Returns None when mediapipe is unavailable.
    """
    global _FACEMESH, _FACEMESH_CONFIDENCE
    if _FACEMESH is not None and _FACEMESH_CONFIDENCE == min_confidence:
        return _FACEMESH
    try:
        import mediapipe as mp  # type: ignore
        # Close the old instance before replacing it.
        if _FACEMESH is not None:
            try:
                _FACEMESH.close()
            except Exception:
                pass
        _FACEMESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence)
        _FACEMESH_CONFIDENCE = min_confidence
    except Exception:
        _FACEMESH = None
        _FACEMESH_CONFIDENCE = None
    return _FACEMESH


def _get_cv2_face():
    """Lazily load OpenCV frontal + profile Haar cascades, or (None, None)."""
    global _CV2_FACE, _CV2_PROFILE, _CV2_TRIED
    if _CV2_TRIED:
        return _CV2_FACE, _CV2_PROFILE
    _CV2_TRIED = True
    try:
        import cv2  # type: ignore
        base = getattr(cv2.data, "haarcascades", "")
        _CV2_FACE = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        _CV2_PROFILE = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if _CV2_FACE.empty():
            _CV2_FACE = None
        if _CV2_PROFILE.empty():
            _CV2_PROFILE = None
    except Exception:
        _CV2_FACE, _CV2_PROFILE = None, None
    return _CV2_FACE, _CV2_PROFILE


def _speaker_mouth_visible(image, shot: "Shot",
                           vcfg: Optional["VideoConfig"] = None,
                           pcfg: Optional["ProjectConfig"] = None) -> Optional[bool]:
    """Best-effort check that a speaking, non-narrator shot's still shows a
    locatable, unobstructed face/mouth region.

    Returns:
      True  — a face with a plausible mouth region is visible (accept the still).
      False — no usable face / mouth found (hand over mouth, silhouette, profile,
              back-of-head) — regenerate with a new seed.
      None  — no detector available; accept the still to avoid an infinite loop.

    Narration gating: callers must only pass shots where shot.is_dialogue is
    True AND shot.speaking_character is not None. Both properties exclude
    NARRATOR lines, so this function never needs to check for narration itself.

    Stylized / cartoon / non-human art:
      mediapipe is trained on photos. On painted, anime, or non-human faces it
      assigns lower confidence scores and may produce degenerate landmark fits
      even for a perfectly clear face. We therefore run with relaxed thresholds
      (lower detection confidence, smaller minimum mouth-span floor, smaller
      minimum Haar face size) when the project theme is stylized or the caller
      explicitly sets mouth_visibility_stylized_art=True. LatentSync is trained
      on cartoons/anime — if LatentSync can sync a face, our check should pass
      it. The goal is catching *genuine* occlusion (hand across mouth, profile
      shot, no character at all), not flagging stylized rendering.
    """
    global _MOUTHVIS_WARNED

    # --- resolve thresholds from config (or caller-supplied vcfg/pcfg) --------
    stylized_flag = getattr(vcfg, "mouth_visibility_stylized_art", "auto")
    if stylized_flag == "auto":
        theme = getattr(pcfg, "theme", "") or ""
        stylized = _is_stylized_theme(theme)
    else:
        stylized = bool(stylized_flag)

    if stylized:
        # Relaxed for painted/cartoon/non-human art styles.
        confidence    = float(getattr(vcfg, "mouth_visibility_min_confidence", 0.25))
        span_frac     = float(getattr(vcfg, "mouth_visibility_min_span_frac",  0.008))
        face_frac     = float(getattr(vcfg, "mouth_visibility_min_face_frac",  0.06))
    else:
        # Tighter defaults for photorealistic art.
        confidence    = 0.50
        span_frac     = 0.020
        face_frac     = 0.10

    # --- convert image to numpy -----------------------------------------------
    try:
        arr = np.asarray(image.convert("RGB")) if hasattr(image, "convert") \
            else np.asarray(image)
    except Exception:
        return None
    if arr is None or arr.ndim != 3:
        return None
    h, w = arr.shape[:2]

    # --- primary: mediapipe FaceMesh ------------------------------------------
    fm = _get_facemesh(confidence)
    if fm is not None:
        try:
            res = fm.process(arr)
        except Exception:
            res = None
        if not res or not getattr(res, "multi_face_landmarks", None):
            # No face detected. For stylized art this is somewhat common even on
            # good frames, so we fall through to the Haar fallback rather than
            # immediately returning False — giving the weaker but broader
            # detector a chance to confirm the face is actually absent.
            if not stylized:
                return False
            # fall through to Haar check below
        else:
            lm = res.multi_face_landmarks[0].landmark
            # Outer-lip landmark indices in the 468-point FaceMesh topology.
            mouth_ids = (61, 291, 0, 17, 13, 14, 78, 308)
            xs, ys = [], []
            for i in mouth_ids:
                if i < len(lm):
                    xs.append(lm[i].x)
                    ys.append(lm[i].y)
            if len(xs) < 4:
                # Not enough landmarks — stylized: fall through to Haar.
                if not stylized:
                    return False
            else:
                # Mouth must sit inside the frame.
                if min(xs) < 0 or max(xs) > 1 or min(ys) < 0 or max(ys) > 1:
                    return False
                # Must have a plausible horizontal span; collapsed = occluded fit.
                span = (max(xs) - min(xs)) * w
                if span < max(2.0, span_frac * w):
                    return False
                return True   # FaceMesh found a plausible, unobstructed mouth

    # --- fallback: OpenCV Haar cascades (presence only, no mouth reasoning) ---
    face_casc, profile_casc = _get_cv2_face()
    if face_casc is not None:
        try:
            import cv2  # type: ignore
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            min_face_px = (max(4, int(face_frac * w)),
                           max(4, int(face_frac * h)))
            found = face_casc.detectMultiScale(gray, 1.05, 3, minSize=min_face_px)
            if len(found) == 0 and profile_casc is not None:
                found = profile_casc.detectMultiScale(gray, 1.05, 3,
                                                      minSize=min_face_px)
            if len(found) > 0:
                return True   # Haar found a face — accept (no occlusion judgement)
            # For stylized art where FaceMesh already failed too, a dual miss is
            # a stronger signal. But we still have uncertainty, so return False
            # (trigger a regeneration attempt) rather than None.
            return False
        except Exception:
            return None

    if not _MOUTHVIS_WARNED:
        _MOUTHVIS_WARNED = True
        logger.info("[MouthVis] neither mediapipe nor opencv is available; "
                    "skipping still-level mouth-visibility checks. "
                    "`pip install mediapipe` to enable regeneration of "
                    "occluded dialogue stills.")
    return None


def generate_stills(shots: List[Shot], characters: List["Character"],
                    pcfg: ProjectConfig, vcfg: VideoConfig, resume: bool = True) -> None:
    """Render one still per shot (+ re-anchor stills) at the video aspect ratio.

    Uses the configured image backend (Z-Image by default, KLEIN2 optional),
    loaded once and unloaded afterward so it never shares VRAM with the video
    models.
    """
    name = (pcfg.image_model or "zimage").lower()
    if name not in _IMAGE_REGISTRY:
        raise ValueError(f"Unknown image_model {pcfg.image_model!r} "
                         f"(choose from {sorted(_IMAGE_REGISTRY)}).")
    backend = _IMAGE_REGISTRY[name](pcfg, vcfg)
    logger.info("[IMAGE] loading %s image model…", backend.name)
    backend.load()

    idx = _char_lookup(characters)
    imdir = Path(pcfg.workdir()) / "images"
    imdir.mkdir(parents=True, exist_ok=True)
    W, H = _safe_image_generation_resolution(vcfg.width, vcfg.height, vcfg)
    if (W, H) != (vcfg.width, vcfg.height):
        if getattr(vcfg, "prefer_ltx2", False):
            ltx_long  = int(getattr(vcfg, "ltx2_max_long",  768))
            ltx_short = int(getattr(vcfg, "ltx2_max_short", 448))
            logger.info(
                "[IMAGE] project resolution %sx%s adjusted to LTX-2 seed-frame size %sx%s "
                "(ltx2_max_long=%s ltx2_max_short=%s) — stills now match video generation resolution.",
                vcfg.width, vcfg.height, W, H, ltx_long, ltx_short,
            )
        else:
            logger.info("[IMAGE] project resolution %sx%s adjusted to 64-safe still size %sx%s.",
                        vcfg.width, vcfg.height, W, H)

    def _render_once(prompt: str, seed: int, path_name: str):
        """Generate a single image with the existing one-shot OOM retry."""
        for attempt in range(2):                # one retry — OOM/transient failures are common
            try:                                # sequentially loading multiple large models on 24GB
                return backend.generate(prompt, W, H, seed)
            except Exception as e:
                if attempt == 0:
                    logger.warning("  %s failed on %s (%s) — clearing VRAM and retrying once.",
                                   backend.name, path_name, e)
                    _free_vram()
                else:
                    logger.warning("  %s failed on %s after retry: %s",
                                   backend.name, path_name, e)
                    return None
        return None

    def _gen_to(path: str, prompt: str, seed: int, shot: "Shot" = None) -> bool:
        if resume and Path(path).exists():
            return True
        # Only dialogue shots with a visible speaker are worth verifying; for
        # everything else a single render is correct (a silhouette/cutaway must
        # NOT be regenerated just because it has no face).
        verify = bool(
            shot is not None
            and getattr(vcfg, "verify_speaker_mouth_visible", True)
            and getattr(shot, "is_dialogue", False)
            and getattr(shot, "speaking_character", None)
            and _shot_has_visible_people(shot)
        )
        max_tries = (int(getattr(vcfg, "mouth_visibility_max_retries", 2)) + 1) if verify else 1
        name = Path(path).name
        image = None
        for t in range(max_tries):
            # Perturb the seed on retries so we get a genuinely different pose
            # (while the first attempt still honours the character-consistent seed).
            s = seed if t == 0 else (seed + t * 104729) % (2**31)
            candidate = _render_once(prompt, s, name)
            if candidate is None:
                continue
            image = candidate                    # keep the most recent success
            if not verify:
                break
            ok = _speaker_mouth_visible(candidate, shot, vcfg=vcfg, pcfg=pcfg)
            if ok is None:                       # no detector installed → accept
                break
            if ok:
                if t > 0:
                    logger.info("  [MouthVis] shot %04d recovered a clear speaker "
                                "mouth on attempt %d.", getattr(shot, "index", -1), t + 1)
                break
            if t < max_tries - 1:
                logger.info("  [MouthVis] shot %04d still has an occluded/absent "
                            "speaker mouth — regenerating with a new seed "
                            "(%d/%d).", getattr(shot, "index", -1), t + 1, max_tries)
            else:
                logger.warning("  [MouthVis] shot %04d: could not get a clear "
                               "speaker mouth after %d tries; keeping best still. "
                               "Lip-sync may skip this shot.",
                               getattr(shot, "index", -1), max_tries)
        if image is None:
            return False
        try:
            image.save(path)
        except AttributeError:
            Image.fromarray(np.asarray(image)).save(path)
        return True

    try:
        for sh in shots:
            prompt = _finalize_image_prompt(
                sh,
                sh.image_prompt or build_image_prompt(sh, idx, pcfg.theme),
                pcfg.theme,
            )
            sh.image_prompt = prompt
            base = _image_seed_for(sh, pcfg, vcfg)
            out = str(imdir / f"shot{sh.index:04d}.png")
            if _gen_to(out, prompt, base, sh):
                sh.image_path = out
            # Re-anchor stills for long takes on non-native engines: fresh
            # generations of the SAME scene (new seed) that begin each chain
            # group after the cap, resetting drift.
            sh.anchor_images = []
            for k in range(int(getattr(sh, "anchor_count", 0) or 0)):
                apath = str(imdir / f"shot{sh.index:04d}_a{k+1}.png")
                if _gen_to(apath, prompt, (base + (k + 1) * 7919) % (2**31), sh):
                    sh.anchor_images.append(apath)
            extra = f" (+{len(sh.anchor_images)} re-anchor)" if sh.anchor_images else ""
            logger.info("  shot %04d image ✓%s", sh.index, extra)
    finally:
        backend.unload()
        logger.info("[IMAGE] %s unloaded.", backend.name)


# =============================================================================
# VIDEO ENGINES  (pluggable; uniform interface)
# =============================================================================
#
#   load()                              -> models onto GPU (offload-aware)
#   animate(shot, prompt, out_path)     -> writes an MP4 (no audio), returns path
#   unload()                            -> free VRAM
#
# Audio is muxed *after* rendering (so we always carry our own TTS track and
# never the model's native audio). Wan2.2-S2V is the exception: it needs the
# audio as a *driver* for lip-sync, so it reads shot.audio_path at render time.
# -----------------------------------------------------------------------------

class BaseVideo:
    name = "base"

    def __init__(self, vcfg: VideoConfig):
        self.vcfg = vcfg
        self.device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self.high_vram = _vram_free_gb() > vcfg.high_vram_threshold_gb

    def load(self):  # pragma: no cover
        raise NotImplementedError

    def animate(self, shot: Shot, prompt: str, out_path: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def unload(self):
        for attr in ("pipe", "pipeline", "transformer", "vae", "text_encoder", "text_encoder_2",
                     "image_encoder", "models"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
        _free_vram()

    # shared helper: how many frames for a given clip length
    def _frames_for(self, seconds: float) -> int:
        return max(self.vcfg.fps, int(round(seconds * self.vcfg.fps)))

    # Some pipelines (Wan2.2's family) require num_frames such that
    # (num_frames - 1) is divisible by this value — a causal-VAE temporal-
    # compression constraint. Left unrounded, the pipeline silently snaps an
    # off-constraint num_frames to the nearest valid one internally, which
    # means the segment's ACTUAL length quietly differs from what we asked
    # for — exactly what we sliced the audio to and told the chain to expect.
    # We replicate that rounding ourselves so we always know the true frame
    # count in advance. 1 = no constraint (most engines).
    frame_align: int = 1

    def _aligned_frames_for(self, seconds: float) -> int:
        raw = self._frames_for(seconds)
        align = max(1, int(self.frame_align))
        if align <= 1:
            return raw
        k = round((raw - 1) / align)
        return max(align + 1, align * k + 1)

    def _num_frames(self, shot: Shot) -> int:
        """Frame count for a shot, derived from _target_seconds so it is always
        consistent with the duration the animate harness will actually request.

        Previously this read shot.duration directly, which could disagree with
        _target_seconds when the WAV file is longer than shot.duration (e.g.
        after the lead/tail silence was enlarged by dialogue_scene_entry_pad_ms
        and the audio was re-synthesized but the Shot object was not refreshed).
        Going through _target_seconds ensures the frame count always matches
        what the engine will actually be driven by: for audio-driven engines,
        the exact audio duration; for the rest, a floor that guarantees video
        is never shorter than the WAV.
        """
        secs = self._target_seconds(shot)
        # Apply the same cap that animate() applies for non-audio-driven engines
        # without audio — keeps _num_frames consistent with the harness.
        if not self.supports_audio_drive and not (
            shot.audio_path and Path(shot.audio_path).exists()
        ):
            secs = min(secs, self.vcfg.max_seconds)
        return self._frames_for(secs)

    def _seed(self, shot: Shot) -> int:
        return self.vcfg.seed if self.vcfg.seed >= 0 else (abs(hash((self.name, shot.index))) % (2**31))

    def _target_seconds(self, shot: Shot) -> float:
        has_audio = bool(shot.audio_path and Path(shot.audio_path).exists())
        audio_dur = 0.0
        if has_audio:
            # Re-reading the WAV here is the single source of truth: we always
            # honour the actual file length, not the stale shot.duration field
            # (the WAV can grow after Shot.duration was last set — e.g. lead/
            # tail silence enlarged by dialogue_scene_entry_pad_ms and the
            # audio re-synthesized without clearing the resume cache).
            audio_dur = _wav_duration(shot.audio_path)
            if audio_dur > 0 and audio_dur > shot.duration + 0.01:
                shot.duration = audio_dur

        # Audio-driven engines (Wan-S2V, LTX-2's a2vid path) generate motion
        # AND lip-sync directly from the waveform — the render length must
        # come EXACTLY from that audio's duration. No flooring against an
        # unrelated planning-time duration_hint/default, and no
        # min_audio_padding safety buffer: that buffer exists so a clip is
        # never shorter than its (separately-muxed) audio on NON-audio-driven
        # engines, but here it would just ask the model to render seconds the
        # audio doesn't cover — extra length that then has to be silently
        # trimmed back off at the per-shot mux step, which is exactly the
        # kind of post-hoc audio/video reconciliation this is meant to avoid.
        # frames_for()/_aligned_frames_for() turn this exact duration into
        # the engine's frame count (duration*fps, snapped to that engine's
        # k*align+1 constraint) — the audio itself is not touched again after
        # this point; it is muxed into the final clip byte-for-byte.
        if self.supports_audio_drive and has_audio and audio_dur > 0:
            return audio_dur

        secs = (shot.duration_hint if shot.duration_hint
                else (shot.duration if shot.duration > 0 else self.vcfg.default_seconds))
        secs = float(secs)
        # Non-audio-driven engines render silent video and audio as
        # independent tracks that only meet at the mux step, so the clip must
        # be AT LEAST the audio's length (plus a small buffer) or that mux
        # would truncate the dialogue. This floor is skipped above because
        # audio-driven engines don't need it: their output is already exactly
        # as long as the audio that drove it.
        if has_audio and audio_dur > 0:
            secs = max(secs, audio_dur + self.vcfg.min_audio_padding)
        return secs

    # Per-engine knob: the comfortable length of a single generation. Long takes
    # override with their native window. The harness never asks an engine to make
    # one clip longer than this.
    @property
    def segment_seconds(self) -> float:
        return float(self.vcfg.segment_seconds)

    # Audio-driven engines (Wan-S2V) override → True so the harness slices the
    # shot's TTS track per segment to keep lip-sync aligned across the chain.
    supports_audio_drive = False

    # Engines whose generator handles long video on its own (FramePack) set this
    # True → exempt from the per-chain segment cap and the re-anchor mechanism.
    native_long_video = False

    # Most engines produce silent video and leave this False.
    preserves_generated_audio = False

    # ── the universal long-take harness ──────────────────────────────────────
    def animate(self, shot: Shot, prompt: str, out_path: str) -> str:
        """Render a (possibly long) take by chaining short segments.

        Within a chain group, each segment starts from the previous segment's
        LAST FRAME, so motion and identity carry across the seam. For engines
        that don't support long video natively, a group is capped at
        ``max_segments_per_chain`` segments; the next group then starts from a
        freshly generated still (shot.anchor_images), resetting drift at the cost
        of a soft cut. FramePack (native_long_video) uses a single group.
        Segments are concatenated and trimmed to the exact target length. Audio
        is muxed later (the full TTS track); Wan-S2V also reads per-segment audio
        slices here to drive lip-sync.

        Frame-count alignment (frame_align): each segment's ideal length is
        first snapped to a valid frame count (see _aligned_frames_for), giving
        an ACTUAL render length that can differ slightly from the ideal one.
        That rounding error is carried forward into the next segment's ask
        (carry), so it's absorbed rather than silently accumulating across the
        chain. Audio is sliced against each segment's ACTUAL length, not the
        ideal one, so within a segment audio and video never desync — only the
        final concat's length trim is what reconciles the whole take against
        the true target.
        """
        # Audio-driven (lip-sync) takes cover the FULL spoken line (uncapped),
        # and _target_seconds() returns that audio's EXACT duration for them —
        # no floor, no buffer, nothing to undercut. For non-audio-driven
        # engines, max_seconds is a soft cap — but only for shots with NO
        # audio at all; _target_seconds() guarantees a floor of
        # audio_len+padding for any shot of theirs that carries audio, and
        # capping here must never undercut that floor (that would silently
        # truncate dialogue/narration when -shortest later muxes the two).
        target = self._target_seconds(shot)
        has_audio = bool(shot.audio_path and Path(shot.audio_path).exists())
        if not self.supports_audio_drive and not has_audio:
            target = min(target, self.vcfg.max_seconds)
        seg_secs, groups = _plan_segments(
            target, self.segment_seconds, self.vcfg.chain_segments,
            self.native_long_video, self.vcfg.max_segments_per_chain)

        work = Path(out_path).parent
        stem = Path(out_path).stem
        seg_paths: List[str] = []
        base_seed = self._seed(shot)
        start_img = Image.open(shot.image_path).convert("RGB")
        gi = 0                                              # global segment index
        cursor = 0.0                                         # actual elapsed seconds rendered
        carry = 0.0                                          # ideal-vs-actual delta, folded into the next ask
        try:
            for grp_idx, gcount in enumerate(groups):
                if grp_idx > 0:
                    # new chain → fresh starting image (pre-rendered re-anchor).
                    anchor = (shot.anchor_images[grp_idx - 1]
                              if grp_idx - 1 < len(shot.anchor_images) else None)
                    if anchor and Path(anchor).exists():
                        start_img = Image.open(anchor).convert("RGB")
                        logger.info("  shot %04d  new starting image for chain %d",
                                    shot.index, grp_idx + 1)
                    # else: no anchor available → continue from last frame.
                for s in range(gcount):
                    ideal = seg_secs + carry
                    frames = self._aligned_frames_for(ideal)
                    actual = frames / float(self.vcfg.fps)
                    carry = ideal - actual          # fold the rounding error into the next ask

                    seg_out = str(work / f"{stem}_seg{gi:02d}.mp4")
                    audio_slice = None
                    if self.supports_audio_drive and shot.audio_path:
                        if self.native_long_video:
                            # The engine's own generate() paces itself against
                            # the WHOLE audio file internally (e.g. Wan2.2-S2V's
                            # num_repeat/infer_frames chunking) — hand it the
                            # original file rather than a sliced copy.
                            audio_slice = shot.audio_path
                        else:
                            audio_slice = _slice_audio(
                                shot.audio_path, cursor, actual,
                                str(work / f"{stem}_seg{gi:02d}.wav"))
                    self._render_segment(start_img, prompt, shot, actual,
                                         base_seed + gi, seg_out, audio_slice)
                    cursor += actual
                    seg_paths.append(seg_out)
                    # last frame → next segment, but ONLY within this group
                    # (the next group restarts from its own anchor image).
                    if s < gcount - 1:
                        nxt = _extract_last_frame(
                            seg_out, str(work / f"{stem}_seg{gi:02d}_last.png"))
                        if nxt:
                            start_img = Image.open(nxt).convert("RGB")
                    gi += 1
            _concat_segments(seg_paths, out_path, self.vcfg.fps,
                             target_seconds=target if gi > 1 else None,
                             keep_audio=self.preserves_generated_audio and not has_audio)
        finally:
            for p in seg_paths:                     # tidy intermediates
                for ext in (p, p.replace(".mp4", ".wav"),
                            p.replace(".mp4", "_last.png")):
                    try:
                        os.remove(ext)
                    except OSError:
                        pass
        return out_path

    # Subclasses implement ONE segment. `image` is a PIL.Image (the start frame),
    # `seconds` the segment length, `audio_path` a per-segment slice or None.
    def _render_segment(self, image, prompt: str, shot: Shot, seconds: float,
                        seed: int, out_path: str, audio_path: Optional[str] = None) -> str:
        raise NotImplementedError


def _round_to_multiple(value: int, multiple: int, mode: str = "ceil") -> int:
    multiple = max(1, int(multiple or 1))
    value = max(multiple, int(value))
    mode = (mode or "ceil").lower()
    if mode == "floor":
        return max(multiple, (value // multiple) * multiple)
    if mode == "nearest":
        return max(multiple, int(round(value / multiple)) * multiple)
    return max(multiple, int(math.ceil(value / multiple)) * multiple)


class Wan22S2VVideo(BaseVideo):
    """Wan2.2-S2V-14B — audio-driven, lip-synced. The dialogue engine.

    Runs through the OFFICIAL local repo (https://github.com/Wan-Video/Wan2.2),
    not diffusers — diffusers ships no S2V pipeline. Everything here is local:
    a cloned repo + a downloaded checkpoint directory on disk, plain GPU
    inference. No cloud/remote calls (the only network calls anywhere in this
    module are to the text LLM for story/prompt authoring).

    Setup: clone the repo and either `pip install -e .` it (so `import wan`
    just works) or set VideoConfig.wan_repo_dir to its path; download the
    s2v-14B checkpoint and point VideoConfig.wan_s2v_ckpt_dir at it.

    The model's own generate() ALREADY handles long-form audio-driven
    generation internally — it chunks by infer_frames and loops via
    num_repeat to cover the whole audio file in one call. So unlike our other
    engines, this one is native_long_video: the harness hands it the WHOLE
    shot (and the whole original audio file, not a slice) in a single call,
    rather than chaining external segments itself.
    """
    name = "wan_s2v"
    supports_audio_drive = True
    native_long_video = True   # generate() paces itself against the full audio internally

    @property
    def segment_seconds(self) -> float:
        # Always exactly one external "segment" — generate() takes the whole
        # shot/audio length itself and chunks internally via infer_frames.
        return max(self.vcfg.max_seconds, self.vcfg.segment_seconds, 36000.0)

    def load(self):
        if self.vcfg.try_sage_attention:
            try:
                import sageattention  # noqa: F401  (registers the kernel; the local
                # Wan repo picks it up itself if it's importable, on builds that support it)
                logger.info("  SageAttention available — using it for Wan.")
            except Exception:
                pass
        import sys as _sys
        # Resolve effective local paths, auto-downloading/cloning into
        # cache_dir if they're missing and auto_download_models is on. Both
        # are one-time local fetches — inference itself never calls out.
        repo_dir = self.vcfg.wan_repo_dir or str(Path(self.vcfg.cache_dir) / "Wan2.2")
        ckpt_dir = self.vcfg.wan_s2v_ckpt_dir or str(Path(self.vcfg.cache_dir) / "Wan2.2-S2V-14B")
        if not (Path(repo_dir) / "wan").is_dir():
            repo_dir = _ensure_local_repo(self.vcfg.wan_repo_url, repo_dir,
                                          enabled=self.vcfg.auto_download_models)
        if repo_dir not in _sys.path:
            _sys.path.insert(0, repo_dir)
        try:
            import wan
            from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
        except ImportError as e:
            # If `wan` itself can't be found at all, it's a path/install issue.
            # If `wan` IS found but a submodule import fails (e.g. a missing
            # dependency like decord/safetensors/torchvision that wan.py
            # imports transitively), Python's error here names exactly what's
            # missing — but a bare `except ImportError: raise RuntimeError(...)`
            # without including str(e) THROWS THAT DETAIL AWAY, leaving only a
            # generic "could not import wan" message that's true but useless
            # for actually fixing it. Always include the real message.
            hint = ("`wan` itself isn't on the import path" if "wan" in str(e).lower()
                   and "no module named 'wan'" in str(e).lower() else
                   "`wan` imported, but one of ITS OWN dependencies failed to "
                   "import (the error above names exactly which one — common "
                   "ones for this repo: decord, safetensors, torchvision, "
                   "flash-attn). Install whatever it names, not `wan` itself")
            raise RuntimeError(
                f"Could not import the local Wan2.2 repo's `wan` package from "
                f"{repo_dir!r}: {e}\n"
                f"  → {hint}.\n"
                f"  If `wan` truly isn't reachable: clone {self.vcfg.wan_repo_url} "
                f"there yourself (or `pip install -e .` it), or check "
                f"VideoConfig.wan_repo_dir."
            ) from e
        ckpt_dir = _ensure_hf_snapshot(self.vcfg.wan_s2v_hf_repo, ckpt_dir,
                                       enabled=self.vcfg.auto_download_models)
        if self.vcfg.wan_s2v_size not in MAX_AREA_CONFIGS:
            logger.warning(
                "  wan_s2v_size=%r isn't a key in this repo's MAX_AREA_CONFIGS "
                "— check wan.configs.SIZE_CONFIGS for the exact valid strings "
                "and set VideoConfig.wan_s2v_size to one of them.",
                self.vcfg.wan_s2v_size)
        self._cfg = WAN_CONFIGS["s2v-14B"]
        self._max_area = MAX_AREA_CONFIGS.get(self.vcfg.wan_s2v_size)
        try:
            from wan.utils.utils import save_video as _save_video
            self._save_video = _save_video
        except Exception:
            self._save_video = None
        with _patch_wan_dit_loading(self.vcfg):
            self.model = wan.WanS2V(
                config=self._cfg,
                checkpoint_dir=ckpt_dir,
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=not self.high_vram,        # keep the T5 text encoder off-GPU when VRAM-tight
                convert_model_dtype=True,
            )
        self._log_dit_quant_status()
        self._quantize_dit()
        self._enable_memory_savers()

    def _log_dit_quant_status(self) -> None:
        """Confirm (don't assume) what state the DiT actually loaded in.

        _patch_wan_dit_loading is best-effort: it can silently fail to apply
        (wrong from_pretrained signature, bitsandbytes missing, this repo
        rejecting quantization_config, etc.), in which case the 14B DiT loads
        unquantized at full bf16 (~28GB) straight onto the GPU — which on a
        24GB card either OOMs outright or, worse, corrupts the CUDA context
        and surfaces later as a confusing unrelated-looking "device not
        ready" error rather than a clean out-of-memory message. Checking
        actual parameter dtypes here — right after construction, before
        anything else runs — means that failure mode gets caught and
        explained immediately instead of via a crash a minute later.
        """
        want_bnb = (self.vcfg.wan_s2v_quantize or "").lower() in ("bnb4", "nf4", "int4")
        dit = getattr(self.model, "noise_model", None) or getattr(self.model, "model", None)
        if dit is None or not hasattr(dit, "parameters"):
            logger.debug("  Wan2.2-S2V: couldn't locate the DiT submodule to "
                        "check its quantization status (checked noise_model, model).")
            return
        total_params = 0
        quant_params = 0
        total_bytes = 0
        for p in dit.parameters():
            n = p.numel()
            total_params += n
            try:
                total_bytes += n * p.element_size()
            except Exception:
                pass
            cls_name = type(p).__name__
            if "Params4bit" in cls_name or "Int8Params" in cls_name or p.dtype in (
                    getattr(torch, "uint8", None), getattr(torch, "int8", None)):
                quant_params += n
        total_gb = total_bytes / (1024 ** 3)
        if want_bnb:
            if quant_params > 0:
                logger.info(
                    "  Wan2.2-S2V DiT confirmed 4-bit (%d/%d params, ~%.1fGB "
                    "resident) — bnb4 quantization took effect.",
                    quant_params, total_params, total_gb)
            else:
                logger.warning(
                    "  Wan2.2-S2V DiT requested bnb4 but NO 4-bit params were "
                    "found after construction (%d params, ~%.1fGB resident — "
                    "expect ~4GB if bnb4 actually applied, ~28GB if it didn't). "
                    "This means the from_pretrained patch did not actually "
                    "take hold (see the patch-status log line just above this "
                    "one for why) — the DiT is sitting on the GPU unquantized "
                    "at full precision. An OOM or a 'device not ready' error "
                    "(corrupted CUDA context from a near-OOM shard copy) is "
                    "likely imminent.", total_params, total_gb)
        else:
            logger.debug("  Wan2.2-S2V DiT loaded (%d params, ~%.1fGB resident, "
                        "quantize=%r).", total_params, total_gb,
                        self.vcfg.wan_s2v_quantize)

    def _enable_memory_savers(self) -> None:
        """Best-effort VAE tiling/slicing and attention-slicing on whatever
        the loaded wan.WanS2V instance actually exposes.

        wan.WanS2V's VAE (Wan2_1_VAE) and DiT (WanModel_S2V) are this repo's
        own custom classes, not diffusers ModelMixin/AutoencoderKL subclasses
        — so there's no guarantee either ships the same enable_vae_tiling()/
        enable_attention_slicing() methods a diffusers pipeline would. We
        probe a handful of plausible method names on each and use whichever
        exists, logging exactly what got enabled (or that nothing did) rather
        than assuming either silently works.
        """
        vae = getattr(self.model, "vae", None)
        if vae is not None:
            for fn in ("enable_tiling", "enable_vae_tiling", "enable_slicing", "enable_vae_slicing"):
                if hasattr(vae, fn):
                    try:
                        getattr(vae, fn)()
                        logger.info("  Wan2.2-S2V vae.%s() enabled.", fn)
                    except Exception as e:
                        logger.debug("  vae.%s() present but failed (%s).", fn, e)

        dit = getattr(self.model, "noise_model", None) or getattr(self.model, "model", None)
        enabled_slicing = False
        if dit is not None:
            for fn in ("enable_attention_slicing", "set_attention_slice"):
                if hasattr(dit, fn):
                    try:
                        getattr(dit, fn)("auto" if fn == "enable_attention_slicing" else 1)
                        logger.info("  Wan2.2-S2V dit.%s() enabled.", fn)
                        enabled_slicing = True
                    except Exception as e:
                        logger.debug("  dit.%s() present but failed (%s).", fn, e)
        if not enabled_slicing:
            logger.debug("  Wan2.2-S2V: no attention-slicing hook found on this "
                        "repo's DiT class — bnb4 quantization already covers most "
                        "of the same memory-saving intent, so this is low priority.")

    def _quantize_dit(self) -> None:
        """Best-effort int8/fp8 weight-only quantization of the S2V DiT
        (torchao), applied AFTER construction. Only runs for
        wan_s2v_quantize in ("int8", "fp8") — "bnb4" is handled earlier, at
        LOAD TIME, by _patch_wan_dit_loading (the model is already quantized
        by the time this would run, so there's nothing to do here for it).

        14B params is ~28GB at bf16 — already over a 4090's 24GB for weights
        alone. The real wan.WanS2V source stores the DiT as `self.noise_model`
        (confirmed from the repo's own __init__:
        ``self.noise_model = WanModel_S2V.from_pretrained(...)``), so that's
        tried first; the older generic guesses (`model`, `dit`, `transformer`,
        `net`, `diffusion_model`) stay as a fallback in case a different repo
        version renames it. If none match your installed repo's actual
        attribute, this logs a clear message — tell me the right name and
        I'll wire it in directly — and the engine still runs, just relying on
        t5_cpu/offload_model instead.
        """
        mode = (self.vcfg.wan_s2v_quantize or "none").lower()
        if mode not in ("int8", "fp8"):
            return
        quantize_ = _torchao_quantize_fn()
        make_cfg = _resolve_torchao_quant(mode)
        if not (quantize_ and make_cfg):
            logger.warning("  Wan2.2-S2V %s quantization unavailable in this torchao "
                           "build — running unquantized (t5_cpu/offload_model still "
                           "apply). Upgrade torchao for int8.", mode)
            return
        candidates = ((self.vcfg.wan_s2v_quantize_attr,) if self.vcfg.wan_s2v_quantize_attr
                     else ()) + ("noise_model", "model", "dit", "transformer",
                                 "net", "diffusion_model")
        for name in candidates:
            mod = getattr(self.model, name, None)
            if isinstance(mod, torch.nn.Module):
                try:
                    quantize_(mod, make_cfg())
                    logger.info("  Wan2.2-S2V '%s' quantized (%s).", name, mode)
                except Exception as e:
                    logger.warning("  Wan2.2-S2V %s quantization of '%s' failed (%s) — "
                                   "running unquantized.", mode, name, e)
                return
        logger.warning(
            "  Could not find the Wan2.2-S2V DiT submodule to quantize (tried "
            "%s on the loaded wan.WanS2V instance) — running unquantized "
            "(t5_cpu/offload_model still apply). If you know the right "
            "attribute name on your installed repo's WanS2V class, set "
            "VideoConfig.wan_s2v_quantize_attr to it.", candidates)

    def _render_segment(self, image, prompt, shot, seconds, seed, out_path, audio_path=None):
        if not audio_path:
            raise RuntimeError("Wan2.2-S2V needs audio to drive generation.")
        img_path = str(Path(out_path).with_suffix(".ref.png"))
        image.save(img_path)
        try:
            num_repeat = int(self.vcfg.wan_s2v_num_clip) or None
            shift = (self.vcfg.wan_s2v_shift if self.vcfg.wan_s2v_shift is not None
                     else self._cfg.sample_shift)
            with warnings.catch_warnings():
                # The repo's own wan/modules/vae2_1.py calls the deprecated
                # torch.cuda.amp.autocast(...) spelling internally — harmless
                # (just an old API spelling, not a real problem), but noisy.
                # Scoped to this call only, so nothing else gets silenced.
                warnings.filterwarnings("ignore", category=FutureWarning,
                                       message=".*torch.cuda.amp.autocast.*")
                video = self.model.generate(
                    input_prompt=prompt,
                    ref_image_path=img_path,
                    audio_path=audio_path,
                    enable_tts=False,
                    # Required positional params in the real
                    # wan.WanS2V.generate() signature — no defaults — even
                    # though they're only READ internally when
                    # enable_tts=True. Omitting them raises TypeError before
                    # generation ever starts.
                    tts_prompt_audio=None,
                    tts_prompt_text=None,
                    tts_text=None,
                    num_repeat=num_repeat,
                    pose_video=None,
                    max_area=self._max_area,
                    infer_frames=int(self.vcfg.wan_s2v_infer_frames),
                    shift=shift,
                    sample_solver="unipc",
                    sampling_steps=int(self.vcfg.wan_s2v_steps),
                    guide_scale=float(self.vcfg.wan_s2v_guidance_scale),
                    n_prompt=self.vcfg.video_negative or "",
                    seed=seed,
                    offload_model=not self.high_vram,
                    init_first_frame=False,
                )
            if self._save_video is not None:
                self._save_video(tensor=video[None], save_file=out_path,
                                 fps=self._cfg.sample_fps, nrow=1,
                                 normalize=True, value_range=(-1, 1))
            else:
                # Fallback if wan.utils.utils.save_video isn't importable:
                # tensor is [C, T, H, W] in [-1, 1] — convert to frames ourselves.
                arr = ((video.clamp(-1, 1) + 1) / 2 * 255).byte()
                arr = arr.permute(1, 2, 3, 0).cpu().numpy()   # T,H,W,C
                from diffusers.utils import export_to_video as _etv
                _etv(list(arr), out_path, fps=self._cfg.sample_fps)
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass
        return out_path

    def unload(self):
        self.model = None
        _free_vram()


class FramePackVideo(BaseVideo):
    """FramePack (HunyuanVideo packed I2V) — your notebook's engine, for long takes.

    This mirrors the generate_video() flow in MVSW_FramePack_ImageZ.ipynb
    (text-encode → VAE-encode the still → CLIP-Vision encode → sectioned
    sampling → decode → MP4). It imports the same `diffusers_helper` package
    your notebook uses, so it must be importable on this machine.

    If you'd rather use your notebook's exact generate_video verbatim, expose it
    as a module and call it from animate(); the interface is identical.
    """
    name = "framepack"
    native_long_video = True                 # internal sectioning handles long takes

    @property
    def segment_seconds(self) -> float:
        # Large window → the harness asks for a single segment and FramePack's own
        # sectioning builds the full-length take (no 3-segment cap, no re-anchor).
        return max(self.vcfg.max_seconds, self.vcfg.segment_seconds)

    def load(self):
        from transformers import (LlamaModel, CLIPTextModel, LlamaTokenizerFast,
                                  CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel)
        from diffusers import AutoencoderKLHunyuanVideo
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

        # Pass torch_dtype into from_pretrained() rather than calling .to(dtype=...)
        # afterward. Both AutoencoderKLHunyuanVideo and HunyuanVideoTransformer3DModelPacked
        # register certain modules to stay in float32; a post-load .to() overrides that
        # and produces the "should be kept in float32" diffusers warning AND can silently
        # produce inconsistent results. from_pretrained(torch_dtype=...) honours those
        # per-module registrations correctly.
        self.text_encoder = LlamaModel.from_pretrained(
            FP_HUNYUAN_ID, subfolder="text_encoder", torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            FP_HUNYUAN_ID, subfolder="text_encoder_2", torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained(FP_HUNYUAN_ID, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(FP_HUNYUAN_ID, subfolder="tokenizer_2")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            FP_HUNYUAN_ID, subfolder="vae",
            torch_dtype=torch.float16).cpu()            # ← dtype in from_pretrained, not .to()
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            FP_REDUX_ID, subfolder="feature_extractor")
        self.image_encoder = SiglipVisionModel.from_pretrained(
            FP_REDUX_ID, subfolder="image_encoder", torch_dtype=torch.float16).cpu()
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            FP_TRANSFORMER,
            torch_dtype=torch.bfloat16).cpu()          # ← dtype in from_pretrained, not .to()

        for m in (self.text_encoder, self.text_encoder_2, self.image_encoder,
                  self.vae, self.transformer):
            m.eval(); m.requires_grad_(False)
        self.transformer.high_quality_fp32_output_for_inference = True

        # Do NOT call blanket .to(dtype=...) on the whole model — use from_pretrained
        # torch_dtype above so float32 sub-modules are preserved. Only set the
        # inference precision flag (already done) on the transformer here.

        if self.high_vram:
            for m in (self.text_encoder, self.text_encoder_2, self.image_encoder,
                      self.vae, self.transformer):
                m.to("cuda")
        else:
            from diffusers_helper.memory import DynamicSwapInstaller
            # VAE slicing + tiling must be enabled BEFORE DynamicSwap installs
            # its hooks, so the tiling state is part of the installed model and
            # the VAE never has to move more than one tile at a time. In the
            # notebook this happens right after the low-VRAM branch is chosen.
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            DynamicSwapInstaller.install_model(self.transformer, device="cuda")
            DynamicSwapInstaller.install_model(self.text_encoder, device="cuda")

    def _render_segment(self, image, prompt, shot, seconds, seed, out_path, audio_path=None):
        import einops
        from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode, vae_decode
        from diffusers_helper.bucket_tools import find_nearest_bucket
        from diffusers_helper.clip_vision import hf_clip_vision_encode
        from diffusers_helper.utils import (crop_or_pad_yield_mask, resize_and_center_crop,
                                            soft_append_bcthw)
        from diffusers_helper.memory import (gpu, unload_complete_models,
                                            load_model_as_complete, fake_diffusers_current_device,
                                            move_model_to_device_with_memory_preservation,
                                            offload_model_from_device_for_memory_preservation)
        from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

        # ---------------------------------------------------------------------------
        # save_bcthw_as_mp4 — local wrapper that falls back to cv2 when the installed
        # torchvision version has removed torchvision.io.write_video (torchvision
        # ≥ 0.21 dropped it in favour of torchcodec, which requires FFmpeg shared
        # libraries that are often absent in WSL2 / conda envs).
        #
        # Input tensor shape: [B, C, T, H, W], float32, values in [0, 1] (or [−1, 1]
        # — diffusers_helper normalises to [0, 1] before calling us).
        # save_bcthw_as_mp4 from diffusers_helper also accepts a crf kwarg; the cv2
        # fallback ignores crf (cv2's VideoWriter quality is controlled by codec
        # defaults, not a CRF slider) and uses mp4v which is always available.
        # ---------------------------------------------------------------------------
        def _try_import_save_bcthw():
            try:
                from diffusers_helper.utils import save_bcthw_as_mp4 as _fn
                # Probe whether the underlying torchvision.io.write_video symbol
                # still exists — if it doesn't, the import succeeds but the first
                # call raises AttributeError at runtime (the exact error we're fixing).
                import torchvision.io as _tvio
                if not hasattr(_tvio, "write_video"):
                    raise AttributeError("torchvision.io.write_video removed")
                return _fn
            except (ImportError, AttributeError):
                return None

        _dh_save = _try_import_save_bcthw()

        def _save_bcthw_as_mp4(tensor: "torch.Tensor", path: str,
                                fps: float = 30, crf: int = 18) -> None:
            """Save a [B, C, T, H, W] float tensor as an MP4.

            Tries diffusers_helper's save_bcthw_as_mp4 first; falls back to a
            cv2-based writer when torchvision.io.write_video is absent.
            """
            if _dh_save is not None:
                try:
                    _dh_save(tensor, path, fps=fps, crf=crf)
                    return
                except AttributeError:
                    pass   # write_video disappeared at runtime — use cv2

            # cv2 fallback
            # tensor: [B, C, T, H, W] float32 in [0, 1]
            # squeeze batch dim → [C, T, H, W], then permute → [T, H, W, C]
            import cv2 as _cv2
            t = tensor
            if t.ndim == 5:
                t = t[0]               # drop batch dim → [C, T, H, W]
            if t.ndim == 4:
                t = t.permute(1, 2, 3, 0)   # [C, T, H, W] → [T, H, W, C]
            # Convert to uint8 in [0, 255]
            if t.dtype != torch.uint8:
                t = (t.clamp(0.0, 1.0) * 255).to(torch.uint8)
            t_np = t.cpu().numpy()
            T_frames, H, W, C = t_np.shape
            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            writer = _cv2.VideoWriter(path, fourcc, float(fps), (W, H))
            try:
                for frame in t_np:
                    if C == 3:
                        frame = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
                    writer.write(frame)
            finally:
                writer.release()
            logger.debug("[FramePack] saved %d frames via cv2 fallback → %s", T_frames, path)

        v = self.vcfg
        total_seconds = float(seconds)
        latent_window = int(max(1, v.fp_latent_window))
        internal_fps = int(max(1, getattr(v, "fp_internal_fps", 30) or 30))
        section_units = (total_seconds * internal_fps) / float(latent_window * 4)
        rounding = (getattr(v, "fp_section_rounding", "ceil") or "ceil").lower()
        if rounding == "floor":
            total_sections = int(max(math.floor(section_units), 1))
        elif rounding == "round":
            total_sections = int(max(round(section_units), 1))
        else:
            # Ceil avoids under-generating; fp_force_exact_duration trims the
            # overage cleanly after the native FramePack render.
            total_sections = int(max(math.ceil(section_units), 1))
        n_prompt = ""

        logger.info("[FramePack] target %.3fs, native_fps=%d, latent_window=%d, sections=%d.",
                    total_seconds, internal_fps, latent_window, total_sections)

        input_image = np.array(image.convert("RGB"))

        if not self.high_vram:
            unload_complete_models(self.text_encoder, self.text_encoder_2,
                                   self.image_encoder, self.vae, self.transformer)
            fake_diffusers_current_device(self.text_encoder, gpu)
            load_model_as_complete(self.text_encoder_2, target_device=gpu)

        llama_vec, clip_pool = encode_prompt_conds(
            prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
        if v.fp_cfg == 1:
            llama_vec_n = torch.zeros_like(llama_vec); clip_pool_n = torch.zeros_like(clip_pool)
        else:
            llama_vec_n, clip_pool_n = encode_prompt_conds(
                n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)
        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        H, W, _ = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        img_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
        img_pt = img_pt.permute(2, 0, 1)[None, :, None, :, :]

        if not self.high_vram:
            load_model_as_complete(self.vae, target_device=gpu)
        start_latent = vae_encode(img_pt, self.vae)

        if not self.high_vram:
            load_model_as_complete(self.image_encoder, target_device=gpu)
        clip_out = hf_clip_vision_encode(img_np, self.feature_extractor, self.image_encoder)
        clip_hidden = clip_out.last_hidden_state

        dt = self.transformer.dtype
        llama_vec = llama_vec.to(dt); llama_vec_n = llama_vec_n.to(dt)
        clip_pool = clip_pool.to(dt); clip_pool_n = clip_pool_n.to(dt)
        clip_hidden = clip_hidden.to(dt)

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window * 4 - 3
        history_latents = torch.zeros(
            (1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated = 0
        latent_paddings = list(reversed(range(total_sections)))
        if total_sections > 4:
            latent_paddings = [3] + [2] * (total_sections - 3) + [1, 0]

        for pad in latent_paddings:
            is_last = pad == 0
            pad_size = pad * latent_window
            indices = torch.arange(0, sum([1, pad_size, latent_window, 1, 2, 16])).unsqueeze(0)
            (clean_pre, blank, latent_idx, clean_post, clean_2x, clean_4x) = \
                indices.split([1, pad_size, latent_window, 1, 2, 16], dim=1)
            clean_idx = torch.cat([clean_pre, clean_post], dim=1)

            clean_pre_lat = start_latent.to(history_latents)
            clean_post_lat, clean_2x_lat, clean_4x_lat = \
                history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
            clean_lat = torch.cat([clean_pre_lat, clean_post_lat], dim=2)

            if not self.high_vram:
                move_model_to_device_with_memory_preservation(
                    self.transformer, target_device=gpu,
                    preserved_memory_gb=v.fp_gpu_preserve_gb)
            if v.fp_use_teacache:
                self.transformer.initialize_teacache(enable_teacache=True, num_steps=v.fp_steps)
            else:
                self.transformer.initialize_teacache(enable_teacache=False)

            generated = sample_hunyuan(
                transformer=self.transformer, sampler="unipc", width=width, height=height,
                frames=num_frames, real_guidance_scale=v.fp_cfg, distilled_guidance_scale=v.fp_gs,
                guidance_rescale=v.fp_rs, num_inference_steps=v.fp_steps, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_mask, prompt_poolers=clip_pool,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_mask_n,
                negative_prompt_poolers=clip_pool_n, device=gpu, dtype=torch.bfloat16,
                image_embeddings=clip_hidden, latent_indices=latent_idx,
                clean_latents=clean_lat, clean_latent_indices=clean_idx,
                clean_latents_2x=clean_2x_lat, clean_latent_2x_indices=clean_2x,
                clean_latents_4x=clean_4x_lat, clean_latent_4x_indices=clean_4x,
            )
            if is_last:
                generated = torch.cat([start_latent.to(generated), generated], dim=2)
            total_generated += int(generated.shape[2])
            history_latents = torch.cat([generated.to(history_latents), history_latents], dim=2)

            if not self.high_vram:
                offload_model_from_device_for_memory_preservation(
                    self.transformer, target_device=gpu,
                    preserved_memory_gb=getattr(v, "fp_transformer_offload_preserve_gb", 8))
                load_model_as_complete(self.vae, target_device=gpu)

            real = history_latents[:, :, :total_generated]
            if history_pixels is None:
                history_pixels = vae_decode(real, self.vae).cpu()
            else:
                section_lat = min(total_generated, latent_window * 2 + 1)
                cur = vae_decode(real[:, :, :section_lat], self.vae).cpu()
                overlap = latent_window * 4 - 3
                history_pixels = soft_append_bcthw(cur, history_pixels, overlap)

            if not self.high_vram:
                # Synchronize before unloading the VAE so every CUDA kernel
                # launched by vae_decode (including any tiling/slicing kernels)
                # has completed and its results are fully copied to CPU.
                # Skipping this is what causes the "CUDA driver error: device not
                # ready" on the next model swap — the driver is still draining
                # async work when DynamicSwap tears the VAE off the device.
                torch.cuda.synchronize()
                unload_complete_models()

            if getattr(v, "fp_save_section_progress", False):
                progress_dir = getattr(v, "fp_section_progress_dir", "") or str(Path(out_path).parent / "framepack_sections")
                Path(progress_dir).mkdir(parents=True, exist_ok=True)
                save_path = str(Path(progress_dir) / f"{Path(out_path).stem}_section_{total_generated:04d}.mp4")
                _save_bcthw_as_mp4(history_pixels, save_path, fps=internal_fps, crf=v.fp_mp4_crf)
                logger.info("[FramePack] progress section → %s", save_path)

            if is_last:
                break

        native_out = out_path
        tmp_native = None
        if getattr(v, "fp_force_exact_duration", True):
            tmp_native = str(Path(out_path).with_suffix(".framepack_native.mp4"))
            native_out = tmp_native

        # All GPU models are now unloaded. Do a final sync + cache clear before
        # writing to disk so no async CUDA ops are still in-flight — this is what
        # the standalone notebook's reset_memory() provides between shots. Without
        # it the driver context can be "not ready" for subsequent model loads on
        # the next shot even though history_pixels itself is a CPU tensor.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Save at FramePack's native cadence first. The final clip is then
        # normalized to project fps and exact requested duration when enabled.
        _save_bcthw_as_mp4(history_pixels, native_out, fps=internal_fps, crf=v.fp_mp4_crf)

        if tmp_native:
            _normalize_silent_video_duration(tmp_native, out_path, fps=self.vcfg.fps, target_seconds=total_seconds)
            try:
                os.remove(tmp_native)
            except OSError:
                pass

        return out_path


class Wan22I2VVideo(BaseVideo):
    """Wan2.2 TI2V-5B — plain image-to-video (no audio driver), via diffusers.

    Unlike S2V (which has no diffusers pipeline and runs through the local
    Wan-Video/Wan2.2 repo instead), the TI2V-5B image-to-video model IS
    supported natively by diffusers, so this engine stays on that simpler
    path. Default engine for non-dialogue motion.
    """
    name = "wan_i2v"
    supports_audio_drive = False
    frame_align = 4                          # Wan causal VAE: (num_frames - 1) % 4 == 0

    @property
    def segment_seconds(self) -> float:
        return float(self.vcfg.segment_seconds)

    def load(self):
        if self.vcfg.try_sage_attention:
            try:
                import sageattention  # noqa: F401  (registers the kernel)
                logger.info("  SageAttention available — using it for Wan.")
            except Exception:
                pass
        import diffusers
        from diffusers.utils import export_to_video
        self._export = export_to_video
        Pipe = getattr(diffusers, "WanImageToVideoPipeline", None)
        if Pipe is None:
            raise RuntimeError("WanImageToVideoPipeline not in diffusers — pip install -U diffusers.")
        # 4090 (24 GB) defaults: fp8 weights when available, CPU offload into the
        # 64 GB system RAM, and VAE tiling/slicing — set by high_vram_threshold.
        use_fp8 = bool(getattr(self.vcfg, "use_fp8", True))
        dtype = torch.float8_e4m3fn if use_fp8 else torch.bfloat16
        with _quiet_config_mismatch_warnings():
            try:
                self.pipe = Pipe.from_pretrained(WAN22_I2V_MODEL_ID, torch_dtype=dtype)
            except Exception:
                self.pipe = Pipe.from_pretrained(WAN22_I2V_MODEL_ID, torch_dtype=torch.bfloat16)
        if self.high_vram:
            self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()
        v = getattr(self.pipe, "vae", None)
        if v is not None:
            for fn in ("enable_tiling", "enable_slicing"):
                if hasattr(v, fn):
                    getattr(v, fn)()

    def _render_segment(self, image, prompt, shot, seconds, seed, out_path, audio_path=None):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt=prompt, image=image, width=self.vcfg.width, height=self.vcfg.height,
            negative_prompt=self.vcfg.video_negative or None,
            num_frames=self._aligned_frames_for(seconds), num_inference_steps=self.vcfg.wan_i2v_steps,
            guidance_scale=self.vcfg.wan_i2v_guidance_scale, generator=gen)
        self._export(result.frames[0], out_path, fps=self.vcfg.fps)
        return out_path


# =============================================================================
# LTX-2  ·  joint audio-video generation (Lightricks)
# =============================================================================

# The runner executes INSIDE the LTX-2 repo's own uv venv as a subprocess. It
# reads one JSON job file and writes one MP4. Written to disk by LTX2Video.load()
# so the exact code used is always inspectable next to the outputs.
#
# Why a Python runner instead of the repo's own CLI: the pipelines' Python API
# is documented and stable (constructor + __call__ signatures verified against
# a2vid_two_stage.py / distilled.py), while the CLI's --images/--distilled-lora
# argv encodings are argparse Actions that have already changed between
# revisions. The runner also introspects signatures (call_supported) so minor
# upstream signature drift degrades gracefully instead of crashing.
_LTX2_RUNNER_SOURCE = r'''
"""story_to_animation → LTX-2 bridge. Runs inside the LTX-2 repo venv.

Usage: python sta_ltx2_runner.py <job.json>

Job keys: checkpoint, spatial_upsampler, gemma_root, distilled_lora,
distilled_lora_strength, quantization, offload_mode, prompt, negative_prompt,
seed, height, width, num_frames, frame_rate, num_inference_steps,
a2v_guidance_scale, image, audio (optional), out.
"""
import inspect, json, sys
from pathlib import Path


def call_supported(fn, /, **kwargs):
    """Call fn with only the kwargs its signature accepts (drift tolerance)."""
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    import torch

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    # Image conditioning input — prefer the repo's own NamedTuple (the
    # pipelines access .path/.frame_idx/.strength attributes); fall back to a
    # plain tuple for revisions where it is one.
    def make_image(path):
        try:
            from ltx_pipelines.utils.args import ImageConditioningInput
            try:
                return ImageConditioningInput(path, 0, 1.0, 33)   # (path, frame, strength, crf)
            except TypeError:
                return ImageConditioningInput(path, 0, 1.0)
        except ImportError:
            return (path, 0, 1.0)

    ckpt = job["checkpoint"]
    ckpt_name = Path(ckpt).name.lower()
    # A pre-quantized fp8 checkpoint (filename contains "fp8") already stores
    # FP8 weights WITH their own per-tensor scales, and keeps certain layers
    # (to_gate_logits, norms, biases) in BF16. Applying a quantization POLICY
    # on top of it is double-quantization: the block-streaming builder's
    # derive_layout() then expects an `input_scale` for every param and raises
    #   KeyError: 'attn1.to_gate_logits.input_scale'
    # on the BF16 gate layers. The correct handling for a pre-quantized file is
    # quantization=None — the pipeline loads the file's own scales natively.
    ckpt_is_prequant_fp8 = "fp8" in ckpt_name

    quant = None
    quant_str = (job.get("quantization") or "").lower()
    if ckpt_is_prequant_fp8:
        # File is already FP8 — do NOT layer a cast policy on top.
        if quant_str in ("fp8-cast", "fp8-scaled-mm"):
            print(f"[LTX2-runner] checkpoint '{Path(ckpt).name}' is already FP8 — "
                  f"ignoring quantization='{quant_str}' and loading its embedded scales "
                  "(quantization=None). This avoids the block-streaming "
                  "'to_gate_logits.input_scale' KeyError.")
        quant = None
    elif quant_str and quant_str != "none":
        # bf16 checkpoint -> apply a cast policy so it fits in VRAM.
        # CORRECT import is ltx_core.quantization (NOT ltx_pipelines.utils.types).
        # Prefer the QuantizationPolicy classmethods: fp8_cast() builds the policy
        # from static op maps and does NOT read the checkpoint, so it avoids the
        # 46GB mmap that build_policy()/_read_scales() performs (that mmap fails
        # with 'Cannot allocate memory' under strict overcommit / low RAM).
        made = False
        try:
            from ltx_core.quantization import QuantizationPolicy
            if quant_str == "fp8-cast":
                quant = QuantizationPolicy.fp8_cast(); made = True
            elif quant_str == "fp8-scaled-mm":
                quant = QuantizationPolicy.fp8_scaled_mm(); made = True
            else:
                print(f"[LTX2-runner] unknown quantization '{quant_str}' — none applied")
                made = True
        except (ImportError, AttributeError):
            pass
        if not made:
            # Some older layouts expose the enum under ltx_pipelines.utils.types.
            try:
                from ltx_pipelines.utils.types import QuantizationPolicy
                if quant_str == "fp8-cast":
                    quant = QuantizationPolicy.fp8_cast(); made = True
                elif quant_str == "fp8-scaled-mm":
                    quant = QuantizationPolicy.fp8_scaled_mm(); made = True
            except (ImportError, AttributeError):
                pass
        if not made and quant_str == "fp8-cast":
            # Last resort: build_policy reads the checkpoint (mmap). Only reachable
            # if neither QuantizationPolicy location exists. May fail on large
            # files under strict memory overcommit — handled with a clear message.
            try:
                from ltx_core.quantization.fp8_cast import build_policy
                quant = build_policy(job["checkpoint"]); made = True
            except ImportError:
                print("[LTX2-runner] WARNING: no fp8 quantization API found — "
                      "running without fp8 (much higher VRAM).")
            except (RuntimeError, OSError) as e:
                # Typically 'unable to mmap ... Cannot allocate memory' on the 46GB
                # checkpoint. build_policy needs to map the whole file; the classmethod
                # path above doesn't. If we're here, that path was unavailable.
                raise RuntimeError(
                    "fp8-cast policy construction failed while reading the checkpoint "
                    f"({e}). This build lacks QuantizationPolicy.fp8_cast(); the fallback "
                    "build_policy() must mmap the full 46GB file and the OS refused it "
                    "(strict memory overcommit or an address-space limit). Fixes: raise "
                    "overcommit ('sudo sysctl vm.overcommit_memory=1'), add swap, or "
                    "upgrade ltx-core to a version exposing "
                    "ltx_core.quantization.QuantizationPolicy.") from e
        if not made and quant_str == "fp8-scaled-mm":
            print("[LTX2-runner] WARNING: fp8-scaled-mm API not found — running without it.")

    offload = None
    try:
        from ltx_pipelines.utils.types import OffloadMode
        want = (job.get("offload_mode") or "none")
        try:
            offload = OffloadMode(want)
        except ValueError:
            offload = OffloadMode[want.upper()]
    except Exception:
        offload = None

    distilled_ckpt = "distilled" in ckpt_name
    n_frames = int(job["num_frames"])
    fps = float(job["frame_rate"])
    tiling = TilingConfig.default()
    images = [make_image(job["image"])] if job.get("image") else []

    common = dict(
        prompt=job["prompt"],
        negative_prompt=job.get("negative_prompt", ""),
        seed=int(job["seed"]),
        height=int(job["height"]),
        width=int(job["width"]),
        num_frames=n_frames,
        frame_rate=fps,
        images=images,
        tiling_config=tiling,
        enhance_prompt=False,
    )

    with torch.no_grad():   # no_grad (not inference_mode): the spatial upsampler's
        # conv3d path has raised "Inference tensors cannot be saved for backward"
        # under inference_mode on some revisions; no_grad is strictly safe.
        if job.get("audio"):
            # ---- image + audio → video with native lip-sync -----------------
            from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
            lora_list = []
            if not distilled_ckpt and job.get("distilled_lora"):
                from ltx_core.loader import (LTXV_LORA_COMFY_RENAMING_MAP,
                                             LoraPathStrengthAndSDOps)
                lora_list = [LoraPathStrengthAndSDOps(
                    job["distilled_lora"],
                    float(job.get("distilled_lora_strength", 0.8)),
                    LTXV_LORA_COMFY_RENAMING_MAP)]
            pipe = call_supported(
                A2VidPipelineTwoStage,
                checkpoint_path=ckpt, distilled_lora=lora_list,
                spatial_upsampler_path=job["spatial_upsampler"],
                gemma_root=job["gemma_root"], loras=[],
                quantization=quant, offload_mode=offload)
            if distilled_ckpt:
                # Distilled checkpoint: guidance-free, fixed sigma schedule.
                from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
                s1_sigmas = DISTILLED_SIGMAS
                gp = MultiModalGuiderParams(
                    cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
                    modality_scale=1.0, skip_step=0, stg_blocks=[])
            else:
                s1_sigmas = None
                gp = MultiModalGuiderParams(
                    cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
                    modality_scale=float(job.get("a2v_guidance_scale", 3.0)),
                    skip_step=0, stg_blocks=[29])
            video, audio = call_supported(
                pipe.__call__, **common,
                num_inference_steps=int(job.get("num_inference_steps", 24)),
                video_guider_params=gp,
                audio_path=job["audio"],
                audio_start_time=0.0,
                audio_max_duration=n_frames / fps,
                stage_1_sigmas=s1_sigmas)
        else:
            # ---- image → video (no audio): fastest distilled i2v ------------
            # DistilledPipeline accepts either 'checkpoint_path' or
            # 'distilled_checkpoint_path' depending on the ltx-pipelines version;
            # call_supported() filters to whichever the installed version accepts.
            from ltx_pipelines.distilled import DistilledPipeline
            pipe = call_supported(
                DistilledPipeline,
                checkpoint_path=ckpt, distilled_checkpoint_path=ckpt,
                spatial_upsampler_path=job["spatial_upsampler"],
                gemma_root=job["gemma_root"], loras=[],
                quantization=quant, offload_mode=offload)
            out = call_supported(
                pipe.__call__, **common,
                video_guider_params=MultiModalGuiderParams(
                    cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
                    modality_scale=0.0, skip_step=0, stg_blocks=[]),
                num_inference_steps=int(job.get("num_inference_steps", 8)))
            video, audio = out if isinstance(out, tuple) else (out, None)

        encode_video(video=video, fps=fps, audio=audio,
                     output_path=job["out"],
                     video_chunks_number=get_video_chunks_number(n_frames, tiling))
    print("LTX2_RUNNER_OK " + job["out"])


if __name__ == "__main__":
    main()
'''


def download_ltx2_models(dest_dir: str = "~/repos/LTX-2/models",
                         which: str = "bf16-distilled",
                         include_upsampler: bool = True,
                         include_gemma: bool = False,
                         hf_token: str = "") -> Dict[str, str]:
    """Download the LTX-2 model files needed to run on an RTX 3090 (Ampere).

    The default fetches exactly what a 3090 needs that isn't already common:
    the BF16 distilled checkpoint (runs via fp8-cast weight-only compression,
    unlike the pre-quantized fp8 file which needs Ada+ FP8 tensor cores) and
    the x2 spatial upsampler. Files land in ``dest_dir`` and are picked up by
    find_ltx2_assets automatically.

    which:
      "bf16-distilled" -> ltx-2.3-22b-distilled-1.1.safetensors  (46GB, recommended)
      "bf16-dev"       -> ltx-2.3-22b-dev.safetensors            (46GB) + distilled LoRA
    include_gemma: also pull the Gemma text encoder repo (large; usually you
      already have it -- the earlier log showed gemma-3-12b was found).

    Returns a dict of the local paths downloaded. Requires huggingface_hub and,
    for Gemma, a HF token with the Gemma license accepted.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required "
                           "(pip install huggingface_hub).") from e

    dest = Path(os.path.expanduser(dest_dir))
    dest.mkdir(parents=True, exist_ok=True)
    repo = "Lightricks/LTX-2.3"
    tok = hf_token or os.environ.get("HF_TOKEN") or None
    out: Dict[str, str] = {}

    def _grab(filename: str, key: str):
        logger.info("[LTX2-DL] fetching %s from %s (large file -- first time is slow)...",
                    filename, repo)
        p = hf_hub_download(repo_id=repo, filename=filename,
                            local_dir=str(dest), token=tok)
        out[key] = str(Path(p).resolve())
        logger.info("[LTX2-DL] -> %s", out[key])

    if which == "bf16-dev":
        _grab("ltx-2.3-22b-dev.safetensors", "checkpoint")
        _grab("ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "distilled_lora")
    else:
        _grab("ltx-2.3-22b-distilled-1.1.safetensors", "checkpoint")

    if include_upsampler:
        _grab("ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "spatial_upsampler")

    if include_gemma:
        logger.info("[LTX2-DL] fetching Gemma text encoder "
                    "(google/gemma-3-12b-it-qat-q4_0-unquantized)...")
        gdir = snapshot_download(repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                                 local_dir=str(dest / "gemma-3-12b-it-qat-q4_0-unquantized"),
                                 token=tok)
        out["gemma_root"] = str(Path(gdir).resolve())
        logger.info("[LTX2-DL] -> %s", out["gemma_root"])

    logger.info("[LTX2-DL] done. find_ltx2_assets() will now auto-select the bf16 "
                "checkpoint (preferred over any fp8 file on this GPU).")
    return out


def find_ltx2_assets(repo_dir: str = "~/repos/LTX-2",
                     models_dir: str = "",
                     checkpoint: str = "",
                     spatial_upsampler: str = "",
                     distilled_lora: str = "",
                     gemma_root: str = "") -> Dict[str, Any]:
    """Locate the LTX-2 repo, python interpreter, and model files without
    importing anything heavy — safe to call from a notebook before a render.

    Python interpreter resolution order:
      1. explicit ltx2_python VideoConfig knob
      2. <repo>/.venv/bin/python  (if a uv venv was set up)
      3. sys.executable           (current environment — typical when ltx-core /
                                   ltx-pipelines are pip-installed into the same
                                   conda / virtualenv the notebook runs in)

    Model search order per asset: explicit path → models_dir →
    <repo>/models/ → repo root → HuggingFace cache. Newest file wins when
    multiple matches exist. The fp8 checkpoint is found correctly — its name
    matches ltx-2*distilled*.safetensors and it is not excluded.
    """
    import sys as _sys
    repo = Path(os.path.expanduser(repo_dir)).resolve()
    result: Dict[str, Any] = {
        "repo_dir": str(repo), "repo_exists": repo.is_dir(),
        "python": "", "python_exists": False,
        "checkpoint": "", "checkpoint_is_distilled": False,
        "spatial_upsampler": "", "distilled_lora": "", "gemma_root": "",
        "ready": False, "missing": [],
    }
    if not repo.is_dir():
        result["missing"].append(f"LTX-2 repo not found: {repo} "
                                 "(git clone https://github.com/Lightricks/LTX-2)")
        return result

    # Python: prefer a repo venv if one was set up, otherwise fall back to
    # the current process's interpreter (works when ltx-core/ltx-pipelines
    # are installed into the same conda/system env — no venv required).
    venv_py = repo / ".venv" / "bin" / "python"
    if venv_py.exists():
        py = str(venv_py)
    else:
        py = _sys.executable   # current env — most common without a venv
    result["python"], result["python_exists"] = py, Path(py).exists()

    roots = [Path(os.path.expanduser(models_dir)).resolve()] if models_dir else []
    # The LTX-2 README shows models stored in <repo>/models/ltx-2.3/ — include
    # that as a high-priority root so the layout from the quickstart "just works".
    repo_models = repo / "models"
    if repo_models.is_dir():
        roots.insert(0, repo_models)
    roots.append(repo)
    # Final fallback: HuggingFace cache (covers `hf_hub_download` downloads).
    hf_cache = Path(os.path.expanduser(
        os.environ.get("HF_HOME", os.environ.get("HUGGINGFACE_HUB_CACHE",
                       "~/.cache/huggingface")))).resolve()
    if hf_cache.is_dir():
        roots.append(hf_cache)

    def _find(explicit: str, *patterns: str, exclude: Tuple[str, ...] = ()) -> str:
        # Always return an ABSOLUTE path: the render subprocess runs with
        # cwd=repo_dir, so a relative model path would resolve against the repo.
        if explicit and Path(os.path.expanduser(explicit)).exists():
            return str(Path(os.path.expanduser(explicit)).resolve())
        hits: List[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for pat in patterns:
                hits.extend(root.rglob(pat))
        hits = [h for h in hits if h.is_file()
                and not any(x in h.name.lower() for x in exclude)]
        if not hits:
            return ""
        return str(max(hits, key=lambda p: p.stat().st_mtime).resolve())

    # A distilled *LoRA* and the *upscalers* also contain "ltx-2…distilled…"/
    # "ltx-2…" in their names — exclude them during the checkpoint search so
    # mtime ordering can never hand us the wrong file type.
    #
    # bf16-vs-fp8 preference: on a GPU without native FP8 matmul (Ampere / RTX
    # 30xx), a pre-quantized fp8 checkpoint cannot run. So when both a bf16 and
    # an fp8 distilled checkpoint are present, prefer the bf16 one there. On
    # Ada+ hardware (or when capability is unknown) keep the newest-wins rule.
    prefer_bf16 = (_gpu_has_fp8_tensor_cores() is False)
    if prefer_bf16 and not checkpoint:
        # Try bf16 distilled first (exclude fp8 too), then fall back to any distilled.
        result["checkpoint"] = _find(checkpoint, "ltx-2*distilled*.safetensors",
                                     exclude=("lora", "upscaler", "upsampler", "fp8"))
        if not result["checkpoint"]:
            result["checkpoint"] = _find(checkpoint, "ltx-2*distilled*.safetensors",
                                         exclude=("lora", "upscaler", "upsampler"))
    else:
        result["checkpoint"] = _find(checkpoint, "ltx-2*distilled*.safetensors",
                                     exclude=("lora", "upscaler", "upsampler"))
    if not result["checkpoint"]:
        dev_exclude = ("lora", "upscaler", "upsampler") + (("fp8",) if prefer_bf16 else ())
        result["checkpoint"] = _find(checkpoint, "ltx-2*dev*.safetensors",
                                     exclude=dev_exclude)
        if not result["checkpoint"] and prefer_bf16:
            result["checkpoint"] = _find(checkpoint, "ltx-2*dev*.safetensors",
                                         exclude=("lora", "upscaler", "upsampler"))
    result["checkpoint_is_distilled"] = "distilled" in Path(result["checkpoint"]).name.lower() \
        if result["checkpoint"] else False
    if not result["checkpoint"]:
        result["missing"].append("no LTX-2 checkpoint (*distilled*.safetensors or *dev*.safetensors) found")

    result["spatial_upsampler"] = _find(spatial_upsampler,
                                        "*spatial-upscaler-x2*.safetensors",
                                        "*spatial-upscaler*.safetensors")
    if not result["spatial_upsampler"]:
        result["missing"].append("spatial upscaler (*spatial-upscaler-x2*.safetensors) not found "
                                 "— required by both two-stage pipelines")

    result["distilled_lora"] = _find(distilled_lora, "*distilled-lora*.safetensors")
    if not result["distilled_lora"] and not result["checkpoint_is_distilled"]:
        result["missing"].append("distilled LoRA (*distilled-lora*.safetensors) not found "
                                 "— required when using the DEV checkpoint")

    # Gemma root: a directory whose name mentions gemma and contains config.json.
    if gemma_root and Path(os.path.expanduser(gemma_root)).is_dir():
        result["gemma_root"] = str(Path(os.path.expanduser(gemma_root)).resolve())
    else:
        for root in roots:
            if not root.is_dir():
                continue
            for cfg in root.rglob("config.json"):
                if "gemma" in str(cfg.parent).lower():
                    result["gemma_root"] = str(cfg.parent.resolve())
                    break
            if result["gemma_root"]:
                break
    if not result["gemma_root"]:
        result["missing"].append("Gemma text-encoder dir not found "
                                 "(google/gemma-3-12b-it-qat-q4_0-unquantized)")

    result["ready"] = not result["missing"]
    return result


class LTX2Video(BaseVideo):
    """LTX-2 (Lightricks) — one engine, two internal paths.

    Shots WITH audio use A2VidPipelineTwoStage: the still image anchors the
    subject's identity while the shot's own TTS audio drives motion AND
    lip-sync natively — the mouth is generated in sync with the waveform, so
    these shots are exempted from the external LatentSync pass entirely
    (see _needs_lipsync). Shots WITHOUT audio use DistilledPipeline, the
    fastest LTX-2 image-to-video (8 sigmas stage 1 + 4 stage 2).

    Both are TWO-STAGE pipelines: stage 1 denoises at the requested resolution
    and stage 2 does an internal latent 2x upsample + refinement to produce
    the final output at exactly the requested resolution. No upscaling beyond
    the project resolution occurs here — the user's 4K post-process is separate.

    Runs as a subprocess using the configured Python interpreter (defaults to
    sys.executable — the current conda/system env — when no venv is present).
    No venv is required; install ltx-core and ltx-pipelines into the running
    environment once:
        pip install -e ~/repos/LTX-2/packages/ltx-core
        pip install -e ~/repos/LTX-2/packages/ltx-pipelines
    Every render subprocess exits cleanly, returning its VRAM to the OS.

    Long takes: supports_audio_drive=True + native_long_video=False means the
    universal harness chains fixed-length segments, slicing the TTS audio per
    segment and continuing each segment from the previous one's last frame —
    so dialogue of any length stays lip-synced across the whole take.
    """
    name = "ltx2"
    supports_audio_drive = True     # harness slices the shot's audio per segment
    native_long_video = False       # chain segments through the standard harness
    preserves_generated_audio = True    # keep LTX-2's generated soundscape on
                                        # shots with NO dialogue (silent-shot ambient/FX);
                                        # dialogue shots still drop it via the keep_audio
                                        # gate below and mux the clean TTS track instead
    frame_align = 8                 # LTX-2 requires 8k+1 frames

    @property
    def segment_seconds(self) -> float:
        return float(getattr(self.vcfg, "ltx2_segment_seconds", 6.0))

    def load(self):
        v = self.vcfg
        repo = Path(os.path.expanduser(getattr(v, "ltx2_repo_dir", "~/repos/LTX-2"))).resolve()

        # ── 0. Optional: auto-fetch the bf16 checkpoint on Ampere ────────────
        # If this GPU has no FP8 tensor cores and the only checkpoint present is
        # a pre-quantized fp8 file (which can't run here), optionally download
        # the bf16 distilled checkpoint so the engine can actually run.
        def _discover():
            return find_ltx2_assets(
                repo_dir=str(repo),
                models_dir=getattr(v, "ltx2_models_dir", ""),
                checkpoint=getattr(v, "ltx2_checkpoint", ""),
                spatial_upsampler=getattr(v, "ltx2_spatial_upsampler", ""),
                distilled_lora=getattr(v, "ltx2_distilled_lora", ""),
                gemma_root=getattr(v, "ltx2_gemma_root", ""))

        assets = _discover()
        ckpt_name0 = Path(assets["checkpoint"]).name.lower() if assets["checkpoint"] else ""
        need_bf16 = (_gpu_has_fp8_tensor_cores() is False
                     and "fp8" in ckpt_name0
                     and not getattr(v, "ltx2_allow_fp8_on_ampere", False))
        if need_bf16 and getattr(v, "ltx2_auto_download_bf16", False):
            dest = getattr(v, "ltx2_models_dir", "") or str(repo / "models")
            logger.info("[LTX2] Ampere GPU + only an fp8 checkpoint found — "
                        "auto-downloading the bf16 distilled checkpoint to %s "
                        "(ltx2_auto_download_bf16=True). This is a 46GB one-time fetch.",
                        dest)
            download_ltx2_models(dest_dir=dest, which="bf16-distilled",
                                 include_upsampler=not bool(assets["spatial_upsampler"]),
                                 include_gemma=False)
            assets = _discover()   # bf16 is now preferred over the fp8 file

        # ── 1. Resolve the Python interpreter ────────────────────────────────
        # Priority: explicit ltx2_python → repo .venv → current sys.executable.

        if not assets["ready"]:
            lines = "\n  - ".join(assets["missing"])
            raise RuntimeError(
                f"LTX-2 is not ready:\n  - {lines}\n\n"
                "Download checklist (HuggingFace repo: Lightricks/LTX-2.3):\n"
                "  1. Checkpoint — for an RTX 3090 (Ampere) you need a BF16 file:\n"
                "       ltx-2.3-22b-distilled-1.1.safetensors   (46GB bf16 — USE THIS on 3090)\n"
                "       ltx-2.3-22b-dev.safetensors             (46GB bf16, slower) + distilled LoRA\n"
                "       ltx-2.3-22b-distilled-fp8.safetensors   (29.5GB — ONLY runs on RTX 40xx+)\n"
                "     Fastest way to get the right file:\n"
                "       sta.download_ltx2_models()   # fetches bf16 distilled + upsampler\n"
                "     Then use ltx2_quantization='fp8-cast' (weight-only, Ampere-safe).\n"
                "  2. Spatial upsampler:\n"
                "       ltx-2.3-spatial-upscaler-x2-1.1.safetensors  (1GB)\n"
                "  3. Distilled LoRA (only needed with the DEV checkpoint):\n"
                "       ltx-2.3-22b-distilled-lora-384-1.1.safetensors  (7.6GB)\n"
                "  4. Gemma text encoder — download the entire model repo:\n"
                "       huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized\n"
                "  Install ltx-core and ltx-pipelines into your Python environment:\n"
                f"       pip install -e {repo}/packages/ltx-core\n"
                f"       pip install -e {repo}/packages/ltx-pipelines\n"
                "  Place model files anywhere inside the repo or set VideoConfig.ltx2_models_dir\n"
                "  to a directory containing them (scanned recursively).\n"
                "  Call sta.find_ltx2_assets() from a notebook cell to check what was found.")

        self._assets = assets

        # Use explicit python, then venv python, then current interpreter.
        py = (getattr(v, "ltx2_python", "") or "").strip() or assets["python"]
        if not Path(py).exists():
            raise RuntimeError(
                f"LTX-2 python not found: {py!r}\n"
                "Set VideoConfig.ltx2_python to the interpreter that has "
                "ltx-core and ltx-pipelines installed.")
        self._python = py

        # We invoke the official CLI modules (`python -m ltx_pipelines.*`)
        # directly, so no runner script is written into the repo.

        quant = (getattr(v, "ltx2_quantization", "fp8-cast") or "fp8-cast").strip()
        offload = (getattr(v, "ltx2_offload_mode", "cpu") or "cpu").strip()
        ckpt_name = Path(assets["checkpoint"]).name
        is_fp8_file = "fp8" in ckpt_name.lower()

        # ── Ampere FP8 guard ────────────────────────────────────────────────
        # A pre-quantized fp8 checkpoint needs native FP8 (e4m3) matmul tensor
        # cores at RUNTIME. Those exist only on Ada Lovelace (RTX 40xx, cc 8.9)
        # and newer. On Ampere (RTX 30xx, cc 8.6) the runtime cannot dispatch
        # the FP8 path and raises 'fp8 matmul not supported' — no loading policy
        # changes that. Refuse early with an actionable message rather than
        # burning a ~90s load and then crashing deep in the transformer.
        if is_fp8_file:
            has_fp8 = _gpu_has_fp8_tensor_cores()
            if has_fp8 is False and not getattr(v, "ltx2_allow_fp8_on_ampere", False):
                try:
                    gpu = torch.cuda.get_device_name(0)
                    cc = torch.cuda.get_device_capability()
                except Exception:
                    gpu, cc = "your GPU", ("?", "?")
                raise RuntimeError(
                    f"LTX-2 checkpoint '{ckpt_name}' is a pre-quantized FP8 model, but "
                    f"{gpu} (compute capability {cc[0]}.{cc[1]}) has no native FP8 matmul "
                    "tensor cores. At inference it would raise "
                    "'RuntimeError: fp8 matmul not supported'.\n"
                    "FP8 matmul requires Ada Lovelace (RTX 40xx) / Hopper / Blackwell.\n\n"
                    "For an RTX 3090 (Ampere), use the BF16 distilled checkpoint with "
                    "fp8-cast instead — it compresses the weights to FP8 for storage but "
                    "keeps a matmul path Ampere supports:\n"
                    "    download  ltx-2.3-22b-distilled-1.1.safetensors   (46GB, bf16)\n"
                    "    set       VideoConfig.ltx2_checkpoint = '<path to that file>'\n"
                    "              VideoConfig.ltx2_quantization = 'fp8-cast'\n"
                    "  (or point ltx2_models_dir at a folder containing it and delete/move\n"
                    "   the fp8 file so it isn't auto-picked.)\n\n"
                    "To attempt the fp8 file anyway (expected to fail on Ampere), set "
                    "VideoConfig.ltx2_allow_fp8_on_ampere = True.")
            # fp8 file on capable HW, or explicitly forced: the runner loads its
            # embedded FP8 weights + scales directly (quantization=None), which
            # also sidesteps the block-streaming 'to_gate_logits.input_scale'
            # KeyError that a cast policy on top of an fp8 file would cause.
            if has_fp8 is False:
                logger.warning("[LTX2] forcing fp8 checkpoint on a non-Ada GPU "
                               "(ltx2_allow_fp8_on_ampere=True) — expect a runtime "
                               "'fp8 matmul not supported' error.")
            else:
                logger.info("[LTX2] pre-quantized fp8 checkpoint — loading embedded FP8 "
                            "weights + scales directly (native FP8 matmul available).")
        elif quant == "fp8-cast":
            logger.info("[LTX2] fp8-cast: bf16 weights compressed to FP8 for storage; "
                        "matmul path works on Ampere (3090). cpu offload peak VRAM ~18-22GB.")
        elif quant == "fp8-scaled-mm":
            logger.warning("[LTX2] fp8-scaled-mm requires Ada Lovelace (RTX 40xx) or newer. "
                           "On a 3090 this will likely crash. Switch to fp8-cast + a bf16 ckpt.")
        if not assets["checkpoint_is_distilled"]:
            logger.info("[LTX2] DEV checkpoint — distilled LoRA will be applied in stage 2.")
        eff_quant = "none (fp8 file, embedded scales)" if is_fp8_file else quant
        logger.info("[LTX2] ready:\n"
                    "  checkpoint:        %s\n"
                    "  spatial_upsampler: %s\n"
                    "  distilled_lora:    %s\n"
                    "  gemma:             %s\n"
                    "  quantization:      %s  offload: %s\n"
                    "  python:            %s",
                    ckpt_name,
                    Path(assets["spatial_upsampler"]).name,
                    Path(assets["distilled_lora"]).name if assets["distilled_lora"] else "(n/a — distilled ckpt)",
                    Path(assets["gemma_root"]).name,
                    eff_quant, offload, py)

    def unload(self):
        # Nothing lives in this process — each render subprocess already
        # returned its VRAM on exit.
        pass

    def _resolution(self) -> Tuple[int, int]:
        # Minimum alignment is 16 (LTX spatial VAE downsampling factor).
        # Using max(16, ...) rather than max(32, ...) allows preset resolutions
        # like 1280×720 (720 is a multiple of 16 but not 32 or 64) to pass
        # through without being incorrectly snapped down to 1280×704.
        mult = max(16, int(getattr(self.vcfg, "ltx2_resolution_multiple", 64)))
        w = _round_to_multiple(self.vcfg.width, mult, "nearest")
        h = _round_to_multiple(self.vcfg.height, mult, "nearest")
        # Apply the LTX-2 generation ceiling. GPU presets (applied via
        # apply_gpu_preset) set ltx2_max_long/short AND vcfg.width/height to
        # the exact target resolution, so this cap is a no-op in the normal case
        # and only activates if someone passes an oversized vcfg manually.
        max_long  = int(getattr(self.vcfg, "ltx2_max_long",  960))
        max_short = int(getattr(self.vcfg, "ltx2_max_short", 576))
        w, h = _cap_resolution_for_4090(w, h, max_long=max_long,
                                        max_short=max_short, multiple=mult)
        return w, h

    def _render_segment(self, image, prompt, shot, seconds, seed, out_path,
                        audio_path: Optional[str] = None) -> str:
        import subprocess
        v = self.vcfg
        w, h = self._resolution()
        fps = float(getattr(v, "ltx2_frame_rate", 0.0) or 0.0) or float(v.fps)
        frames = self._aligned_frames_for(seconds)   # 8k+1 via frame_align

        # Everything handed to the CLI must be absolute: the subprocess runs with
        # cwd=repo_dir, so a relative path (e.g. "animation_out/…") would resolve
        # against the repo dir, where nothing exists.
        def _abs(p) -> str:
            return str(Path(os.path.expanduser(str(p))).resolve()) if p else ""

        out_path = _abs(out_path)
        work = Path(out_path).parent
        stem = Path(out_path).stem
        img_path = str(work / f"{stem}_ltx2_start.png")
        image.save(img_path)
        img_path = _abs(img_path)

        ckpt = _abs(self._assets["checkpoint"])
        upsampler = _abs(self._assets["spatial_upsampler"])
        gemma = _abs(self._assets["gemma_root"])
        lora = _abs(self._assets["distilled_lora"])
        lora_strength = float(getattr(v, "ltx2_distilled_lora_strength", 0.8))
        quant = (getattr(v, "ltx2_quantization", "fp8-cast") or "").strip()
        offload = (getattr(v, "ltx2_offload_mode", "cpu") or "").strip()
        ckpt_is_distilled = self._assets.get("checkpoint_is_distilled", "distilled" in Path(ckpt).name.lower())
        ckpt_is_prequant_fp8 = "fp8" in Path(ckpt).name.lower()
        # Upsampler is optional — skip it for faster renders on 3090.
        # ltx2_use_upsampler=True (default): pass the upsampler path to the CLI
        # so the pipeline runs the 2× spatial upscale pass after generation.
        # ltx2_use_upsampler=False: omit the flag entirely; output stays at the
        # generation resolution (faster, uses less VRAM during the upscale pass).
        use_upsampler = bool(getattr(v, "ltx2_use_upsampler", True)) and bool(upsampler)
        # The prompt IS the shot's motion / action_sequence text.
        prompt = prompt or ""

        # ── Build the CLI command ────────────────────────────────────────────
        py = self._python
        base = [py, "-m", ""]   # module filled below

        def add_common(cmd: List[str], ckpt_flag: str):
            cmd += [ckpt_flag, ckpt,
                    "--gemma-root", gemma,
                    "--seed", str(int(seed)),
                    "--output-path", out_path,
                    "--prompt", prompt,
                    "--image", img_path, "0", "1.0", "33",
                    "--height", str(int(h)),
                    "--width", str(int(w)),
                    "--num-frames", str(int(frames)),
                    "--frame-rate", str(fps)]
            if use_upsampler:
                cmd += ["--spatial-upsampler-path", upsampler]
            if quant and quant.lower() != "none" and not ckpt_is_prequant_fp8:
                cmd += ["--quantization", quant]
            if offload and offload.lower() != "none":
                cmd += ["--offload", offload]

        if audio_path:
            # image + audio → video with native lip-sync (two-stage a2vid).
            module = getattr(v, "ltx2_a2v_module", "ltx_pipelines.a2vid_two_stage")
            cmd = [py, "-m", module]
            add_common(cmd, "--checkpoint-path")
            # a2vid_two_stage REQUIRES --distilled-lora PATH [STRENGTH] (it's a
            # required arg in the CLI, not optional — stage 2 always refines with
            # it, even when the base checkpoint is the distilled one).
            if not lora:
                raise RuntimeError(
                    f"LTX-2 audio shot {shot.index} needs the distilled LoRA, which "
                    "a2vid_two_stage requires but wasn't found: "
                    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors. Download it "
                    "(hf download Lightricks/LTX-2.3 "
                    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors --local-dir "
                    "<models_dir>) or set VideoConfig.ltx2_distilled_lora to its path. "
                    "Silent shots (distilled pipeline) don't need it, only audio ones.")
            cmd += ["--distilled-lora", lora, str(lora_strength)]
            audio_abs = _abs(audio_path)
            # a2vid sizes the target audio latent from the VIDEO segment
            # (num_frames/frame_rate) and TRUNCATES the encoded audio to it -- it
            # never pads. The harness pads video slightly past the spoken line, so a
            # raw slice underfills the segment and trips a latent-shape assert. Pad
            # the audio with trailing silence to just past the segment duration;
            # a2vid truncates it back to the exact target, keeping A/V aligned.
            _seg_secs = float(frames) / float(fps)
            _pad_secs = _seg_secs + 0.5
            _padded = str(Path(audio_abs).with_name(Path(audio_abs).stem + "_padded.wav"))
            audio_abs = _pad_audio_to_stereo(audio_abs, _pad_secs, _padded)
            cmd += ["--audio-path", audio_abs,
                    "--audio-start-time", "0",
                    "--audio-max-duration", str(_pad_secs)]
            # a2vid also accepts a negative prompt; pass the project's.
            if getattr(v, "video_negative", ""):
                cmd += ["--negative-prompt", v.video_negative]
            a2v_gs = float(getattr(v, "ltx2_a2v_guidance_scale", 0.0) or 0.0)
            if a2v_gs > 0:
                cmd += ["--a2v-guidance-scale", str(a2v_gs)]
        else:
            # image → video, no audio: fastest distilled pipeline.
            module = getattr(v, "ltx2_distilled_module", "ltx_pipelines.distilled")
            cmd = [py, "-m", module]
            add_common(cmd, "--distilled-checkpoint-path")

        env = dict(os.environ)
        # PyTorch renamed this env var; set only the new name and strip the
        # deprecated one (newer torch warns if PYTORCH_CUDA_ALLOC_CONF is present).
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        for pair in (getattr(v, "ltx2_extra_env", "") or "").split():
            if "=" in pair:
                k, _, val = pair.partition("=")
                env[k] = val

        mode = "audio→video (native lip-sync)" if audio_path else "image→video (distilled)"
        logger.info("[LTX2] shot %04d seg: %s  %dx%d %df @%gfps  [%s]",
                    shot.index, mode, w, h, frames, fps, module)
        logger.debug("[LTX2] cmd: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd, cwd=self._assets["repo_dir"], env=env,
            capture_output=True, text=True,
            timeout=int(getattr(v, "ltx2_timeout_sec", 5400)))
        if proc.returncode != 0 or not (Path(out_path).exists()
                                        and Path(out_path).stat().st_size > 0):
            out_all = (proc.stderr or "") + "\n" + (proc.stdout or "")
            tail = "\n".join(out_all.splitlines()[-30:])
            # Host-side mmap failure on the 46GB checkpoint — a system memory /
            # overcommit limit, not a pipeline bug. --offload disk is the setting
            # that avoids the big mmap by streaming weights from disk.
            if ("unable to mmap" in out_all or
                    ("Cannot allocate memory" in out_all and "mmap" in out_all.lower())):
                raise RuntimeError(
                    f"LTX-2 shot {shot.index}: the OS could not memory-map the "
                    "checkpoint (safetensors mmap failed with 'Cannot allocate "
                    "memory'). The 46GB file needs enough virtual-memory commit "
                    "to map. Fixes, easiest first:\n"
                    "  1. Stream weights from disk instead of mapping the whole file:\n"
                    "       VideoConfig.ltx2_offload_mode = 'disk'\n"
                    "  2. Allow memory overcommit:\n"
                    "       sudo sysctl -w vm.overcommit_memory=1\n"
                    "     (persist: echo 'vm.overcommit_memory=1' | sudo tee -a /etc/sysctl.conf)\n"
                    "  3. Add swap so there's backing commit for the mapping:\n"
                    "       sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile\n"
                    "       sudo mkswap /swapfile && sudo swapon /swapfile\n"
                    "  4. Raise the mmap-count limit if it's low:\n"
                    "       sudo sysctl -w vm.max_map_count=1048576\n\n"
                    f"Subprocess tail:\n{tail}")
            raise RuntimeError(
                f"LTX-2 CLI failed for shot {shot.index} (exit {proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n\nLast output:\n{tail}")
        try:
            os.remove(img_path)
        except OSError:
            pass
        return out_path


_VIDEO_REGISTRY = {
    "wan_s2v": Wan22S2VVideo,
    "wan_i2v": Wan22I2VVideo,
    "framepack": FramePackVideo,
    "ltx2": LTX2Video,
}


# =============================================================================
# NOTEBOOK-SAFE S2V GUARDS
# =============================================================================

def _s2v_allowed(vcfg: VideoConfig) -> bool:
    """True only when it is safe to even try Wan-S2V.

    In Jupyter, a hard GPU/driver crash from the 14B S2V load can kill the
    kernel or reboot the display driver before our normal try/except fallback
    can run. `notebook_safe_mode=True` therefore prevents S2V from being
    selected at all, including old plans that explicitly contain
    `engine: "wan_s2v"`.
    """
    return not bool(getattr(vcfg, "wan_s2v_disable", False) or
                    getattr(vcfg, "notebook_safe_mode", False))


def _s2v_fallback_engine(vcfg: VideoConfig) -> str:
    return (getattr(vcfg, "notebook_safe_dialogue_engine", "")
            or vcfg.wan_s2v_fallback_engine
            or "wan_i2v")


def _sanitize_engine_choice(engine_name: str, vcfg: VideoConfig) -> str:
    eng = (engine_name or "").lower().strip()

    # Historical / convenience aliases for the LTX-2 engine (the notebook used
    # "ltx23_audio" before this engine existed).
    if eng in ("ltx", "ltx-2", "ltx2.3", "ltx23", "ltx23_audio", "ltx2_audio"):
        eng = "ltx2"

    # This build intentionally supports only WAN, FramePack, and LTX-2 animation.
    removed_or_unsupported = {
    }
    if eng in removed_or_unsupported:
        return "wan_i2v"

    if eng == "ltx2" and getattr(vcfg, "ltx2_disable", False):
        return getattr(vcfg, "video_engine_fallback_engine", "wan_i2v") or "wan_i2v"
    if eng == "wan_s2v" and not _s2v_allowed(vcfg):
        return _s2v_fallback_engine(vcfg)
    if eng == "framepack" and getattr(vcfg, "notebook_safe_disable_framepack", False):
        return "wan_i2v"
    if eng not in _VIDEO_REGISTRY:
        return "wan_i2v"
    return eng


def find_latentsync_assets(repo_dir: str = "./model_cache/LatentSync",
                           checkpoint: str = "",
                           unet_config: str = "auto") -> Dict[str, Any]:
    """Locate a local LatentSync repo, checkpoint, Whisper tiny weights, and
    UNet config without importing LatentSync or touching CUDA.

    This is intentionally lightweight for Jupyter: it only inspects paths. It
    supports the common layout produced by LatentSync's setup_env.sh or a manual
    Hugging Face download:

        model_cache/LatentSync/
          scripts/inference.py
          configs/unet/stage2.yaml
          configs/unet/stage2_512.yaml       # present in newer repos
          checkpoints/latentsync_unet.pt
          checkpoints/whisper/tiny.pt

    Returns a dict so notebook cells can print it before a long render.
    """
    repo = _resolve_project_path(repo_dir)
    whisper_layout = ensure_latentsync_whisper_layout(str(repo))
    result: Dict[str, Any] = {
        "repo_dir": str(repo),
        "repo_exists": repo.is_dir(),
        "inference_script": str(repo / "scripts" / "inference.py"),
        "inference_script_exists": (repo / "scripts" / "inference.py").exists(),
        "checkpoint": "",
        "checkpoint_exists": False,
        "whisper_tiny": "",
        "whisper_tiny_exists": False,
        "whisper_tiny_expected": str(repo / "checkpoints" / "whisper" / "tiny.pt"),
        "whisper_tiny_expected_exists": (repo / "checkpoints" / "whisper" / "tiny.pt").exists(),
        "whisper_layout": whisper_layout,
        "unet_config": "",
        "unet_config_exists": False,
        "ok": False,
        "missing": [],
    }

    # Checkpoint: prefer explicit value, then the official default filename.
    candidates: List[Path] = []
    if checkpoint:
        candidates.append(Path(checkpoint).expanduser())
    candidates += [
        repo / "checkpoints" / "latentsync_unet.pt",
        repo / "latentsync_unet.pt",
    ]
    if repo.exists():
        candidates += list(repo.rglob("latentsync_unet.pt"))
        # Some downloads rename the file; keep this as a weaker fallback.
        candidates += list(repo.rglob("*unet*.pt"))
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            result["checkpoint"] = str(c)
            result["checkpoint_exists"] = True
            break

    # Whisper tiny is required by the official LatentSync inference path.
    whisper_candidates = [
        repo / "checkpoints" / "whisper" / "tiny.pt",
        repo / "whisper" / "tiny.pt",
    ]
    if repo.exists():
        whisper_candidates += list(repo.rglob("tiny.pt"))
    seen.clear()
    for c in whisper_candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            result["whisper_tiny"] = str(c)
            result["whisper_tiny_exists"] = True
            break
    result["whisper_tiny_expected_exists"] = (repo / "checkpoints" / "whisper" / "tiny.pt").exists()

    # Config: auto prefers 512 config when present because LatentSync 1.6 was
    # trained at 512x512. Fall back to stage2.yaml, then stage2_efficient.yaml.
    cfg_candidates: List[Path] = []
    if unet_config and unet_config != "auto":
        cfg_candidates.append(Path(unet_config).expanduser())
        cfg_candidates.append(repo / unet_config)
    else:
        cfg_candidates += [
            repo / "configs" / "unet" / "stage2_512.yaml",
            repo / "configs" / "unet" / "stage2.yaml",
            repo / "configs" / "unet" / "stage2_efficient.yaml",
        ]
    seen.clear()
    for c in cfg_candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            # The LatentSync CLI expects this path relative to cwd when cwd is
            # repo_dir; keep relative paths in that case for cleaner commands.
            try:
                result["unet_config"] = str(c.relative_to(repo))
            except Exception:
                result["unet_config"] = str(c)
            result["unet_config_exists"] = True
            break

    for key, label in [
        ("repo_exists", "LatentSync repo directory"),
        ("inference_script_exists", "scripts/inference.py"),
        ("checkpoint_exists", "latentsync_unet.pt checkpoint"),
        ("whisper_tiny_expected_exists", "checkpoints/whisper/tiny.pt"),
        ("unet_config_exists", "UNet config YAML"),
    ]:
        if not result[key]:
            result["missing"].append(label)
    result["ok"] = not result["missing"]
    return result


def validate_latentsync_install(repo_dir: str = "./model_cache/LatentSync",
                                checkpoint: str = "",
                                unet_config: str = "auto") -> Dict[str, Any]:
    """Notebook-friendly LatentSync validator.

    This does not run inference; it just reports whether the files needed by the
    pipeline are in the expected local locations. Use this before a long render.
    """
    assets = find_latentsync_assets(repo_dir, checkpoint, unet_config)
    if assets["ok"]:
        logger.info("[LatentSync] ready: repo=%s checkpoint=%s config=%s",
                    assets["repo_dir"], assets["checkpoint"], assets["unet_config"])
    else:
        logger.warning("[LatentSync] missing: %s", ", ".join(assets["missing"]))
    return assets

def sanitize_plan_for_no_s2v(plan_path: str, out_path: Optional[str] = None,
                             fallback_engine: str = "wan_i2v") -> str:
    """Rewrite an existing plan so it cannot select Wan-S2V.

    Useful in notebooks when an older plan already has `engine: wan_s2v` on
    dialogue shots. The runtime guards also protect you, but editing the plan
    makes the file itself honest and avoids confusion when reviewing plan.md.
    """
    p = Path(plan_path)
    plan = json.loads(p.read_text(encoding="utf-8"))
    cfg = plan.setdefault("config", {}).setdefault("video", {})
    cfg["wan_s2v_disable"] = True
    cfg["notebook_safe_mode"] = True
    cfg["notebook_safe_dialogue_engine"] = fallback_engine
    cfg.setdefault("lipsync_engine", "latentsync")
    changed = 0
    for scene in plan.get("scenes", []):
        for shot in scene.get("shots", []):
            if (shot.get("engine") or "").lower() == "wan_s2v":
                shot["engine"] = fallback_engine
                changed += 1
    out = Path(out_path) if out_path else p.with_name(p.stem + "_no_s2v.json")
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[PLAN] wrote S2V-free plan → %s (%d shot engine tag(s) changed).", out, changed)
    return str(out)


def apply_gpu_preset(
    video: "VideoConfig",
    gpu: str,
    gen_width: int = 0,
    gen_height: int = 0,
) -> "VideoConfig":
    """Apply GPU-specific LTX-2 and rendering settings to an existing VideoConfig.

    Call this after creating your VideoConfig with one of the notebook helpers,
    passing the GPU tier you are running on.  It sets every performance-sensitive
    field to the best-known values for that card so you only need to specify
    ``gpu`` once rather than remembering which quantization / resolution /
    offload combination works on each architecture.

    Supported GPU tiers
    -------------------
    "3090"  RTX 3090 / 3090 Ti  (24 GB, Ampere — no native FP8 tensor cores)
    "4090"  RTX 4090             (24 GB, Ada Lovelace — native FP8 scaled-mm)
    "5090"  RTX 5090             (32 GB, Blackwell   — native FP8 + more VRAM)

    Parameters
    ----------
    video      : VideoConfig to mutate in-place (and return).
    gpu        : One of "3090", "4090", "5090" (case-insensitive, strips
                 whitespace and common prefixes like "RTX ", "rtx").
    gen_width  : LTX-2 generation width.  0 → use the preset default for the
                 chosen GPU tier.  If provided it overrides the preset value,
                 but ltx2_max_long/short are still derived from it correctly.
    gen_height : LTX-2 generation height.  0 → use the preset default.

    Returns
    -------
    The same VideoConfig object (mutated), so you can chain:
        video = apply_gpu_preset(my_video_config, "4090")

    Generation resolutions and quantization per architecture
    ---------------------------------------------------------
    3090 (Ampere, 24 GB): 960×576 (16:9, both 64-multiples).
        fp8-cast is the correct quantization — fp8-scaled-mm requires Ada/Hopper/
        Blackwell and will crash on Ampere.  CPU offload is essential; the bf16
        checkpoint is 46 GB.  The upsampler is disabled so output stays at 960×576.

    4090 (Ada Lovelace, 24 GB): 1280×720 (16:9 / 720p).
        fp8-scaled-mm runs natively on Ada tensor cores.  720 is a multiple of 16
        (the LTX spatial alignment floor) so ltx2_resolution_multiple=16 is set to
        let it pass through _resolution() unchanged.  CPU offload keeps VRAM below
        the 24 GB ceiling.  The upsampler is disabled so output stays at 1280×720.

    5090 (Blackwell, 32 GB): 1280×720 (16:9 / 720p).
        fp8-scaled-mm runs fastest here.  32 GB VRAM comfortably fits the model
        without any offload.  Same 720p target as the 4090 for a consistent output
        format; disable the upsampler for fastest renders at that resolution.
    """
    # Normalise the gpu string: strip whitespace, common prefixes, and case.
    gpu_key = gpu.strip().upper().replace("RTX", "").replace("NVIDIA", "").replace(" ", "").replace("_", "")

    # ── Per-GPU preset tables ─────────────────────────────────────────────────
    # Each entry is a flat dict of VideoConfig field names → values.
    # Fields not listed here are left unchanged (whatever the helper set).
    _PRESETS: Dict[str, Dict] = {
        # ── RTX 3090 (Ampere, 24 GB) ─────────────────────────────────────────
        # Generation resolution: 960×576 (exact 64-multiples, 16:9).
        # fp8-cast: weights stored FP8, matmul runs in a mode Ampere supports.
        # CPU offload is essential: the bf16 checkpoint is 46 GB.
        # width/height are set here so every downstream path (image generation,
        # LTX2._resolution, compose, log messages) all agree on the same dims.
        "3090": dict(
            width                   = 960,
            height                  = 576,
            ltx2_max_long           = 960,
            ltx2_max_short          = 576,
            ltx2_use_upsampler      = True,  # 960×576 is the final output resolution
            ltx2_quantization       = "fp8-cast",
            ltx2_offload_mode       = "cpu",
            ltx2_segment_seconds    = 4.0,   # shorter segments → less VRAM per call
            image_max_long_side     = 960,
            high_vram_threshold_gb  = 20.0,  # 3090 has 24 GB; stay well under
            try_sage_attention      = True,
            use_fp8                 = True,
            wan_i2v_steps           = 28,    # Wan fallback: fewer steps on tighter VRAM
            wan_i2v_guidance_scale  = 5.0,
            fp_steps                = 28,
            fp_transformer_offload_preserve_gb = 6,
            fp_use_teacache         = False,
        ),
        # ── RTX 4090 (Ada Lovelace, 24 GB) ───────────────────────────────────
        # Generation resolution: 1280×720 (720p, 16:9).
        # 720 is a multiple of 16 but not 32 or 64; ltx2_resolution_multiple=16
        # lets LTX2._resolution() pass it through unchanged.
        # fp8-scaled-mm: native FP8 tensor cores on Ada — fastest quantization.
        # width/height are set here so every downstream path (image generation,
        # LTX2._resolution, compose, log messages) all agree on the same dims.
        "4090": dict(
            width                   = 1280,
            height                  = 720,
            ltx2_max_long           = 1280,
            ltx2_max_short          = 720,
            ltx2_resolution_multiple = 16,   # allows 720 (mult of 16) through unchanged
            ltx2_use_upsampler      = True,  # 1280×720 is the final output resolution
            ltx2_quantization       = "fp8-scaled-mm",
            ltx2_offload_mode       = "cpu",
            ltx2_segment_seconds    = 6.0,
            image_max_long_side     = 1280,
            high_vram_threshold_gb  = 40.0,
            try_sage_attention      = True,
            use_fp8                 = True,
            wan_i2v_steps           = 32,
            wan_i2v_guidance_scale  = 5.0,
            fp_steps                = 32,
            fp_transformer_offload_preserve_gb = 8,
            fp_use_teacache         = False,
        ),
        # ── RTX 5090 (Blackwell, 32 GB) ──────────────────────────────────────
        # Generation resolution: 1280×720 (720p, 16:9).
        # 32 GB VRAM comfortably fits the model without offload.
        # ltx2_resolution_multiple=16 allows 720 (mult of 16) through unchanged.
        # width/height are set here so every downstream path (image generation,
        # LTX2._resolution, compose, log messages) all agree on the same dims.
        "5090": dict(
            width                   = 1280,
            height                  = 720,
            ltx2_max_long           = 1280,
            ltx2_max_short          = 720,
            ltx2_resolution_multiple = 16,   # allows 720 (mult of 16) through unchanged
            ltx2_use_upsampler      = True,  # 1280×720 is the final output resolution
            ltx2_quantization       = "fp8-scaled-mm",
            ltx2_offload_mode       = "none",  # 32 GB fits the model without offload
            ltx2_segment_seconds    = 8.0,     # longer segments, more VRAM to spare
            image_max_long_side     = 1280,
            high_vram_threshold_gb  = 60.0,    # treat 5090 as high-VRAM throughout
            try_sage_attention      = True,
            use_fp8                 = True,
            wan_i2v_steps           = 40,
            wan_i2v_guidance_scale  = 5.0,
            fp_steps                = 40,
            fp_transformer_offload_preserve_gb = 12,
            fp_use_teacache         = True,    # 32 GB headroom makes TeaCache safe
        ),
    }

    preset = _PRESETS.get(gpu_key)
    if preset is None:
        known = ", ".join(sorted(_PRESETS))
        raise ValueError(
            f"Unknown GPU preset {gpu!r}. Known presets: {known}. "
            "Pass the VRAM size (e.g. '3090', '4090', '5090') or set "
            "VideoConfig fields manually."
        )

    # Apply every preset field to the VideoConfig.
    for field, value in preset.items():
        setattr(video, field, value)

    # ── Override resolution if caller supplied explicit dimensions ────────────
    # gen_width / gen_height are the LTX-2 *generation* dimensions (before any
    # upsampler pass). When provided they take precedence over the preset values.
    # Derive ltx2_max_long/short using the same orientation-safe max()/min()
    # logic so landscape and portrait both work correctly.
    if gen_width > 0 or gen_height > 0:
        w = gen_width  if gen_width  > 0 else video.width
        h = gen_height if gen_height > 0 else video.height
        # Snap to min 16-pixel multiple (LTX alignment floor)
        mult = max(16, int(getattr(video, "ltx2_resolution_multiple", 16)))
        w = max(mult, round(w / mult) * mult)
        h = max(mult, round(h / mult) * mult)
        video.width          = w
        video.height         = h
        video.ltx2_max_long  = max(w, h)
        video.ltx2_max_short = min(w, h)
        video.image_max_long_side = max(w, h)

    logger.info(
        "[GPU preset: %s] ltx2 %dx%d gen → %dx%d output | quant=%s offload=%s "
        "upsampler=%s seg=%.1fs",
        gpu_key,
        video.ltx2_max_long, video.ltx2_max_short,
        video.ltx2_max_long  * (2 if video.ltx2_use_upsampler else 1),
        video.ltx2_max_short * (2 if video.ltx2_use_upsampler else 1),
        video.ltx2_quantization, video.ltx2_offload_mode,
        video.ltx2_use_upsampler, video.ltx2_segment_seconds,
    )
    return video


def notebook_safe_4090_video_config(
    *,
    lipsync_repo_dir: str = "./model_cache/LatentSync",
    lipsync_checkpoint: str = "",
    lipsync_engine: str = "latentsync",
    lipsync_unet_config: str = "auto",
    lipsync_python_exe: str = "",
    lipsync_extra_args: str = "--inference_steps 25 --guidance_scale 1.5",
    width: int = 960,
    height: int = 544,
    fps: int = 24,
    fallback_engine: str = "wan_i2v",
    disable_framepack: bool = False,
    wan_i2v_steps: int = 32,
    require_lipsync: bool = True,
) -> VideoConfig:
    """Conservative local preset for RTX 4090 + Jupyter.

    It never selects Wan-S2V. Dialogue is rendered as ordinary image-to-video
    motion, then lip-synced with LatentSync/MuseTalk/Wav2Lip. For LatentSync,
    this helper auto-detects the local checkpoint/config under
    ./model_cache/LatentSync so notebook cells stay short and repeatable.
    """
    resolved_ckpt = lipsync_checkpoint
    resolved_cfg = lipsync_unet_config
    if (lipsync_engine or "").lower() == "latentsync":
        assets = find_latentsync_assets(lipsync_repo_dir, lipsync_checkpoint, lipsync_unet_config)
        if assets.get("checkpoint"):
            resolved_ckpt = assets["checkpoint"]
        if assets.get("unet_config"):
            resolved_cfg = assets["unet_config"]
        if not assets.get("ok"):
            logger.warning("[LatentSync] preset created, but missing: %s",
                           ", ".join(assets.get("missing", [])))

    return _make_video_config_filtered(
        routing="auto",
        single_engine=fallback_engine,
        wan_s2v_disable=True,
        notebook_safe_mode=True,
        notebook_safe_dialogue_engine=fallback_engine,
        notebook_safe_disable_framepack=disable_framepack,
        wan_s2v_fallback_engine=fallback_engine,
        width=width, height=height, fps=fps,
        cap_resolution_for_4090=True,
        wan_i2v_steps=wan_i2v_steps, wan_i2v_guidance_scale=5.0,
        fp_steps=32,
        fp_use_teacache=False,
        fp_internal_fps=30,
        fp_force_exact_duration=True,
        framepack_min_seconds=6.0,
        dialogue_long_takes_use_framepack=True,
        lipsync_engine=lipsync_engine,
        lipsync_repo_dir=lipsync_repo_dir,
        lipsync_checkpoint=resolved_ckpt,
        lipsync_unet_config=resolved_cfg if resolved_cfg != "auto" else "configs/unet/stage2.yaml",
        lipsync_python_exe=lipsync_python_exe,
        lipsync_extra_args=lipsync_extra_args,
        exact_audio_video_duration=True,
        write_sync_report=True,
        require_lipsync_for_dialogue_fallback=require_lipsync,
    )


def notebook_safe_4090_latentsync16_video_config(
    *,
    lipsync_repo_dir: str = "./model_cache/LatentSync",
    lipsync_checkpoint: str = "",
    lipsync_python_exe: str = "",
    lipsync_extra_args: str = "--inference_steps 25 --guidance_scale 1.5",
    width: int = 960,
    height: int = 544,
    fps: int = 24,
    fallback_engine: str = "wan_i2v",
    disable_framepack: bool = False,
    wan_i2v_steps: int = 32,
    require_lipsync: bool = True,
) -> VideoConfig:
    """LatentSync 1.6 preset for RTX 4090 + Jupyter.

    LatentSync 1.6 was trained at 512x512, so this preset explicitly prefers
    configs/unet/stage2_512.yaml. If that file is missing, it falls back to the
    normal auto-detector so the notebook can still run, but a warning is logged
    because a 1.6 install should normally include the 512 config. Wan-S2V stays
    disabled; dialogue is rendered with I2V and then synchronized by LatentSync.
    """
    repo = Path(lipsync_repo_dir).expanduser()
    cfg512 = repo / "configs" / "unet" / "stage2_512.yaml"
    unet_cfg = "configs/unet/stage2_512.yaml"
    if not cfg512.exists():
        logger.warning(
            "[LatentSync 1.6] expected %s but it was not found; falling back to auto config detection.",
            cfg512,
        )
        unet_cfg = "auto"

    return notebook_safe_4090_video_config(
        lipsync_repo_dir=lipsync_repo_dir,
        lipsync_checkpoint=lipsync_checkpoint,
        lipsync_engine="latentsync",
        lipsync_unet_config=unet_cfg,
        lipsync_python_exe=lipsync_python_exe,
        lipsync_extra_args=lipsync_extra_args,
        width=width, height=height, fps=fps,
        fallback_engine=fallback_engine,
        disable_framepack=disable_framepack,
        wan_i2v_steps=wan_i2v_steps,
        require_lipsync=require_lipsync,
    )

# =============================================================================
# ROUTING
# =============================================================================

def route_engine(shot: Shot, vcfg: VideoConfig) -> str:
    """Decide which engine renders this shot."""
    if vcfg.routing == "per_shot" and shot.engine:
        return _sanitize_engine_choice(shot.engine, vcfg)
    # LTX-2 handles BOTH shot types with one engine: shots with audio go
    # through its audio-to-video pipeline (image anchors identity, the TTS
    # track drives motion and native lip-sync); silent shots go through its
    # distilled image-to-video pipeline. When preferred, it takes every shot.
    if getattr(vcfg, "prefer_ltx2", False) and not getattr(vcfg, "ltx2_disable", False):
        return "ltx2"
    # Dialogue with audio prefers the safest audio-sync path for the current
    # environment. In notebook_safe_mode this is NEVER Wan-S2V; it is I2V first
    # and a separate lip-sync pass afterward. That is less magical than S2V,
    # but it survives a 24GB/Jupyter workflow and keeps renders resumable.
    if shot.is_dialogue and shot.audio_path:
        fallback = _s2v_fallback_engine(vcfg)
        secs = shot.duration if shot.duration > 0 else vcfg.default_seconds
        if (not _s2v_allowed(vcfg)
                and getattr(vcfg, "dialogue_long_takes_use_framepack", True)
                and not getattr(vcfg, "notebook_safe_disable_framepack", False)
                and secs >= float(getattr(vcfg, "framepack_min_seconds", 6.0))):
            # FramePack gives better long-form motion; LatentSync/MuseTalk/Wav2Lip
            # can sync the visible speaker afterward.
            return "framepack"
        if not _s2v_allowed(vcfg):
            return fallback
        # Wan-S2V is strongest when one audio track drives one visible speaking
        # face. Multi-speaker tracks and very long monologues are more reliable
        # as I2V motion followed by a dedicated lip-sync pass.
        if vcfg.prefer_single_speaker_s2v and _has_multispeaker_dialogue(shot):
            return fallback
        if (vcfg.max_s2v_dialogue_seconds and shot.duration
                and shot.duration > float(vcfg.max_s2v_dialogue_seconds)):
            return fallback
        return "wan_s2v"
    if vcfg.routing == "single":
        return _sanitize_engine_choice(vcfg.single_engine, vcfg)
    secs = shot.duration if shot.duration > 0 else vcfg.default_seconds
    if secs >= float(getattr(vcfg, "framepack_min_seconds", 6.0)) and not getattr(vcfg, "notebook_safe_disable_framepack", False):
        return "framepack"                     # long takes → FramePack's strength
    # Default is Wan2.2 for every shot type except dialogue (S2V) and long
    # takes (FramePack).
    return "wan_i2v"                            # default: Wan2.2 I2V


def _video_prompt(shot: Shot, theme: str, vcfg: Optional[VideoConfig] = None) -> str:
    """The SHORT prompt fed to the animation engine — motion, not description.

    Uses the per-shot motion_prompt authored in Phase 1 when present (the
    intended fix for video encoders that truncate long prompts). Falls back to
    a concise motion line built from the shot's cues — deliberately NOT the long
    image-prompt blob — and clamps to the engine-safe length.

    Injects _lipsync_motion_framing_cue (face in frame, hands down) rather than
    _lipsync_face_safety_instruction (which is for still image prompts and includes
    mouth-visibility language that would cue the animator to move the lips).
    """
    budget = int(vcfg.motion_prompt_max_chars) if vcfg is not None else 600
    framing = _lipsync_motion_framing_cue(shot)
    if shot.motion_prompt:
        prompt = _sanitize_motion_prompt_to_match_image(shot, shot.motion_prompt, vcfg)
        if framing and framing not in prompt:
            prompt = prompt.rstrip(". ") + ". " + framing
        return _clip_prompt(prompt, budget)
    if vcfg is not None and getattr(vcfg, "motion_prompts_respect_no_people_scenes", True) and not _shot_has_visible_people(shot):
        return _nonhuman_motion_prompt(shot, vcfg, cinematic=(vcfg is None or vcfg.cinematic_motion))
    desc = _visual_safe_description(shot).strip().rstrip(".")
    for sep in (". ", "; ", ", "):
        if sep in desc:
            desc = desc.split(sep)[0]
            break
    words = desc.split()
    if len(words) > 20:
        desc = " ".join(words[:20])
    bits = [desc]
    if vcfg is None or vcfg.cinematic_motion:
        c = _cinematic_cues(shot, vcfg)
        bits.append(f"Camera: {c['camera']}")
        bits.append(c["motion"])
    if framing:
        bits.append(framing)
    return _clip_prompt(
        _sanitize_motion_prompt_to_match_image(shot, ". ".join(b for b in bits if b) + ".", vcfg),
        budget,
    )


def _engine_candidates(shot: Shot, vcfg: VideoConfig) -> List[str]:
    """Which engine(s) will render this shot (mirrors animate_shots)."""
    if vcfg.routing == "compare_all":
        if not getattr(vcfg, "notebook_safe_disable_framepack", False):
            base.append("framepack")
        use_s2v = shot.is_dialogue and shot.audio_path and _s2v_allowed(vcfg)
        return base + (["wan_s2v"] if use_s2v else [])
    if vcfg.routing == "per_shot" and shot.engine:
        return [_sanitize_engine_choice(shot.engine, vcfg)]
    return [route_engine(shot, vcfg)]


def _anchor_count(cls, vcfg: VideoConfig, shot: Shot) -> int:
    """How many re-anchor stills `cls` needs for this shot (0 for native-long)."""
    try:
        eng = cls(vcfg)
    except Exception:
        return 0
    target = eng._target_seconds(shot)
    has_audio = bool(shot.audio_path and Path(shot.audio_path).exists())
    if not eng.supports_audio_drive and not has_audio:
        target = min(target, vcfg.max_seconds)
    _, groups = _plan_segments(target, eng.segment_seconds, vcfg.chain_segments,
                               eng.native_long_video, vcfg.max_segments_per_chain)
    return max(0, len(groups) - 1)


def _first_non_narrator_line(shot: "Shot") -> Optional["Line"]:
    for ln in shot.lines or []:
        if (ln.text or "").strip() and (ln.speaker or "").strip().upper() != "NARRATOR":
            return ln
    return None


def _speaker_visible_in_shot(speaker: str, shot: "Shot") -> bool:
    sp = (speaker or "").strip().lower()
    return any(str(c).strip().lower() == sp for c in (shot.characters_in_frame or []))


def enforce_visible_speaker_first_dialogue(shots: List["Shot"],
                                           enabled: bool = True) -> List["Shot"]:
    """Normalize shots so the first character voice matches the visible face.

    The external lip-sync stage assumes one visible face is driven by that same
    character's audio. The safest rule is: ignore narrator/caption audio, then
    the first non-narrator line in the shot should belong to the character that
    the shot image/video is about.

    When a mismatch is found, this does NOT rewrite dialogue text or reorder
    lines. It changes the shot direction to a cutaway/close-up of the first
    actual speaker, so the image prompt and I2V clip will show the correct face.
    Later, split_shots_on_audio_speaker_change() can safely split narrator and
    speaker changes into separate shots.
    """
    if not enabled:
        return shots

    changed = 0
    for sh in shots:
        first = _first_non_narrator_line(sh)
        if first is None:
            # Narration-only or silent shot: no visible speaking face required.
            continue

        speaker = (first.speaker or "").strip()
        if not speaker:
            continue

        # Already good: speaker is in frame.
        if _speaker_visible_in_shot(speaker, sh):
            continue

        # If the shot has no visible characters, make it a speaker close-up.
        # If it has a different visible character, retarget the shot to the
        # actual first speaker rather than syncing that other face to this voice.
        old_chars = list(sh.characters_in_frame or [])
        sh.characters_in_frame = [speaker]
        sh.composition = "medium_close_up"
        sh.description = (
            f"Cut to {speaker} during a visual dialogue beat in {sh.setting or 'the scene'}. "
            f"{speaker} is the only clear visible human face; mouth and eyes are unobstructed. "
            "No captions, subtitles, speech bubbles, or written words are visible."
        )
        sh.motion_prompt = (
            f"{speaker} holds a medium close-up with subtle facial movement and natural mouth movement; "
            "one centered clear face, front-facing or near-front-facing, mouth visible, eyes visible, no text."
        )
        # Force prompts/stills to regenerate from the corrected visual direction.
        sh.image_prompt = None
        changed += 1
        logger.debug(
            "  shot %04d retargeted visible speaker from %s to %s",
            sh.index, old_chars or "(none)", speaker,
        )

    if changed:
        logger.info("[DIALOGUE] retargeted %d shot(s) so the first non-narrator line matches the visible speaker.",
                    changed)
    return shots


def _speaker_key_for_line(line: "Line") -> str:
    speaker = (line.speaker or "").strip()
    return "NARRATOR" if speaker.upper() == "NARRATOR" else speaker


def _line_groups_by_speaker_change(lines: List["Line"]) -> List[List["Line"]]:
    """Group consecutive lines whenever the audio speaker changes."""
    groups: List[List[Line]] = []
    last_key: Optional[str] = None
    for ln in lines or []:
        if not (ln.text or "").strip():
            continue
        key = _speaker_key_for_line(ln)
        if not groups or key != last_key:
            groups.append([ln])
            last_key = key
        else:
            groups[-1].append(ln)
    return groups


def _shot_has_mixed_lipsync_audio(shot: "Shot") -> bool:
    """True when a shot's audio should be split before external lip-sync.

    A single audio file containing narrator + character speech, or multiple
    character voices, should not be fed to a lip-sync model against one visible
    face. Split whenever the consecutive speaker run changes.
    """
    groups = _line_groups_by_speaker_change(shot.lines)
    return len(groups) > 1


def _make_split_shot(original: "Shot", group: List["Line"], new_index: int,
                     part_no: int, total_parts: int) -> "Shot":
    speaker = _speaker_key_for_line(group[0]) if group else ""
    is_narrator = speaker.upper() == "NARRATOR"
    if is_narrator:
        characters = []
        composition = "wide_shot"
        description = (
            f"Narration cutaway in {original.setting or 'the scene'}. "
            "Visualize the environment, objects, weather, symbolic detail, or reaction mood. "
            "No speaking mouth is required and no written captions are present."
        )
        image_prompt = None
        motion_prompt = (
            "Cinematic cutaway with subtle environmental motion and slow camera movement; "
            "no visible speaking mouth, no captions, no subtitles, no text."
        )
        engine = original.engine
    else:
        # Cut to the actual voice source. If the speaker was not already in
        # frame, this intentionally creates a new shot/scene for that speaker
        # so LatentSync does not sync someone else's face to this voice.
        characters = [speaker]
        composition = "medium_close_up"
        description = (
            f"Cut to {speaker} during a dialogue beat in {original.setting or 'the scene'}. "
            "The speaker is the only clear visible face in frame; no written captions are present."
        )
        image_prompt = None
        motion_prompt = (
            f"{speaker} holds a medium close-up with subtle facial movement and natural mouth movement; "
            "one clear centered human face, front-facing or near-front-facing, mouth and eyes unobstructed, no text."
        )
        engine = original.engine

    sh = Shot(
        index=new_index,
        description=description,
        setting=original.setting,
        mood=original.mood,
        composition=composition,
        characters_in_frame=characters,
        lines=[Line(speaker=ln.speaker, text=ln.text, emotion=ln.emotion) for ln in group],
        image_prompt=image_prompt,
        motion_prompt=motion_prompt,
        duration_hint=None,
    )
    sh.engine = engine if engine and engine != "auto" else ""
    return sh


def split_shots_on_audio_speaker_change(shots: List["Shot"], enabled: bool = True) -> List["Shot"]:
    """Split shots so each produced audio file has one lip-sync voice target.

    This prevents the common failure mode where a shot shows one character but
    its audio contains narrator speech or another character's line. External
    lip-sync sees only one mixed audio waveform; it cannot know which words
    belong to the visible mouth.

    The split is conservative:
      • one line, or multiple consecutive lines by the same speaker -> unchanged
      • narrator + character, or speaker A -> speaker B -> split into new shots
      • narrator split shots use no characters_in_frame, so _needs_lipsync skips them
      • character split shots cut to that speaker as the only visible face
    """
    if not enabled:
        return shots

    out: List[Shot] = []
    next_index = 1
    split_count = 0

    for original in sorted(shots, key=lambda s: s.index):
        groups = _line_groups_by_speaker_change(original.lines)
        if len(groups) <= 1:
            # Reindex sequentially for clean filenames/manifests.
            original.index = next_index
            next_index += 1
            out.append(original)
            continue

        split_count += len(groups) - 1
        for part_no, group in enumerate(groups, 1):
            out.append(_make_split_shot(original, group, next_index, part_no, len(groups)))
            next_index += 1

    if split_count:
        logger.info("[PLAN] split %d mixed-audio boundary/boundaries into %d single-speaker/narration shot(s).",
                    split_count, len(out))
    return out


def plan_engines_and_anchors(shots: List[Shot], vcfg: VideoConfig) -> None:
    """Assign each shot its engine and compute how many starting images it needs.

    Runs BEFORE the image stage so the extra re-anchor stills can be generated
    while KLEIN2 is loaded (the video models never co-reside with it). For
    compare_all, anchor_count is the max across the candidate engines so every
    engine has enough fresh starts.
    """
    for sh in shots:
        cands = _engine_candidates(sh, vcfg)
        if vcfg.routing not in ("compare_all", "per_shot") or not sh.engine:
            sh.engine = cands[0]
        sh.anchor_count = max((_anchor_count(_VIDEO_REGISTRY[c], vcfg, sh)
                               for c in cands if c in _VIDEO_REGISTRY), default=0)
    n_anchored = sum(1 for s in shots if s.anchor_count)
    if n_anchored:
        logger.info("[PLAN] %d shot(s) need re-anchor stills (cap %d seg/chain).",
                    n_anchored, vcfg.max_segments_per_chain)


# =============================================================================
# STAGE 4 · ANIMATE  +  STAGE 5 · COMPOSE
# =============================================================================

def _plan_segments(target: float, seg_len: float, chain: bool = True,
                   native_long: bool = False, cap: int = 3):
    """Decide how to cover `target` seconds.

    Returns (seg_secs, groups) where `groups` is a list of segment-counts. Each
    GROUP is one continuous chain (last frame → next start). Engines that don't
    support long video are limited to `cap` segments per group; when more are
    needed the next group begins from a freshly generated starting image, which
    resets accumulated drift. Native-long engines (FramePack) use one group.
    """
    seg_len = max(0.5, seg_len)
    if not chain or target <= seg_len * 1.25:
        return target, [1]
    n_total = int(math.ceil(target / seg_len))
    seg_secs = target / n_total
    if native_long or cap <= 0:
        return seg_secs, [n_total]
    groups, rem = [], n_total
    while rem > 0:
        g = min(cap, rem)
        groups.append(g)
        rem -= g
    return seg_secs, groups


def _extract_last_frame(video_path: str, out_png: str) -> Optional[str]:
    """Grab the final frame of a clip as a PNG (the seed for the next segment)."""
    try:
        _run_ffmpeg(["-sseof", "-1", "-i", video_path, "-update", "1",
                     "-q:v", "1", out_png])
        return out_png if Path(out_png).exists() else None
    except Exception as e:
        logger.warning("  last-frame extract failed for %s: %s", video_path, e)
        return None


def _slice_audio(audio_path: str, start: float, dur: float, out_wav: str) -> Optional[str]:
    """Cut [start, start+dur) from an audio track (drives one S2V segment)."""
    try:
        _run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", audio_path, out_wav])
        return out_wav if Path(out_wav).exists() else None
    except Exception as e:
        logger.warning("  audio slice failed: %s", e)
        return None


def _pad_audio_to_stereo(src: str, seconds: float, out: str) -> str:
    """Pad `src` with trailing silence to `seconds` and force stereo.

    LTX-2 a2vid sizes the target audio latent from the VIDEO duration and
    truncates (never pads) the encoded audio to it, so an audio clip shorter
    than the padded video segment raises a latent-shape assertion. Padding to
    just past the segment lets a2vid truncate back to the exact target. Stereo
    is enforced because the audio VAE conv_in expects 2 channels.
    """
    try:
        _run_ffmpeg(["-i", src, "-af", "apad", "-t", f"{seconds:.3f}", "-ac", "2", out])
        return out if Path(out).exists() else src
    except Exception as e:
        logger.warning("  audio pad failed: %s", e)
        return src


def _concat_segments(seg_paths: List[str], out_path: str, fps: int,
                     target_seconds: Optional[float] = None,
                     keep_audio: bool = False):
    """Concatenate segment clips.

    audio/video can render useful SFX, so keep_audio=True preserves that stream
    through the chain before the final dialogue/SFX mix.
    """
    if len(seg_paths) == 1 and target_seconds is None:
        # A plain copy would carry the segment's generated audio through as-is.
        # That's correct only when we mean to keep it (keep_audio=True); when we
        # don't (e.g. a dialogue shot whose clean TTS is muxed later), strip it
        # here so the generated audio can never leak into the mux and double the
        # voice. This mirrors the multi-segment branch's -an handling.
        if keep_audio or not _clip_has_audio(seg_paths[0]):
            shutil.copy(seg_paths[0], out_path)
        else:
            _run_ffmpeg([
                "-i", seg_paths[0],
                "-c:v", "copy", "-an", out_path,
            ])
        return
    lst = Path(out_path).with_suffix(".segments.txt")
    lst.write_text("".join(f"file '{os.path.abspath(p)}'\n" for p in seg_paths))
    args = ["-f", "concat", "-safe", "0", "-i", str(lst)]
    if target_seconds:
        args += ["-t", f"{target_seconds:.3f}"]            # trim to the exact length
    if keep_audio and any(_clip_has_audio(p) for p in seg_paths):
        args += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", out_path,
        ]
    else:
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", "-r", str(fps), out_path]
    _run_ffmpeg(args)
    try:
        lst.unlink()
    except OSError:
        pass


def _mux_audio(video_no_audio: str, audio_path: Optional[str], out_path: str,
               fps: int, target_dur: Optional[float] = None,
               mix_existing_audio: bool = False,
               generated_audio_gain_db: float = -9.0,
               dialogue_gain_db: float = 0.0):
    """Attach a shot's TTS audio and force deterministic clip duration.

    The old failure mode for long animated films is subtle: one clip is a few
    frames shorter than its audio, the next is a few frames longer, ffmpeg uses
    ``-shortest`` at different stages, and small errors accumulate into visible
    dialogue drift. This muxer makes every rendered shot explicit:

    * if a real audio file exists, the output duration is exactly the audio
      duration (or ``target_dur`` if supplied), with the video padded by holding
      the final frame when needed;
    * if no audio exists, a silent 48 kHz stereo track is added for exactly the
      video duration (or ``target_dur``);
    * all outputs are H.264/yuv420p + 48 kHz stereo AAC, so concat is stable.
    """
    common = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
              "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    vdur = _probe_stream_duration(video_no_audio, "v:0")
    tol = 0.02

    # that generated audio and mix dialogue/TTS over it instead of replacing it.
    if mix_existing_audio and _clip_has_audio(video_no_audio):
        generated_audio_filter = f"volume={float(generated_audio_gain_db):.2f}dB,apad"
        dialogue_filter = f"volume={float(dialogue_gain_db):.2f}dB,apad"
        if audio_path and Path(audio_path).exists():
            adur = _wav_duration(audio_path)
            desired = float(target_dur or max(adur, vdur, 0.1))
            src = video_no_audio
            padded = None
            if vdur > 0 and desired > vdur + tol:
                extra = desired - vdur + (1.0 / max(1, fps))
                padded = str(Path(out_path).with_suffix(".padded.mp4"))
                _run_ffmpeg([
                    "-i", video_no_audio,
                    "-vf", f"tpad=stop_mode=clone:stop_duration={extra:.3f},fps={fps}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                    "-c:a", "copy", padded,
                ])
                src = padded
            fc = (
                f"[0:a:0]{generated_audio_filter},atrim=0:{desired:.3f},asetpts=N/SR/TB[a0];"
                f"[1:a:0]{dialogue_filter},atrim=0:{desired:.3f},asetpts=N/SR/TB[a1];"
                f"[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0,"
                f"atrim=0:{desired:.3f},asetpts=N/SR/TB[a]"
            )
            _run_ffmpeg([
                "-i", src, "-i", audio_path,
                "-filter_complex", fc,
                "-map", "0:v:0", "-map", "[a]",
                "-t", f"{desired:.3f}", *common, out_path,
            ])
            if padded:
                try:
                    os.remove(padded)
                except OSError:
                    pass
            return
        else:
            desired = float(target_dur or vdur or 0.1)
            _run_ffmpeg([
                "-i", video_no_audio,
                "-map", "0:v:0", "-map", "0:a:0",
                "-filter:a", f"{generated_audio_filter},atrim=0:{desired:.3f},asetpts=N/SR/TB",
                "-t", f"{desired:.3f}", *common, out_path,
            ])
            return

    if audio_path and Path(audio_path).exists():
        adur = _wav_duration(audio_path)
        desired = float(target_dur or adur or vdur or 0.0)
        if desired <= 0:
            desired = max(adur, vdur, 0.1)
        src = video_no_audio
        padded = None
        if vdur > 0 and desired > vdur + tol:
            extra = desired - vdur + (1.0 / max(1, fps))
            padded = str(Path(out_path).with_suffix(".padded.mp4"))
            _run_ffmpeg([
                "-i", video_no_audio,
                "-vf", f"tpad=stop_mode=clone:stop_duration={extra:.3f},fps={fps}",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), padded,
            ])
            src = padded
        # Use -t rather than -shortest so the final duration is explicit and
        # repeatable. If audio is a hair shorter than desired, apad fills the
        # tail with silence; if it is longer, atrim cuts it exactly.
        af = f"apad,atrim=0:{desired:.3f},asetpts=N/SR/TB"
        _run_ffmpeg([
            "-i", src, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-filter:a", af, "-t", f"{desired:.3f}", *common, out_path,
        ])
        if padded:
            try:
                os.remove(padded)
            except OSError:
                pass
    else:
        desired = float(target_dur or vdur or 0.1)
        _run_ffmpeg([
            "-i", video_no_audio, "-f", "lavfi", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map", "0:v:0", "-map", "1:a:0", "-t", f"{desired:.3f}", *common, out_path,
        ])


def _fallback_engine_for_failed_video_engine(eng_name: str, vcfg: VideoConfig) -> str:
    """Return a safe replacement engine for a failed video engine."""
    if eng_name == "wan_s2v":
        fallback = getattr(vcfg, "wan_s2v_fallback_engine", "") or ""
    else:
        fallback = getattr(vcfg, "video_engine_fallback_engine", "") or ""
    fallback = _sanitize_engine_choice(fallback, vcfg)
    if not fallback or fallback == eng_name or fallback not in _VIDEO_REGISTRY:
        return ""
    return fallback


def animate_shots(shots: List[Shot], vcfg: VideoConfig, pcfg: ProjectConfig,
                  resume: bool = True) -> None:
    """Render every shot, grouped by engine so each model loads/unloads once."""
    clips_dir = Path(pcfg.workdir()) / "clips"

    # assign engines first (plan_engines_and_anchors usually did this already)
    for sh in shots:
        if not sh.image_path:
            logger.warning("  shot %04d has no image — skipping animation.", sh.index)
            continue
        if vcfg.routing == "compare_all":
            cands = _engine_candidates(sh, vcfg)
            sh.engine = cands[0] if cands else route_engine(sh, vcfg)
        elif sh.engine:
            sh.engine = _sanitize_engine_choice(sh.engine, vcfg)
        else:
            sh.engine = route_engine(sh, vcfg)

    if vcfg.routing == "compare_all":
        engines_for = {sh.index: _engine_candidates(sh, vcfg) for sh in shots if sh.image_path}
    else:
        engines_for = {sh.index: [_sanitize_engine_choice(sh.engine, vcfg)] for sh in shots if sh.image_path}

    # bucket shots by engine
    buckets: Dict[str, List[Shot]] = {}
    for sh in shots:
        if not sh.image_path:
            continue
        for eng in engines_for[sh.index]:
            buckets.setdefault(eng, []).append(sh)

    # A list-based work queue (not buckets.items() directly) so a failed
    # engine's shots can be rerouted to a fallback engine and pushed back on
    # for processing, without mutating a dict mid-iteration.
    work: List[Tuple[str, List[Shot]]] = list(buckets.items())
    while work:
        eng_name, eng_shots = work.pop(0)
        safe_eng = _sanitize_engine_choice(eng_name, vcfg)
        if safe_eng != eng_name:
            logger.warning("[ANIMATE] %s is disabled in notebook-safe mode; rerouting %d shot(s) to %s.",
                           eng_name, len(eng_shots), safe_eng)
            for sh in eng_shots:
                sh.engine = safe_eng
            work.append((safe_eng, eng_shots))
            continue
        if eng_name not in _VIDEO_REGISTRY:
            logger.warning("  unknown engine %r — skipping %d shots.", eng_name, len(eng_shots))
            continue
        # Resume: shots whose final clip already exists AND is actually valid
        # (has audio if it's supposed to, and is at least as long as that
        # audio) don't need the engine. A stale clip from before an earlier
        # fix — e.g. one rendered without audio — is NOT trusted; it's
        # regenerated instead of silently shipping a known-bad cached file.
        pending = []
        for sh in eng_shots:
            final = str(clips_dir / f"shot{sh.index:04d}_{eng_name}.mp4")
            if resume and _cached_clip_is_valid(final, sh):
                if vcfg.routing == "compare_all":
                    sh.extra_videos[eng_name] = final
                    sh.video_path = sh.video_path or final
                else:
                    sh.video_path = final
                logger.info("  shot %04d  [%s] — cached", sh.index, eng_name)
            else:
                if resume and Path(final).exists():
                    logger.info("  shot %04d  [%s] — cached clip is stale "
                               "(missing/short audio), re-rendering.", sh.index, eng_name)
                pending.append(sh)
        if not pending:
            logger.info("[ANIMATE] %s — all shots cached, engine not loaded.", eng_name)
            continue
        logger.info("[ANIMATE] loading video engine: %s  (%d shots)", eng_name, len(pending))
        engine = _VIDEO_REGISTRY[eng_name](vcfg)
        loaded = False
        last_err: Optional[Exception] = None
        for attempt in range(2):                # one retry — driver hiccups right after a long,
            try:                                 # heavy prior workload are usually transient
                engine.load()
                loaded = True
                break
            except Exception as e:
                last_err = e
                _diagnose_device_not_ready(e)
                if attempt == 0:
                    logger.warning("  engine %s failed to load (%s) — clearing VRAM, "
                                   "giving the driver a moment, and retrying once.",
                                   eng_name, e)
                    _free_vram()
                    time.sleep(3)
        if not loaded:
            fallback = (
                _fallback_engine_for_failed_video_engine(eng_name, vcfg)
                if getattr(vcfg, "video_engine_fallback_on_load_failure", True)
                else ""
            )
            if fallback:
                logger.error(
                    "  engine %s failed to load after retry (%s) — falling back to "
                    "%s for its %d shot(s) instead of dropping them from the film.",
                    eng_name, last_err, fallback, len(pending))
                for sh in pending:
                    sh.engine = fallback
                work.append((fallback, pending))
            else:
                logger.error("  engine %s failed to load (%s) — skipping its %d shot(s).",
                             eng_name, last_err, len(pending))
            continue
        failed_for_fallback: List[Shot] = []
        engine_poisoned = False    # set once the CUDA context is unrecoverable
        for sh in pending:
            if engine_poisoned:
                # The GPU context is already corrupted by an earlier shot on this
                # engine; every remaining render would fail identically. Send them
                # straight to the fallback instead of wasting a full render + retry
                # on each one.
                failed_for_fallback.append(sh)
                continue
            prompt = _video_prompt(sh, pcfg.theme, vcfg)
            raw = str(clips_dir / f"shot{sh.index:04d}_{eng_name}_raw.mp4")
            final = str(clips_dir / f"shot{sh.index:04d}_{eng_name}.mp4")
            ok = False
            fatal = False
            last_exc: Optional[Exception] = None
            for attempt in range(2):            # one retry — these are 5–14B+ models on a 24GB
                try:                             # card; a transient OOM shouldn't permanently lose a shot
                    engine.animate(sh, prompt, raw)
                    ok = True
                    break
                except Exception as e:
                    last_exc = e
                    if _is_fatal_gpu_error(e):
                        # Unrecoverable in-process; don't retry, don't touch the
                        # rest — tear the engine down and reroute.
                        logger.error(
                            "  %s hit an UNRECOVERABLE GPU fault on shot %04d (%s). "
                            "The CUDA context is corrupted, so retrying it or the "
                            "remaining shots on this engine would fail identically; "
                            "rerouting them to the fallback engine instead.",
                            eng_name, sh.index, e)
                        fatal = True
                        break
                    if attempt == 0:
                        logger.warning("  %s failed on shot %04d (%s) — clearing VRAM "
                                       "and retrying once.", eng_name, sh.index, e)
                        _free_vram()
                    else:
                        logger.warning("  %s failed on shot %04d after retry: %s",
                                       eng_name, sh.index, e)
            if not ok:
                failed_for_fallback.append(sh)
                # Poison the engine if the fault is fatal, or if a shot failed both
                # attempts with a CUDA-type error (two consecutive GPU faults on the
                # same shot mean the context is almost certainly gone).
                if fatal or _is_cuda_error(last_exc):
                    engine_poisoned = True
                    _free_vram()
                    logger.error(
                        "[ANIMATE] %s: GPU context looks unrecoverable after shot "
                        "%04d — stopping this engine and rerouting its remaining "
                        "shot(s) to the fallback without retrying them.",
                        eng_name, sh.index)
                continue
            # Wan S2V already carries lip-sync timing; we still mux our clean
            # TTS track so the audible voice is exactly our render.
            target_dur = _wav_duration(sh.audio_path) if sh.audio_path else None
            # Keep an engine's generated audio (LTX-2's ambient/soundscape) ONLY
            # on non-dialogue shots. On dialogue shots we drop it and mux the
            # clean TTS track alone — mixing the model's own rendering of the same
            # line under the TTS doubles the voice (a slightly-offset second copy
            # at -9 dB) and reads as the audio being "off". This matches the
            # stated design intent, which the single-segment concat path was not
            # previously honoring.
            mix_gen_audio = bool(
                getattr(engine, "preserves_generated_audio", False)
            ) and not _dialogue_speakers(sh)
            _mux_audio(
                raw, sh.audio_path, final, vcfg.fps, target_dur=target_dur,
                mix_existing_audio=mix_gen_audio,
            )
            if sh.audio_path and vcfg.exact_audio_video_duration:
                delta = _sync_delta(final, sh.audio_path)
                if delta is not None and abs(delta) > vcfg.sync_tolerance_sec:
                    logger.warning("  shot %04d [%s] sync delta after mux: video-audio=%+.3fs",
                                   sh.index, eng_name, delta)
            try:
                os.remove(raw)
            except OSError:
                pass
            if vcfg.routing == "compare_all":
                sh.extra_videos[eng_name] = final
                if sh.video_path is None:
                    sh.video_path = final
            else:
                sh.video_path = final
            logger.info("  shot %04d  [%s] ✓", sh.index, eng_name)
        try:
            engine.unload()
        except Exception as e:
            # After a fatal GPU fault, unload itself can throw; don't let that
            # abort the reroute of the shots that still need rendering.
            logger.warning("[ANIMATE] %s unload raised (%s) — continuing to fallback.",
                           eng_name, e)
            _free_vram()
        logger.info("[ANIMATE] %s unloaded.", eng_name)
        if failed_for_fallback and getattr(vcfg, "video_engine_fallback_on_render_failure", True):
            fallback = _fallback_engine_for_failed_video_engine(eng_name, vcfg)
            if fallback:
                logger.warning(
                    "[ANIMATE] rerouting %d shot(s) that failed in %s to fallback engine %s.",
                    len(failed_for_fallback), eng_name, fallback)
                for sh in failed_for_fallback:
                    sh.engine = fallback
                work.append((fallback, failed_for_fallback))
            else:
                logger.warning(
                    "[ANIMATE] %d shot(s) failed in %s and no valid fallback is configured.",
                    len(failed_for_fallback), eng_name)


def _normalize_for_concat(src_path: str, out_path: str, width: int, height: int,
                          fps: int, prefer_interp: bool = True) -> str:
    """Normalize one clip to concat-safe streams without shortening audio/video.

    The output is H.264/yuv420p at project fps and 48 kHz stereo AAC. If one
    stream is shorter, it is padded to the longer stream instead of relying on
    ffmpeg ``-shortest``. That keeps lip-sync stable across the final concat.
    """
    scale_pad = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                 f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    vf_plain = f"{scale_pad},fps={fps}"
    vf_interp = (f"{scale_pad},minterpolate=fps={fps}:mi_mode=mci:"
                 f"mc_mode=aobmc:vsbmc=1")
    src_fps = _probe_fps(src_path)
    vf = vf_interp if (prefer_interp and 0 < src_fps < fps * 0.75) else vf_plain
    has_audio = _clip_has_audio(src_path)
    vdur = _probe_stream_duration(src_path, "v:0")
    adur = _probe_stream_duration(src_path, "a:0") if has_audio else 0.0
    target = max(vdur, adur, _probe_duration(src_path), 0.1)

    def _run(vfilter: str) -> None:
        if has_audio:
            vf2 = vfilter
            if vdur > 0 and target > vdur + 0.02:
                vf2 = f"{vfilter},tpad=stop_mode=clone:stop_duration={target - vdur + 1.0/max(1, fps):.3f}"
            af = f"apad,atrim=0:{target:.3f},asetpts=N/SR/TB"
            _run_ffmpeg([
                "-i", src_path,
                "-map", "0:v:0", "-map", "0:a:0",
                "-filter:v", vf2, "-filter:a", af, "-t", f"{target:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                out_path,
            ])
        else:
            _run_ffmpeg([
                "-i", src_path, "-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-map", "0:v:0", "-map", "1:a:0", "-filter:v", vfilter,
                "-t", f"{target:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                out_path,
            ])

    try:
        _run(vf)
    except Exception:
        if vf == vf_interp:
            _run(vf_plain)
        else:
            raise
    return out_path


def write_sync_report(shots: List[Shot], pcfg: ProjectConfig, vcfg: VideoConfig,
                      label: str = "final") -> Optional[str]:
    """Write a JSON/CSV sync QA report for every rendered shot."""
    if not vcfg.write_sync_report:
        return None
    wd = Path(pcfg.workdir()) / "clips"
    wd.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for sh in sorted(shots, key=lambda s: s.index):
        vpath = sh.video_path or ""
        apath = sh.audio_path or ""
        vdur = _probe_stream_duration(vpath, "v:0") if vpath and Path(vpath).exists() else 0.0
        adur = _wav_duration(apath) if apath and Path(apath).exists() else 0.0
        delta = (vdur - adur) if (vdur and adur) else None
        status = "ok"
        if sh.audio_path and not (apath and Path(apath).exists()):
            status = "missing_audio"
        elif sh.is_dialogue and sh.engine != "wan_s2v" and vcfg.lipsync_engine and not (vpath.endswith("_synced.mp4")):
            status = "needs_lipsync_or_failed"
        elif delta is not None and abs(delta) > vcfg.sync_tolerance_sec:
            status = "duration_delta"
        rows.append({
            "index": sh.index, "engine": sh.engine, "speaker": sh.speaking_character or "",
            "speakers": _dialogue_speakers(sh), "is_dialogue": sh.is_dialogue,
            "video": vpath, "audio": apath, "video_sec": round(vdur, 3),
            "audio_sec": round(adur, 3),
            "delta_video_minus_audio_sec": None if delta is None else round(delta, 3),
            "status": status,
        })
    base = wd / f"sync_report_{label}"
    json_path = str(base.with_suffix(".json"))
    csv_path = str(base.with_suffix(".csv"))
    Path(json_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) if rows else ["index", "status"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
    except Exception as e:
        logger.debug("  could not write sync CSV (%s)", e)
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        logger.warning("[SYNC] %d shot(s) need attention; report → %s", len(bad), json_path)
    else:
        logger.info("[SYNC] report → %s", json_path)
    return json_path


def _concat_copy(paths: List[str], out_path: str) -> str:
    """Loss-less concat of stream-compatible clips (the fast, hard-cut path)."""
    if len(paths) == 1:
        shutil.copy(paths[0], out_path)
        return out_path
    lst = Path(out_path).with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{os.path.abspath(p)}'\n" for p in paths))
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", out_path])
    try:
        lst.unlink()
    except OSError:
        pass
    return out_path


def _group_scenes(ordered: List[Shot], norm_paths: List[str]) -> List[List[str]]:
    """Group the normalized clips into scenes by consecutive shared setting.

    A scene boundary is where a shot's setting differs from the previous one's.
    When settings are absent (e.g. script mode never populated them) every clip
    lands in one group, so the transition stage naturally no-ops into a plain
    concat — same output as before.
    """
    def _key(sh: Shot) -> str:
        return (sh.setting or "").strip().lower()

    scenes: List[List[str]] = []
    prev_key: Optional[str] = None
    for sh, p in zip(ordered, norm_paths):
        k = _key(sh)
        if prev_key is None or k != prev_key or not scenes:
            scenes.append([p])
        else:
            scenes[-1].append(p)
        prev_key = k
    return scenes


_XFADE_TRANSITION = {
    "dissolve": "fade",
    "fade_black": "fadeblack",
    "fade_white": "fadewhite",
}


def _assemble_with_transitions(scene_clips: List[List[str]], out_path: str,
                               vcfg: VideoConfig) -> str:
    """Join per-scene clips with a transition at each scene boundary.

    Within a scene: hard cut (loss-less concat). Between scenes: an xfade
    (cross-dissolve / dip-to-black / dip-to-white) whose length is clamped so it
    can only sit inside the silent boundary padding, never over speech, and can
    never swallow a short scene.

    VIDEO gets the xfade dissolve/dip effect. AUDIO is always hard-cut — no
    acrossfade. The TTS stage already bakes scene_transition_pad_ms (~700ms) of
    silence at every scene boundary, so both edges of the join are silent and the
    hard cut is inaudible. Applying acrossfade would overlap audio samples and
    shorten the audio timeline by `d` seconds at each boundary, desynchronising
    it from the video xfade and producing the "audio delayed from animations"
    symptom. Hard-cutting audio keeps it frame-perfectly locked to the video.

    Any failure falls back to a plain hard-cut concat so the film always renders.
    """
    style = (getattr(vcfg, "scene_transition", "none") or "none").lower()
    transition = _XFADE_TRANSITION.get(style)

    # First collapse each scene to a single clip (hard cuts within the scene).
    scene_dir = Path(out_path).parent / "_scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_paths: List[str] = []
    for si, clips in enumerate(scene_clips):
        sp = str(scene_dir / f"scene{si:03d}.mp4")
        _concat_copy(clips, sp)
        scene_paths.append(sp)

    # Nothing to transition between (one scene, or transitions disabled).
    if transition is None or len(scene_paths) < 2:
        _concat_copy(scene_paths, out_path)
        return out_path

    fps = int(vcfg.fps)
    nominal = float(getattr(vcfg, "scene_transition_seconds", 0.5))
    ratio = float(getattr(vcfg, "scene_transition_max_ratio", 0.5))
    min_d = float(getattr(vcfg, "scene_transition_min_seconds", 0.12))
    durs = [_probe_stream_duration(p, "v:0") for p in scene_paths]

    # Per-boundary transition length, clamped to the shorter neighbour so xfade
    # offsets stay valid and a brief scene is never dissolved away entirely.
    Ds: List[float] = []
    for i in range(len(scene_paths) - 1):
        short = min(durs[i], durs[i + 1])
        d = min(nominal, max(0.0, ratio * short))
        Ds.append(d if d >= min_d else 0.0)

    # If every usable boundary collapsed to a hard cut, skip the filtergraph.
    if not any(d > 0 for d in Ds):
        _concat_copy(scene_paths, out_path)
        return out_path

    try:
        # Build a chained xfade (VIDEO ONLY) filtergraph.
        #
        # AUDIO IS NOT CROSS-FADED — it is hard-cut at every scene boundary.
        #
        # Why: acrossfade overlaps `d` seconds of audio from the tail of the
        # outgoing scene with the head of the incoming one, shortening the
        # audio timeline by `d` seconds at each boundary. xfade does the same
        # to video. The two filters operate independently inside the
        # filter_complex, and even tiny scheduling or rounding differences
        # between the audio and video overlap windows cause the audio to shift
        # earlier than its corresponding video frames — exactly the "audio
        # delayed from animations" symptom.
        #
        # The TTS stage already bakes scene_transition_pad_ms (~700ms) of
        # silence into the tail of every outgoing shot and the head of every
        # incoming shot at scene boundaries.  Both sides of the join are
        # (near-)silent, so an audio cross-fade provides zero audible benefit
        # while actively introducing the drift.  Hard-cutting the audio keeps
        # it frame-perfectly locked to the video for the entire film.
        #
        # Strategy:
        #   1. Build a video-only xfade chain (merged inputs, same as before).
        #   2. Extract and concatenate audio streams separately (hard cuts only).
        #   3. Mux the transitioned video with the untouched audio concat.
        #
        # Boundaries whose D rounded to 0 are folded by pre-joining those scenes
        # with a hard cut, so the chain only carries real transitions.
        merged_paths: List[str] = []
        merged_durs: List[float] = []
        cur = [scene_paths[0]]
        cur_dur = durs[0]
        chain_D: List[float] = []
        for i, d in enumerate(Ds):
            if d > 0:
                # close the current hard-cut run as one input, start a new one
                if len(cur) == 1:
                    merged_paths.append(cur[0])
                else:
                    m = str(scene_dir / f"merged{len(merged_paths):03d}.mp4")
                    _concat_copy(cur, m)
                    merged_paths.append(m)
                merged_durs.append(cur_dur)
                chain_D.append(d)
                cur = [scene_paths[i + 1]]
                cur_dur = durs[i + 1]
            else:
                cur.append(scene_paths[i + 1])
                cur_dur += durs[i + 1]
        if len(cur) == 1:
            merged_paths.append(cur[0])
        else:
            m = str(scene_dir / f"merged{len(merged_paths):03d}.mp4")
            _concat_copy(cur, m)
            merged_paths.append(m)
        merged_durs.append(cur_dur)

        if len(merged_paths) < 2:
            _concat_copy(merged_paths, out_path)
            return out_path

        inputs: List[str] = []
        for p in merged_paths:
            inputs += ["-i", p]

        # --- Step 1: video xfade chain (video stream only, no audio) ----------
        video_only = str(scene_dir / "video_xfade.mp4")
        fc_v_parts: List[str] = []
        v_prev = "0:v"
        acc = merged_durs[0]    # running video timeline length (shrinks by d at each xfade)
        for k, d in enumerate(chain_D):
            offset = max(0.0, acc - d)
            v_out = f"vx{k}"
            fc_v_parts.append(
                f"[{v_prev}][{k+1}:v]xfade=transition={transition}:"
                f"duration={d:.3f}:offset={offset:.3f}[{v_out}]")
            v_prev = v_out
            acc = acc + merged_durs[k + 1] - d   # xfade removes d seconds from the video total
        fc_v = ";".join(fc_v_parts)
        video_total = max(0.1, acc)

        _run_ffmpeg([
            *inputs,
            "-filter_complex", fc_v,
            "-map", f"[{v_prev}]",
            "-an",                                 # no audio in this pass
            "-t", f"{video_total:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            video_only,
        ])

        # --- Step 2: audio hard-cut concat, trimmed AT each boundary ---------
        # xfade removes d seconds from the VIDEO timeline at every transition
        # (the outgoing clip's tail and the incoming clip's head are blended
        # into a single d-second window instead of playing sequentially). The
        # audio must lose the same d seconds AT THE SAME POINT, or every shot
        # after the first transition ends up d seconds "behind" where its
        # video actually is — and that offset keeps accumulating at each
        # further boundary, which is exactly the drift-that-gets-worse-through
        # -the-film symptom this was producing.
        #
        # Cutting the aggregate sum(chain_D) off the very end (the old
        # approach) only fixes the TOTAL duration; it does nothing for the
        # mid-film misalignment, since every scene before the last transition
        # still plays its full, uncut audio against a video that already
        # skipped ahead.
        #
        # Fix: trim d off the TAIL of each OUTGOING clip's audio, right at its
        # own boundary. This is always safe — the TTS stage bakes
        # scene_transition_pad_ms/dialogue_scene_entry_pad_ms (>=700ms) of
        # silence into that exact tail specifically so a hard cut there is
        # inaudible, and the dissolve length d (<= scene_transition_seconds,
        # clamped further by scene_transition_max_ratio) is always well under
        # that padding.
        audio_only = str(scene_dir / "audio_concat.wav")
        n = len(merged_paths)
        fc_a_parts: List[str] = []
        for i in range(n):
            d = chain_D[i] if i < len(chain_D) else 0.0
            if d > 0:
                trimmed = max(0.0, merged_durs[i] - d)
                fc_a_parts.append(f"[{i}:a]atrim=0:{trimmed:.3f},asetpts=N/SR/TB[a{i}]")
            else:
                fc_a_parts.append(f"[{i}:a]asetpts=N/SR/TB[a{i}]")
        fc_a = ";".join(fc_a_parts) + ";" + "".join(f"[a{i}]" for i in range(n)) + \
            f"concat=n={n}:v=0:a=1[aout]"
        # audio_total now equals video_total exactly (same d's removed from the
        # same spots), so this is the true, boundary-accurate duration.
        audio_total = sum(merged_durs) - sum(chain_D)
        _run_ffmpeg([
            *inputs,
            "-filter_complex", fc_a,
            "-map", "[aout]",
            "-t", f"{audio_total:.3f}",
            "-ar", "48000", "-ac", "2",
            audio_only,
        ])

        # --- Step 3: mux video (with transitions) + audio (hard-cut, exact) --
        # audio_total == video_total already (both had sum(chain_D) removed at
        # the same boundaries), so this is now just an exact-length mux, not a
        # corrective trim.
        _mux_audio(video_only, audio_only, out_path, fps,
                   target_dur=video_total)

        logger.info("[COMPOSE] %s video transitions applied at %d scene boundary(ies); "
                    "audio hard-cut (no cross-fade) to preserve A/V sync.",
                    style, sum(1 for d in chain_D if d > 0))
        return out_path
    except Exception as e:
        logger.warning("[COMPOSE] scene-transition assembly failed (%s) — "
                       "falling back to hard cuts.", e)
        _concat_copy(scene_paths, out_path)
        return out_path


def compose_film(shots: List[Shot], pcfg: ProjectConfig, vcfg: VideoConfig) -> Optional[str]:
    """Concatenate all primary shot clips into one film.

    project may be 960×544), different fps, and different audio params (TTS mono
    vs. silent stereo vs. lip-sync output). The concat demuxer drops audio or
    fails outright on mismatched streams, so we first re-encode every clip to a
    uniform spec — project W×H (letterboxed, not stretched), project fps,
    48 kHz stereo AAC — then concatenate those. This is what guarantees the
    final film actually carries audio.
    """
    wd = Path(pcfg.workdir())
    norm_dir = wd / "clips" / "_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    ordered = [sh for sh in sorted(shots, key=lambda s: s.index)
               if sh.video_path and Path(sh.video_path).exists()]
    if not ordered:
        logger.warning("[COMPOSE] no clips to assemble.")
        return None

    W, H, fps = vcfg.width, vcfg.height, vcfg.fps
    norm_paths: List[str] = []
    for sh in ordered:
        np_out = str(norm_dir / f"shot{sh.index:04d}.mp4")
        _normalize_for_concat(sh.video_path, np_out, W, H, fps, prefer_interp=True)
        norm_paths.append(np_out)

    listfile = wd / "concat.txt"
    listfile.write_text("".join(f"file '{os.path.abspath(p)}'\n" for p in norm_paths))
    concat_out = str(wd / f"{_slug(pcfg.title)}_concat.mp4")
    style = (getattr(vcfg, "scene_transition", "none") or "none").lower()
    if style == "none":
        # Fast path: hard cuts everywhere (original behaviour).
        _run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-c", "copy", concat_out,
        ])
    else:
        # Group shots into scenes by setting and transition between scenes
        # (hard cuts still used within a scene). Silent boundary padding keeps
        # the dissolve/dip off of any dialogue.
        scenes = _group_scenes(ordered, norm_paths)
        _assemble_with_transitions(scenes, concat_out, vcfg)
        logger.info("[COMPOSE] assembled %d shot(s) across %d scene(s) with '%s' transitions.",
                    len(norm_paths), len(scenes), style)

    out = str(wd / f"{_slug(pcfg.title)}_film.mp4")
    if vcfg.end_fade and vcfg.fade_seconds > 0:
        # Gentle fade-out (video + audio) on the tail only — no fade-IN, so the
        # opening hook still lands hard in the first frame. Editorial, not a
        # grade, so it won't fight a manual color pass in post.
        dur = _probe_duration(concat_out)
        fs = float(vcfg.fade_seconds)
        st = max(0.0, dur - fs)
        _run_ffmpeg([
            "-i", concat_out,
            "-vf", f"fade=t=out:st={st:.3f}:d={fs:.3f}",
            "-af", f"afade=t=out:st={st:.3f}:d={fs:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(vcfg.fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", out,
        ])
        try:
            os.remove(concat_out)
        except OSError:
            pass
    else:
        shutil.move(concat_out, out)

    # Final QA pass — cheap (ffprobe only) but catches a broken film
    # immediately instead of leaving it to be discovered later: a missing
    # audio track, a near-zero duration, or a resolution that doesn't match
    # what was requested all indicate something upstream went wrong even
    # though every individual stage reported success.
    final_dur = _probe_duration(out)
    has_audio = _clip_has_audio(out)
    try:
        res_out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", out]).decode().strip()
    except Exception:
        res_out = "?"
    logger.info("[COMPOSE] film → %s  (%d shots, %.1fs, %s, audio=%s)",
               out, len(norm_paths), final_dur, res_out, "yes" if has_audio else "no")
    if final_dur < 0.5:
        logger.warning("[COMPOSE] final film is suspiciously short (%.2fs) — "
                       "check the shot list and per-stage logs above.", final_dur)
    any_shot_audio = any(sh.audio_path for sh in ordered)
    if any_shot_audio and not has_audio:
        logger.warning("[COMPOSE] shots had audio but the final film does not — "
                       "something in the mux/concat chain dropped it; check "
                       "the per-shot logs above.")
    return out


# =============================================================================
# LIP-SYNC  ·  post-process dialogue shots to match mouths to the TTS speech
# =============================================================================
# render plausible motion but their lips don't track the words. This stage
# re-syncs each dialogue shot's video against its own TTS audio using an
# external lip-sync model, producing a clip whose mouth matches the speech
# (audio included in the output). Wan-S2V output is already lip-synced, so it's
# skipped by default.
#
# These engines are separate research repos with their own CLIs/checkpoints, so
# each adapter shells out to the repo's inference script via a command TEMPLATE
# rather than importing it (the APIs vary too much to bind directly, and a
# subprocess keeps the lip-sync model out of our process so it never competes
# with anything else for VRAM). Configure VideoConfig.lipsync_repo_dir +
# lipsync_checkpoint; override lipsync_command_template if your install differs.
#
# Template placeholders: {py} {script} {video} {audio} {out} {ckpt} {repo} {extra}
# =============================================================================

class BaseLipSync:
    name = "base"
    default_script = ""                 # repo-relative inference script
    default_template = ""               # default command (formatted with the placeholders)

    def __init__(self, vcfg: VideoConfig):
        self.vcfg = vcfg

    def _python(self) -> str:
        import sys
        return self.vcfg.lipsync_python_exe or sys.executable

    def _repo(self) -> Path:
        return _resolve_project_path(self.vcfg.lipsync_repo_dir or ".")

    def _script(self) -> str:
        return self.vcfg.lipsync_inference_script or self.default_script

    def available(self) -> Tuple[bool, str]:
        """(ok, reason). The repo dir is required; the script must exist in it."""
        repo = self._repo()
        if not self.vcfg.lipsync_repo_dir:
            return False, "lipsync_repo_dir not set"
        if not repo.is_dir():
            return False, f"lipsync_repo_dir not found: {repo}"
        script = self._script()
        if script.endswith(".py") and not (repo / script).exists():
            return False, f"inference script not found: {repo / script}"
        if self.name == "latentsync":
            layout = ensure_latentsync_whisper_layout(str(repo))
            if not layout.get("ok"):
                return False, "LatentSync Whisper checkpoint layout invalid: " + "; ".join(layout.get("missing", []))
        return True, ""

    def _resolve_for_repo_or_project(self, value: str, repo: Path) -> Path:
        p = Path(str(value)).expanduser()
        if p.is_absolute():
            return p
        repo_candidate = (repo / p).resolve()
        if repo_candidate.exists():
            return repo_candidate
        return _resolve_project_path(str(p))

    def _command(self, video: str, audio: str, out: str) -> List[str]:
        repo = self._repo()
        video_abs = _resolve_project_path(video)
        audio_abs = _resolve_project_path(audio)
        out_abs = _resolve_project_path(out)
        out_abs.parent.mkdir(parents=True, exist_ok=True)

        ckpt_value = self.vcfg.lipsync_checkpoint
        if self.name == "latentsync" and not ckpt_value:
            ckpt_value = str(repo / "checkpoints" / "latentsync_unet.pt")
        ckpt_abs = self._resolve_for_repo_or_project(ckpt_value, repo) if ckpt_value else Path("")

        unet_value = self.vcfg.lipsync_unet_config
        if self.name == "latentsync" and (not unet_value or unet_value == "auto"):
            for candidate in ("configs/unet/stage2_512.yaml", "configs/unet/stage2.yaml", "configs/unet/stage2_efficient.yaml"):
                if (repo / candidate).exists():
                    unet_value = candidate
                    break
        unet_abs = self._resolve_for_repo_or_project(unet_value, repo) if unet_value else Path("")

        if self.name == "latentsync":
            for label, q in {
                "video": video_abs,
                "audio": audio_abs,
                "checkpoint": ckpt_abs,
                "unet_config": unet_abs,
            }.items():
                if not q.exists():
                    raise RuntimeError(f"LatentSync {label} path not found before launch: {q}")

            layout = ensure_latentsync_whisper_layout(str(repo))
            if not layout.get("ok"):
                raise RuntimeError("LatentSync Whisper layout not valid: " + "; ".join(layout.get("missing", [])))

            extra = shlex.split(self.vcfg.lipsync_extra_args or "")
            return [
                self._python(),
                "-m", "scripts.inference",
                "--unet_config_path", str(unet_abs),
                "--inference_ckpt_path", str(ckpt_abs),
                "--video_path", str(video_abs),
                "--audio_path", str(audio_abs),
                "--video_out_path", str(out_abs),
                *extra,
            ]

        tmpl = self.vcfg.lipsync_command_template or self.default_template
        cmd = tmpl.format(
            py=self._python(), script=self._script(),
            video=str(video_abs), audio=str(audio_abs), out=str(out_abs),
            ckpt=str(ckpt_abs) if ckpt_value else self.vcfg.lipsync_checkpoint,
            repo=str(repo),
            extra=self.vcfg.lipsync_extra_args,
            unet_config=str(unet_abs) if unet_value else self.vcfg.lipsync_unet_config,
        )
        return shlex.split(cmd)

    def sync(self, video_path: str, audio_path: str, out_path: str) -> str:
        repo = self._repo()
        out_abs = _resolve_project_path(out_path)
        cmd = self._command(video_path, audio_path, out_path)

        env = os.environ.copy()
        old_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo) + ((os.pathsep + old_pp) if old_pp else "")

        logger.info("  lipsync cwd: %s", repo)
        logger.info("  lipsync cmd: %s", " ".join(shlex.quote(str(x)) for x in cmd))

        # Stream output so notebook users still see progress, while also keeping
        # the tail so runtime errors like "Face not detected" can be classified
        # cleanly by lipsync_shots().
        tail: List[str] = []
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            tail.append(line.rstrip())
            if len(tail) > 80:
                tail = tail[-80:]
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"{self.name} subprocess failed with exit code {rc}.\\n"
                + "\\n".join(tail[-40:])
            )

        if not out_abs.exists():
            raise RuntimeError(f"{self.name}: no output produced at {out_abs}")
        return str(out_abs)

class LatentSyncLipSync(BaseLipSync):
    """ByteDance LatentSync (https://github.com/bytedance/LatentSync) —
    audio-conditioned latent diffusion lip-sync, ~6.5GB VRAM for inference
    alone so it's comfortable to run right after a video engine unloads.

    The real repo's scripts/inference.py REQUIRES --unet_config_path (see
    its own inference.sh) — earlier versions of this template omitted it,
    which would fail every call. Defaults to configs/unet/stage2.yaml (the
    repo's own inference.sh default, full quality); use stage2_efficient.yaml
    via VideoConfig.lipsync_unet_config if you want the lower-VRAM variant.
    """
    name = "latentsync"
    default_script = "scripts/inference.py"
    default_template = (
        "{py} -m scripts.inference --unet_config_path {unet_config} "
        "--inference_ckpt_path {ckpt} "
        "--video_path {video} --audio_path {audio} --video_out_path {out} {extra}"
    )


class MuseTalkLipSync(BaseLipSync):
    """Tencent MuseTalk — fast latent-space inpainting (256×256 face region).
    MuseTalk usually drives from a YAML listing video/audio pairs; if your build
    needs that, set lipsync_command_template to write+pass a config, or use the
    realtime inference entrypoint shown here."""
    name = "musetalk"
    default_script = "scripts/realtime_inference.py"
    default_template = (
        "{py} -m scripts.realtime_inference --video_path {video} "
        "--audio_path {audio} --result_dir {out} {extra}"
    )


class Wav2LipLipSync(BaseLipSync):
    """Wav2Lip — the classic, most reliable zero-shot sync (lower visual res).
    Good fallback when a face is hard for the diffusion models."""
    name = "wav2lip"
    default_script = "inference.py"
    default_template = (
        "{py} {script} --checkpoint_path {ckpt} --face {video} "
        "--audio {audio} --outfile {out} {extra}"
    )


_LIPSYNC_REGISTRY = {
    "latentsync": LatentSyncLipSync,
    "musetalk":   MuseTalkLipSync,
    "wav2lip":    Wav2LipLipSync,
}


def _needs_lipsync(sh: Shot, vcfg: VideoConfig) -> bool:
    """Whether this shot should be sent to an external lip-sync engine.

    Only character dialogue (non-NARRATOR, non-caption spoken lines) is a
    lip-sync target. Narration / caption-only shots intentionally do NOT get
    synced: the NARRATOR's voice-over is not tied to a visible face, and
    running LatentSync on such a shot would either fail ("no face detected")
    or produce incorrect mouth movement on an unrelated face in frame.

    Both lipsync_only_dialogue and lipsync_skip_narration must be True (the
    default) to enforce this; either alone is sufficient but both are kept for
    belt-and-braces clarity.
    """
    if not sh.audio_path:
        return False

    has_non_narrator_speech = any(
        (ln.text or "").strip() and (ln.speaker or "").upper() != "NARRATOR"
        for ln in (sh.lines or [])
    )

    # Primary narrator gates — checked independently so either flag alone
    # is sufficient to block narration-only shots.
    if getattr(vcfg, "lipsync_skip_narration", True) and not has_non_narrator_speech:
        return False

    if vcfg.lipsync_only_dialogue and not has_non_narrator_speech:
        return False

    dialogue_speakers = _dialogue_speakers(sh)
    if len(dialogue_speakers) != 1:
        # Mixed character voices should have been split before this point.
        # Do not let one visible face sync to multiple voices.
        return False

    if getattr(vcfg, "lipsync_require_visible_speaker", True):
        visible = {str(x).strip().lower() for x in (sh.characters_in_frame or [])}
        speaker = dialogue_speakers[0].strip().lower()
        if speaker not in visible:
            return False

    # Audio-driven engines already generate the mouth in sync with the
    # waveform: Wan-S2V natively, and LTX-2's a2vid path (any ltx2-rendered
    # shot that reaches here had audio, so it went through that path).
    # Running LatentSync on top would overwrite good lip-sync with a second,
    # competing pass.
    if vcfg.lipsync_skip_audio_driven and sh.engine in ("wan_s2v", "ltx2"):
        return False

    return bool(sh.video_path)


def _is_no_face_lipsync_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "face not detected" in msg
        or "no face detected" in msg
        or "0 faces" in msg
        or "no detectable face" in msg
    )


def lipsync_shots(shots: List[Shot], vcfg: VideoConfig, pcfg: ProjectConfig,
                  resume: bool = True) -> None:
    """Re-sync dialogue shots' mouths to their TTS audio (in place).

    With the default routing (dialogue+audio always → wan_s2v, which lip-syncs
    natively; everything else → wan_i2v), this stage should find NOTHING to
    do on a normal run — _needs_lipsync() already skips wan_s2v shots, and
    audio-bearing shots only end up on a different engine if you explicitly
    forced one via a per-shot tag. So in practice this is a rare correction
    path, not something the default workflow depends on. For re-syncing an
    arbitrary video to a DIFFERENT audio track (e.g. a redub into another
    language) outside the story pipeline entirely, use lipsync_video() below
    instead — this function only operates on Shot objects mid-render.

    Loads nothing into our process — each shot is a subprocess call to the
    configured lip-sync repo — so it's VRAM-safe to run right after animation.
    On any failure (no face detected, repo misconfigured, etc.) the shot keeps
    its already-muxed clip, so the film is never worse off than without this
    stage. Idempotent: a shot whose *_synced.mp4 exists is reused.
    """
    eng_name = (vcfg.lipsync_engine or "").lower()
    if not eng_name or eng_name == "none":
        return
    if eng_name not in _LIPSYNC_REGISTRY:
        logger.warning("[LIPSYNC] unknown engine %r — skipping (choose from %s).",
                       vcfg.lipsync_engine, sorted(_LIPSYNC_REGISTRY))
        return

    targets = [sh for sh in shots if _needs_lipsync(sh, vcfg)]
    if not targets:
        # Distinguish a true no-op from the case where animation failed and
        # therefore no video_path exists for otherwise-syncable dialogue.
        missing_video = []
        for sh in shots:
            if not sh.audio_path:
                continue
            if not _dialogue_speakers(sh):
                continue
            if sh.engine == "wan_s2v" and vcfg.lipsync_skip_audio_driven:
                continue
            if not sh.video_path:
                missing_video.append(sh.id)
        if missing_video:
            logger.warning("[LIPSYNC] no shots could be synced because %d dialogue/audio shot(s) have no rendered video_path yet: %s",
                           len(missing_video), ", ".join(missing_video[:12]) + ("..." if len(missing_video) > 12 else ""))
        else:
            logger.info("[LIPSYNC] no shots need external syncing.")
        return

    engine = _LIPSYNC_REGISTRY[eng_name](vcfg)
    ok, reason = engine.available()
    if not ok:
        msg = (
            f"[LIPSYNC] {eng_name} unavailable ({reason}) — {len(targets)} dialogue "
            "shot(s) rendered by non-audio-driven engines will keep their muxed "
            "audio but mouths may not match. Set lipsync_repo_dir/lipsync_checkpoint, "
            "or set VideoConfig.wan_s2v_disable=False to use audio-driven S2V where "
            "possible, or VideoConfig.lipsync_engine=None to silence this.")
        if vcfg.require_lipsync_for_dialogue_fallback:
            raise RuntimeError(msg)
        logger.warning(msg)
        return

    clips_dir = Path(pcfg.workdir()) / "clips"
    logger.info("[LIPSYNC] %s — %d dialogue shot(s)…", eng_name, len(targets))
    for sh in targets:
        out = str(clips_dir / f"shot{sh.index:04d}_synced.mp4")
        if resume and _cached_clip_is_valid(out, sh):
            sh.video_path = out
            logger.info("  shot %04d  [lipsync] — cached", sh.index)
            continue
        try:
            engine.sync(sh.video_path, sh.audio_path, out)
            # Keep the lip-synced mouth motion, but always remux the original
            # clean TTS audio and force exact duration again. Lip-sync repos
            # often re-encode audio/video with slightly different stream lengths.
            fixed = out
            if vcfg.exact_audio_video_duration:
                fixed = str(clips_dir / f"shot{sh.index:04d}_synced_fixed.mp4")
                _mux_audio(out, sh.audio_path, fixed, vcfg.fps,
                           target_dur=_wav_duration(sh.audio_path))
            sh.video_path = fixed               # swap in the synced + duration-fixed clip
            logger.info("  shot %04d  [lipsync:%s] ✓", sh.index, eng_name)
        except Exception as e:
            if getattr(vcfg, "lipsync_skip_if_no_face", True) and _is_no_face_lipsync_error(e):
                logger.info(
                    "  shot %04d  [lipsync:%s] skipped — no detectable face; keeping muxed/narrated clip.",
                    sh.index, eng_name,
                )
            else:
                logger.warning("  shot %04d lip-sync failed (%s) — keeping muxed clip.",
                               sh.index, e)
    logger.info("[LIPSYNC] done.")


def lipsync_video(video_path: str, audio_path: str, out_path: Optional[str] = None,
                  engine: str = "latentsync", vcfg: Optional[VideoConfig] = None) -> str:
    """Stand-alone utility: re-sync ANY video's mouth movement to ANY audio
    track — independent of the story pipeline entirely.

    The main use case this is FOR: you already have a rendered video (from
    this pipeline, or anywhere else) and want to swap in a different audio
    track — most commonly a redub into another language — and have the lips
    actually match the new words, since the original video's mouth movement
    was made for different audio (or none at all, e.g. a wan_i2v clip).

        lipsync_video("scene.mp4", "scene_spanish.wav", "scene_es.mp4")

    `vcfg` supplies the engine's repo_dir/checkpoint/command_template — pass
    your existing VideoConfig (only the lipsync_* fields are read), or a bare
    `VideoConfig(lipsync_repo_dir=..., lipsync_checkpoint=...)` if you're
    calling this outside a full pipeline run. `engine` selects latentsync
    (default, best quality) | musetalk (fast) | wav2lip (most reliable sync).
    Raises RuntimeError with a clear reason if the engine isn't runnable
    (no repo configured, etc.) — unlike lipsync_shots(), this doesn't swallow
    the failure, since here there's no "keep the original" fallback shot to
    silently fall back to: producing a synced video IS the point of the call.
    """
    eng_name = (engine or "").lower()
    if eng_name not in _LIPSYNC_REGISTRY:
        raise ValueError(f"Unknown lip-sync engine {engine!r}. "
                         f"Choose from {sorted(_LIPSYNC_REGISTRY)}.")
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)
    if not Path(audio_path).exists():
        raise FileNotFoundError(audio_path)
    vcfg = vcfg or VideoConfig(lipsync_engine=eng_name)
    eng = _LIPSYNC_REGISTRY[eng_name](vcfg)
    ok, reason = eng.available()
    if not ok:
        raise RuntimeError(
            f"{eng_name} isn't runnable ({reason}). Set lipsync_repo_dir / "
            f"lipsync_checkpoint on the VideoConfig you pass in (or "
            f"lipsync_command_template for a fully custom invocation).")
    out_path = out_path or str(Path(video_path).with_name(
        f"{Path(video_path).stem}_synced.mp4"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("[LIPSYNC] %s: %s + %s → %s", eng_name, video_path, audio_path, out_path)
    eng.sync(video_path, audio_path, out_path)
    logger.info("[LIPSYNC] done → %s", out_path)
    return out_path


# =============================================================================
# THE PLAN  ·  editable hand-off between Phase 1 (plan) and Phase 2 (produce)
# =============================================================================
# Phase 1 writes plan.json (+ a readable plan.md). You open it, tweak anything —
# dialogue text, a character's voice (point ref_wav at your own clip to clone
# it), an image_prompt, a per-shot engine, a duration — then Phase 2 reads it
# back and renders. Re-running Phase 2 is idempotent: finished images/audio/
# clips are reused unless you pass resume=False (or delete them).
# =============================================================================

PLAN_VERSION = 2


class _PlanCharacter:
    """Lightweight stand-in for novel_generator.Character when producing from a
    plan (carries just what image-prompt assembly + voice casting need)."""
    def __init__(self, d: Dict[str, Any]):
        self.name = d.get("name", "")
        self.gender = d.get("gender", "")
        self.physical_build = d.get("physical_build", "")
        self.appearance = d.get("appearance", "")
        for a in ("dialect", "cadence", "vocabulary_level", "speech_pattern",
                  "voice_profile", "humor_style"):
            setattr(self, a, d.get(a, ""))
        self.verbal_tics = d.get("verbal_tics", []) or []
        self.catchphrases = d.get("catchphrases", []) or []


def _char_dict(ch, voice: "VoiceProfile") -> Dict[str, Any]:
    return {
        "name": getattr(ch, "name", ""),
        "gender": getattr(ch, "gender", ""),
        "physical_build": getattr(ch, "physical_build", ""),
        "appearance": getattr(ch, "appearance", ""),
        "role": getattr(ch, "role", ""),
        "voice": {
            "source": voice.source, "engine": "",            # "" = inherit project tts engine
            "ref_wav": voice.ref_wav, "ref_text": voice.params.get("ref_text", voice.ref_text),
            "seed": voice.seed, "params": voice.params,
            "voice_folder": voice.params.get("voice_folder", ""),
            "voice_folder_id": voice.params.get("voice_folder_id", ""),
            "voice_description": voice.params.get("voice_description", ""),
            "voice_file": voice.params.get("voice_file", ""),
            # Baseline reference: edit baseline_text and flip regenerate_baseline
            # to True to redo this character's generated voice clip; or just
            # set ref_wav above to your own file to bypass baselines entirely.
            "baseline_text": voice.baseline_text,
            "baseline_path": voice.baseline_path,
            "regenerate_baseline": voice.regenerate_baseline,
        },
    }


def build_plan(anim: "StoryAnimator") -> Dict[str, Any]:
    """Serialize the current StoryAnimator state into an editable plan dict."""
    p, t, v = anim.p, anim.t, anim.v
    char_by = _char_lookup(anim.characters)
    # group shots into scenes by setting for a readable, editable structure
    scenes, cur, last = [], None, object()
    for sh in anim.shots:
        key = (sh.setting or "").strip().lower()
        if cur is None or key != last:
            cur = {"setting": sh.setting, "mood": sh.mood, "shots": []}
            scenes.append(cur)
            last = key
        cur["shots"].append({
            "index": sh.index,
            "description": sh.description,
            "characters_in_frame": sh.characters_in_frame,
            "composition": sh.composition,
            "setting": sh.setting, "mood": sh.mood,
            "image_prompt": sh.image_prompt or "",
            "motion_prompt": sh.motion_prompt or "",
            "action_sequence": sh.action_sequence if sh.action_sequence is not None else "",
            "engine": sh.engine or "auto",
            "duration_hint": sh.duration_hint,
            "dialogue": [{"speaker": ln.speaker, "text": ln.text, "emotion": ln.emotion}
                         for ln in sh.lines if ln.text.strip()],
        })
    characters = [_char_dict(ch, anim.voices.get(getattr(ch, "name", ""),
                                                 VoiceProfile(getattr(ch, "name", ""), None)))
                  for ch in anim.characters]
    narr = anim.voices.get("NARRATOR", VoiceProfile("NARRATOR", None))
    return {
        "plan_version": PLAN_VERSION,
        "title": p.title,
        "theme": p.theme,
        "story_idea": getattr(anim, "_story_idea_str", ""),
        "ei_dialogue": bool(p.use_ei_dialogue and anim.graph is not None),
        "characters": characters,
        "narrator": {"voice": {"source": narr.source, "engine": "",
                               "ref_wav": narr.ref_wav,
                               "ref_text": narr.params.get("ref_text", narr.ref_text),
                               "seed": narr.seed, "params": narr.params,
                               "voice_folder": narr.params.get("voice_folder", ""),
                               "voice_folder_id": narr.params.get("voice_folder_id", ""),
                               "voice_description": narr.params.get("voice_description", ""),
                               "voice_file": narr.params.get("voice_file", ""),
                               "baseline_text": narr.baseline_text,
                               "baseline_path": narr.baseline_path,
                               "regenerate_baseline": narr.regenerate_baseline}},
        "scenes": scenes,
        "config": {"project": _project_to_dict(p), "tts": asdict(t), "video": asdict(v)},
    }


def _project_to_dict(p: "ProjectConfig") -> Dict[str, Any]:
    d = asdict(p)
    return d


def write_plan(anim: "StoryAnimator", path: Optional[str] = None) -> str:
    plan = build_plan(anim)
    wd = anim.p.workdir()
    path = path or str(wd / "plan.json")
    Path(path).write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    # human-readable companion
    md = [f"# {plan['title']}", "", f"_Theme:_ {plan['theme']}", ""]
    md.append("## Characters & voices")
    for c in plan["characters"]:
        vo = c["voice"]
        md.append(f"- **{c['name']}** ({c.get('gender','?')}) — voice: "
                  f"`{vo['source']}`"
                  + (f", ref=`{Path(vo['ref_wav']).name}`" if vo['ref_wav'] else "")
                  + (f", desc={vo.get('voice_description','')!r}" if vo.get("voice_description") else ""))
    md.append("")
    for si, sc in enumerate(plan["scenes"], 1):
        md.append(f"## Scene {si} — {sc.get('setting','')}  _( {sc.get('mood','')} )_")
        for sh in sc["shots"]:
            md.append(f"- **Shot {sh['index']}** [{sh['engine']}] — {sh['description'][:90]}")
            for d in sh["dialogue"]:
                md.append(f"    - {d['speaker']} <{d['emotion']}>: {d['text']}")
        md.append("")
    Path(str(Path(path).with_suffix(".md"))).write_text("\n".join(md), encoding="utf-8")
    logger.info("[PLAN] wrote %s (+ .md). Edit it, then run produce_from_plan().", path)
    return path


def load_plan(path_or_dict) -> Dict[str, Any]:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text())


def _plan_to_runtime(plan: Dict[str, Any]):
    """Rebuild (shots, characters, voices, project, tts, video) from a plan."""
    cfg = plan.get("config", {})
    pcfg = ProjectConfig(**{k: v for k, v in cfg.get("project", {}).items()
                            if k in ProjectConfig.__dataclass_fields__})
    tts = TTSConfig(**{k: v for k, v in cfg.get("tts", {}).items()
                       if k in TTSConfig.__dataclass_fields__})
    vcfg = VideoConfig(**{k: v for k, v in cfg.get("video", {}).items()
                          if k in VideoConfig.__dataclass_fields__})
    # shots
    shots: List[Shot] = []
    for sc in plan.get("scenes", []):
        for sh in sc.get("shots", []):
            lines = [Line(speaker=d.get("speaker", ""), text=d.get("text", ""),
                          emotion=d.get("emotion", "neutral"))
                     for d in sh.get("dialogue", []) if d.get("text", "").strip()]
            shot = Shot(index=int(sh.get("index", len(shots))),
                        description=sh.get("description", ""),
                        setting=sh.get("setting", sc.get("setting", "")),
                        mood=sh.get("mood", sc.get("mood", "")),
                        composition=sh.get("composition", "medium_shot"),
                        characters_in_frame=sh.get("characters_in_frame", []) or [],
                        lines=lines,
                        image_prompt=sh.get("image_prompt") or None,
                        motion_prompt=sh.get("motion_prompt") or None,
                        action_sequence=sh.get("action_sequence") if "action_sequence" in sh else None,
                        duration_hint=sh.get("duration_hint"))
            eng = (sh.get("engine") or "auto").lower()
            if eng and eng != "auto":
                shot.engine = eng
            shots.append(shot)
    shots.sort(key=lambda s: s.index)
    # characters + voices
    characters = [_PlanCharacter(c) for c in plan.get("characters", [])]
    voices: Dict[str, VoiceProfile] = {}
    for c in plan.get("characters", []):
        vo = c.get("voice", {}) or {}
        params = dict(vo.get("params", {}) or {})
        for _k in ("voice_folder", "voice_folder_id", "voice_description", "voice_file"):
            if vo.get(_k):
                params[_k] = vo[_k]
        if vo.get("ref_text"):
            params["ref_text"] = vo["ref_text"]
        voices[c["name"]] = VoiceProfile(
            name=c["name"], ref_wav=vo.get("ref_wav"), gender=c.get("gender", ""),
            source=vo.get("source", "engine_default"), seed=int(vo.get("seed", 0)),
            ref_text=vo.get("ref_text", ""), params=params,
            baseline_text=vo.get("baseline_text", ""),
            baseline_path=vo.get("baseline_path"),
            regenerate_baseline=bool(vo.get("regenerate_baseline", False)))
    nvo = (plan.get("narrator", {}) or {}).get("voice", {}) or {}
    nparams = dict(nvo.get("params", {}) or {})
    for _k in ("voice_folder", "voice_folder_id", "voice_description", "voice_file"):
        if nvo.get(_k):
            nparams[_k] = nvo[_k]
    if nvo.get("ref_text"):
        nparams["ref_text"] = nvo["ref_text"]
    voices["NARRATOR"] = VoiceProfile(name="NARRATOR", ref_wav=nvo.get("ref_wav"),
                                      source=nvo.get("source", "engine_default"),
                                      seed=int(nvo.get("seed", 0)),
                                      ref_text=nvo.get("ref_text", ""), params=nparams,
                                      baseline_text=nvo.get("baseline_text", ""),
                                      baseline_path=nvo.get("baseline_path"),
                                      regenerate_baseline=bool(nvo.get("regenerate_baseline", False)))
    return shots, characters, voices, pcfg, tts, vcfg


def _writeback_voice_fields(plan: Dict[str, Any], voices: Dict[str, "VoiceProfile"]) -> None:
    """Patch a loaded plan dict's voice entries with the now-RESOLVED ref_wav
    / baseline_path / ref_text (e.g. a generated baseline clip's real file
    path, instead of the `null` it started as before Phase 2 ran).

    This is what makes the persisted plan.json the actual source of truth
    after a run: open it and you see exactly which file each character's
    voice came from, can listen to it, swap it for a different clip by
    editing ref_wav directly, or flip regenerate_baseline — and a later
    produce_from_plan() resumes from those real files instead of generating
    a fresh one with no record of what was used last time.
    """
    def patch(vo_dict: Dict[str, Any], vp: "VoiceProfile") -> None:
        vo_dict["source"] = vp.source
        vo_dict["ref_wav"] = vp.ref_wav
        vo_dict["ref_text"] = vp.params.get("ref_text", vp.ref_text)
        vo_dict["baseline_path"] = vp.baseline_path
        vo_dict["regenerate_baseline"] = vp.regenerate_baseline
        vo_dict["params"] = vp.params

    for c in plan.get("characters", []):
        vp = voices.get(c.get("name"))
        if vp is not None:
            patch(c.setdefault("voice", {}), vp)
    nvp = voices.get("NARRATOR")
    if nvp is not None:
        patch(plan.setdefault("narrator", {}).setdefault("voice", {}), nvp)


def produce_from_plan(plan_path_or_dict, resume: bool = True,
                      project_override: Optional[ProjectConfig] = None,
                      tts_override: Optional[TTSConfig] = None,
                      video_override: Optional[VideoConfig] = None,
                      run_lipsync: bool = True) -> Dict[str, Any]:
    """PHASE 2 — read the (edited) plan and render audio + images + animation.

    Honours every edit in the plan. Idempotent when resume=True: anything
    already on disk is reused, so you can iterate scene-by-scene cheaply.

    Set run_lipsync=False when iterating voices/audio. This still renders or
    reuses the I2V clips and composes a muxed film, but skips the external
    LatentSync/MuseTalk/Wav2Lip pass so you can regenerate audio first and run
    lip-sync later.
    """
    plan = load_plan(plan_path_or_dict)
    shots, characters, voices, pcfg, tts, vcfg = _plan_to_runtime(plan)
    pcfg = project_override or pcfg
    tts = tts_override or tts
    vcfg = video_override or vcfg

    # Normalize shot granularity before audio synthesis. Every shot should have
    # either narration-only audio or a single visible character's dialogue audio,
    # never a mixed narrator/character/multi-character audio file aimed at one face.
    shots = enforce_visible_speaker_first_dialogue(
        shots,
        enabled=bool(getattr(pcfg, "enforce_visible_speaker_first_dialogue", True)),
    )
    shots = split_shots_on_audio_speaker_change(
        shots,
        enabled=bool(getattr(pcfg, "split_audio_on_speaker_change", True)),
    )

    wd = pcfg.workdir()
    if vcfg.cap_resolution_for_4090:
        cw, ch = _cap_resolution_for_4090(vcfg.width, vcfg.height)
        if (cw, ch) != (vcfg.width, vcfg.height):
            logger.info("[PRODUCE] capping resolution %dx%d → %dx%d for a 4090 "
                       "(VideoConfig.cap_resolution_for_4090=True).",
                       vcfg.width, vcfg.height, cw, ch)
            vcfg.width, vcfg.height = cw, ch
    logger.info("[PRODUCE] %s — %d shots (resume=%s, run_lipsync=%s)",
                pcfg.title, len(shots), resume, run_lipsync)

    synthesize_audio(shots, voices, tts, wd, voice_narration=pcfg.voice_narration, resume=resume)

    # Persist the now-RESOLVED voice info (e.g. a generated baseline clip's
    # real path, previously `null`) back into the plan on disk — so opening
    # plan.json after a run shows exactly which file each voice came from,
    # and a later produce_from_plan() resumes from those real files. Only
    # when given a path (a dict input has nowhere to write back to).
    if isinstance(plan_path_or_dict, (str, Path)):
        _writeback_voice_fields(plan, voices)
        try:
            Path(plan_path_or_dict).write_text(
                json.dumps(plan, indent=2, ensure_ascii=False))
            logger.info("[PRODUCE] wrote resolved voice paths back to %s.", plan_path_or_dict)
        except Exception as e:
            logger.warning("  could not write resolved voices back to %s (%s).",
                           plan_path_or_dict, e)

    # Decide engines + how many re-anchor stills each shot needs, BEFORE the
    # image stage, so those extra stills are generated while KLEIN2 is loaded.
    plan_engines_and_anchors(shots, vcfg)
    generate_stills(shots, characters, pcfg, vcfg, resume=resume)
    animate_shots(shots, vcfg, pcfg, resume=resume)
    sync_report_after_animation = write_sync_report(shots, pcfg, vcfg, label="after_animation")
    if run_lipsync:
        lipsync_shots(shots, vcfg, pcfg, resume=resume)   # re-sync dialogue mouths to TTS speech
        sync_report_after_lipsync = write_sync_report(shots, pcfg, vcfg, label="after_lipsync")
    else:
        logger.info("[LIPSYNC] skipped by produce_from_plan(run_lipsync=False).")
        sync_report_after_lipsync = None
    film = compose_film(shots, pcfg, vcfg)

    manifest = {
        "title": pcfg.title, "film": film, "from_plan": True,
        "run_lipsync": run_lipsync,
        "lipsync_skipped": not run_lipsync,
        "split_audio_on_speaker_change": bool(getattr(pcfg, "split_audio_on_speaker_change", True)),
        "enforce_visible_speaker_first_dialogue": bool(getattr(pcfg, "enforce_visible_speaker_first_dialogue", True)),
        "exclude_dialogue_text_from_visual_prompts": bool(getattr(pcfg, "exclude_dialogue_text_from_visual_prompts", True)),
        "motion_prompts_respect_no_people_scenes": bool(getattr(pcfg, "motion_prompts_respect_no_people_scenes", True)),
        "sync_report_after_animation": sync_report_after_animation,
        "sync_report_after_lipsync": sync_report_after_lipsync,
        "shots": [{"index": s.index, "engine": s.engine, "duration": round(s.duration, 3),
                   "image": s.image_path, "audio": s.audio_path, "video": s.video_path,
                   "speaker": s.speaking_character, "is_dialogue": s.is_dialogue,
                   "action_sequence": s.action_sequence if s.action_sequence is not None else "",
                   "motion_prompt": s.motion_prompt or "",
                   "extra_videos": s.extra_videos} for s in shots],
        "voices": {n: asdict(v) for n, v in voices.items()},
    }
    (Path(wd) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("[PRODUCE] done → %s", film or "(no film)")
    return manifest


# =============================================================================
# SINGLE-IMAGE TALKING VIDEO  ·  one image + (text or audio) → local talking clip
# =============================================================================
# A lightweight alternative to the full story pipeline: no scenes, no cast —
# just "this photo, saying this" → I2V motion + external lip-sync by default.
# Wan-S2V is intentionally disabled in notebook-safe mode because it can crash
# the kernel/computer before Python can fall back.

def produce_talking_image(
    image_path: str,
    text: Optional[str] = None,
    audio_path: Optional[str] = None,
    out_dir: str = "./animation_out/talking_image",
    character_name: str = "Speaker",
    project_title: str = "TalkingImage",
    animation_prompt: str = "Natural, subtle motion; speaking expressively to camera.",
    tts: Optional[TTSConfig] = None,
    video: Optional[VideoConfig] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """One reference image + (text to speak, OR a ready audio file) → one
    local talking clip. Notebook-safe default is I2V + external lip-sync.

    Exactly one of `text` / `audio_path` must be given:
      • text       → synthesized to speech with the configured TTS engine
                      first (use tts.character_refs={character_name: clip} to
                      clone a specific voice; otherwise the engine's bare
                      default voice is used for this one line).
      • audio_path → used as-is (any audio ffmpeg can read).

    The reference image is resized to match ITS OWN resolution, capped to a
    720p-equivalent budget for a 4090 (VideoConfig.cap_resolution_for_4090) —
    a smaller image is left alone, a larger one is downscaled, aspect ratio
    preserved. In notebook-safe mode it never selects Wan-S2V; it uses
    notebook_safe_dialogue_engine / wan_s2v_fallback_engine instead, then
    lipsync_shots() corrects the mouth motion when configured.

    Idempotent like the rest of the pipeline: resume=True reuses the
    synthesized audio / rendered clip / lip-sync output if already on disk.
    """
    if bool(text) == bool(audio_path):
        raise ValueError("Provide exactly ONE of `text` or `audio_path`, not both/neither.")
    tts = tts or TTSConfig()
    video = video or VideoConfig()
    pcfg = ProjectConfig(title=project_title, out_root=str(Path(out_dir).parent) or ".")
    wd = pcfg.workdir()

    # Resolution: match the input image, capped for a 4090.
    image_path = str(_resolve_project_path(image_path))
    img = Image.open(image_path).convert("RGB")
    if video.cap_resolution_for_4090:
        W, H = _cap_resolution_for_4090(img.width, img.height)
    else:
        W, H = (img.width // 16) * 16, (img.height // 16) * 16
    if (W, H) != (img.width, img.height):
        logger.info("[IMAGE2VIDEO] reference %dx%d → %dx%d.", img.width, img.height, W, H)
        img = img.resize((W, H), Image.LANCZOS)
    ref_path = str(wd / "images" / "reference.png")
    img.save(ref_path)
    video.width, video.height = W, H

    # Audio: synthesize from text, or use the supplied file as-is.
    final_audio = str(wd / "audio" / "speech.wav")
    if audio_path:
        if not (resume and Path(final_audio).exists()):
            Path(final_audio).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(_resolve_project_path(audio_path)), final_audio)
    else:
        if not (resume and Path(final_audio).exists()):
            if not _HAS_SF:
                raise RuntimeError("soundfile is not installed — pip install soundfile.")
            engine = make_tts(tts)
            logger.info("[IMAGE2VIDEO] loading TTS engine: %s", engine.name)
            engine.load()
            try:
                ref = tts.character_refs.get(character_name)
                engine.register(character_name, ref, {})
                wav = engine.synth(text, character_name, "neutral")
                wav = _normalize_loudness(wav, tts.target_rms, tts.peak_ceiling) \
                    if tts.loudness_normalize else wav
                wav = _apply_edge_fades(wav, engine.sr, tts.edge_fade_ms)
                Path(final_audio).parent.mkdir(parents=True, exist_ok=True)
                sf.write(final_audio, wav, engine.sr)
            finally:
                engine.unload()
            logger.info("[IMAGE2VIDEO] synthesized speech → %s", final_audio)

    audio_dur = _wav_duration(final_audio)
    if audio_dur <= 0:
        raise RuntimeError(f"Could not read a valid audio duration from {final_audio}.")
    logger.info("[IMAGE2VIDEO] audio is %.2fs — rendering one continuous take to match.", audio_dur)

    shot = Shot(index=0, description=text or "", setting="", mood="",
               composition="medium_shot", characters_in_frame=[character_name],
               lines=[Line(speaker=character_name, text=text or "", emotion="neutral")])
    shot.image_path = ref_path
    shot.audio_path = final_audio
    shot.duration = audio_dur
    shot.engine = route_engine(shot, video)
    shot.motion_prompt = animation_prompt

    animate_shots([shot], video, pcfg, resume=resume)
    lipsync_shots([shot], video, pcfg, resume=resume)   # no-op unless you override lipsync_skip_audio_driven
    film = compose_film([shot], pcfg, video)

    manifest = {"title": pcfg.title, "film": film, "reference_image": ref_path,
               "audio": final_audio, "duration": audio_dur, "shot_video": shot.video_path,
               "animation_prompt": animation_prompt}
    (wd / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("[IMAGE2VIDEO] done → %s", film or "(no film)")
    return manifest


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class StoryAnimator:
    """End-to-end: idea/script → cast → rich dialogue → voices → stills → film."""

    def __init__(self, project: ProjectConfig, tts: Optional[TTSConfig] = None,
                 video: Optional[VideoConfig] = None):
        self.p = project
        self.t = tts or TTSConfig()
        self.v = video or VideoConfig()
        self.characters: List["Character"] = []
        self.shots: List[Shot] = []
        self.voices: Dict[str, VoiceProfile] = {}
        self.story_idea = None
        self.graph = None                        # EI CharacterGraph (subtext engine)
        self.visual_bible = None                 # SeriesVisualBible (prop/costume continuity)
        self._story_idea_str = ""

    # ---- input mode A: a free-text story idea --------------------------------
    @classmethod
    def from_story_idea(cls, story_idea: str, project: ProjectConfig,
                        tts: Optional[TTSConfig] = None,
                        video: Optional[VideoConfig] = None,
                        creative_seeds: Optional[Dict] = None) -> "StoryAnimator":
        if not _HAS_CBG:
            raise RuntimeError("comic_book_generator unavailable — story-idea mode needs it.")
        self = cls(project, tts, video)
        logger.info("[STORY] synthesizing story + cast from idea…")
        result = cbg.synthesize_comic_story(
            story_idea, creative_seeds=creative_seeds or {}, cast_size=project.cast_size)
        # synthesize_comic_story's real return arity has drifted across versions
        # of comic_book_generator.py (4-tuple without a visual bible vs. the
        # current 5-tuple that adds one) — unpack defensively instead of
        # assuming a fixed length.
        visual_bible = None
        if len(result) == 5:
            story_idea_obj, characters, graph, _registry, visual_bible = result
        elif len(result) == 4:
            story_idea_obj, characters, graph, _registry = result
        else:
            raise RuntimeError(
                f"synthesize_comic_story returned {len(result)} values "
                "(expected 4 or 5) — comic_book_generator.py's return signature "
                "has changed again; update from_story_idea() to match.")
        self.story_idea = story_idea_obj
        self.characters = characters
        self.graph = graph                        # ← keep the EI graph (was discarded)
        self.visual_bible = visual_bible

        # ── Resolve target_pages, then extract beats and grow the book to fit ──
        # When target_pages == 'auto' we run the same 4-step pipeline that
        # synthesize_comic_book() uses so the page count is driven entirely by
        # the story's structural needs rather than a fixed default:
        #   1. detect act structure (multi-act concepts size each act separately)
        #   2. lightweight storyboard → initial page estimate
        #   3. extract required beats → reconcile estimate to actual coverage
        #   4. enrichment pass (backstory / side arcs / world-building beats)
        #      → adopt any page growth the enrichment adds
        #   5. normalize act numbering, then snap to the furthest beat's end page
        #
        # For an explicit integer we still run steps 3-5 so that required beats
        # are available for the script and the page count is never starved.
        _ppb = project.panels_per_page_avg
        _was_auto = isinstance(project.target_pages, str) and \
                    project.target_pages.strip().lower() == "auto"

        # Step 1 — act structure (needed by both the estimator and beat extractor)
        _act_structure = cbg._detect_act_structure(story_idea)
        if _act_structure.get('has_explicit_acts'):
            logger.info(
                "[STORY] Detected %d-act structure: %s",
                _act_structure['num_acts'],
                " | ".join(_act_structure.get('act_titles', [])[:8]),
            )

        # Step 2 — resolve the initial page count
        if _was_auto:
            _target_pages = cbg._recommend_target_pages(
                story_idea_obj, story_idea,
                characters=characters,
                panels_per_page_avg=_ppb,
                act_structure=_act_structure,
            )
            logger.info("[STORY] target_pages='auto' → initial estimate: %d pages.", _target_pages)
        else:
            _target_pages = int(project.target_pages)

        # Step 3 — extract required beats and reconcile page count to coverage
        logger.info("[STORY] Extracting required story beats…")
        required_beats = cbg.extract_required_story_beats(
            story_idea_obj, story_idea,
            target_pages=_target_pages,
            act_structure=_act_structure,
            characters=characters,
        )
        if _was_auto:
            _target_pages = cbg._reconcile_target_pages_to_beats(
                required_beats, _target_pages, panels_per_page_avg=_ppb,
            )

        # Step 4 — enrichment (backstory / side arcs / world-building)
        logger.info("[STORY] Adding story enrichment beats…")
        try:
            required_beats, _enrich = cbg.generate_story_enrichment_beats(
                story_idea=story_idea_obj,
                characters=characters,
                story_idea_str=story_idea,
                required_beats=required_beats,
                target_pages=_target_pages,
                get_llm_response=cbg.get_openai_prompt_response,
                parse_json=cbg.parse_json_response,
                sanitize=cbg.sanitize_text_for_prompt,
                llm_model=cbg.openai_model_large,
                use_grok=cbg.USE_GROK,
                panels_per_page_avg=_ppb,
            )
            _grown = int(_enrich.get('target_pages', _target_pages) or _target_pages)
            if _grown > _target_pages:
                logger.info(
                    "[STORY] Page count expanded %d → %d after enrichment (%d beat(s) added).",
                    _target_pages, _grown, _enrich.get('added_count', 0),
                )
                _target_pages = _grown
        except Exception as _e:
            logger.warning("[STORY] Enrichment pass skipped (%s).", _e)

        # Step 5 — normalize act numbering, then snap to the furthest beat end
        required_beats = cbg._normalize_act_structure(required_beats)
        _beat_span = cbg._max_beat_end(required_beats)
        if _beat_span > _target_pages:
            logger.info(
                "[STORY] Growing target_pages %d → %d to cover all beats.",
                _target_pages, _beat_span,
            )
            _target_pages = _beat_span

        # Story doctor: pressure-test and strengthen the premise before committing
        # to the full panel script. Only fires when story_doctor=True in config.
        if getattr(self.p, "story_doctor", True):
            logger.info("[StoryDoctor] Evaluating premise…")
            story_idea = story_doctor_pass(story_idea, characters, self.p)

        logger.info("[STORY] Final page count: %d. Writing panel script…", _target_pages)
        script = cbg.generate_comic_script(
            story_idea_obj, characters, graph,
            target_pages=_target_pages,
            panels_per_page_avg=_ppb,
            required_beats=required_beats,
            story_idea_str=story_idea,
            visual_bible=visual_bible,
        )
        self.shots = shots_from_comic_script(script)
        self._story_idea_str = story_idea
        logger.info("[STORY] %d shots from %d pages.", len(self.shots), len(script))
        return self

    # ---- input mode B: a single character + a written script -----------------
    @classmethod
    def from_script(cls, script: List[Dict], project: ProjectConfig,
                    character: Optional["Character"] = None,
                    characters: Optional[List["Character"]] = None,
                    tts: Optional[TTSConfig] = None,
                    video: Optional[VideoConfig] = None) -> "StoryAnimator":
        self = cls(project, tts, video)
        cast = list(characters or [])
        if character is not None:
            cast.append(character)
        if not cast and _HAS_NG:
            # minimal placeholder so voice-casting + image prompts still work
            cast = [Character(name=(script[0].get("speaker") if script else "Narrator") or "Speaker",
                              age="", role="", traits=[], backstory="")]
        self.characters = cast
        default_name = getattr(cast[0], "name", "Speaker") if cast else "Speaker"
        self.shots = shots_from_user_script(script, default_name)
        logger.info("[SCRIPT] %d shots from provided script.", len(self.shots))
        # Optionally synthesize an EI graph so script-mode casts also get subtext.
        if project.build_ei_graph_for_script and _HAS_NG and cast:
            si = self.story_idea
            if si is None and _HAS_NG:
                try:
                    si = StoryIdea(genre="drama", themes=["relationship"],
                                   mood=project.theme, premise=project.title)
                except Exception:
                    si = None
            logger.info("[SCRIPT] synthesizing EI graph for the cast…")
            self.graph = build_ei_graph_from_characters(cast, si)
        return self

    # ---- PHASE 1 ────────────────────────────────────────────────────────────
    def plan(self, write: bool = True, path: Optional[str] = None,
             do_enrich: Optional[bool] = None) -> str:
        """Build the editable plan: EI dialogue, structured image prompts, voices.

        Loads NO image/video/TTS models — voice assignment here is just
        bookkeeping (bank lookup or a deferred engine-specific note); the
        actual TTS engine loads later, in produce(). Writes plan.json +
        plan.md and returns the plan path. Edit it, then call produce().
        """
        wd = self.p.workdir()

        # ── Prompt-cache routing ─────────────────────────────────────────────
        # Set a single Grok conv-id for this entire plan() run so every LLM
        # call shares one cache partition. The stable prefix blocks (story
        # doctor, visual treatment, arc map, image prompts, prompt review,
        # continuity summaries) are all served from cache after the first hit,
        # cutting input-token cost substantially on multi-batch loops.
        if _HAS_NG and getattr(ng, "USE_GROK", False):
            try:
                _conv = ng.set_grok_conv_id()
                logger.info("[Cache] Grok conv-id for this plan() run: %s", _conv)
            except Exception as _ce:
                logger.debug("[Cache] Could not set Grok conv-id (%s).", _ce)

        # Multi-character scenes: add a few reactive lines for present-but-
        # silent characters BEFORE the EI rewrite, so they get the same
        # subtext/voice polish as everything else.
        balance_scene_dialogue(self.shots, self.characters, self.p.theme, self.p)

        if self.p.use_ei_dialogue and self.graph is not None:
            logger.info("[DIALOGUE] EI subtext rewrite (graph-driven)…")
            EIDialogueEngine(self.graph, self.characters).enrich(self.shots, self.p.theme)
        elif (do_enrich if do_enrich is not None else self.p.enrich_dialogue):
            logger.info("[DIALOGUE] spoken-rewrite pass (no EI graph)…")
            enrich_dialogue_for_voice(self.shots, self.characters)

        direct_shot_composition(self.shots, self.p)
        self.shots = merge_dialogue_shots(self.shots, self.p)
        self.shots = enforce_visible_speaker_first_dialogue(
            self.shots,
            enabled=bool(getattr(self.p, "enforce_visible_speaker_first_dialogue", True)),
        )
        add_voice_cue_tokens(self.shots, self.t)

        craft_opening_hook(self.shots, self._story_idea_str, self.characters, self.p)
        enforce_shot_variety(self.shots, self.p)
        shape_pacing(self.shots, self.p)

        # ── Creative quality passes ──────────────────────────────────────────
        # 2. Visual treatment — author the film's palette/light/texture bible
        logger.info("[VisualTreatment] Generating per-film visual treatment…")
        _visual_treatment = generate_visual_treatment(
            self._story_idea_str or self.p.title,
            self.characters,
            self.p.theme,
            self.p,
        )

        # 3. Emotional arc map — compute the film's intended emotional trajectory
        logger.info("[ArcMap] Computing emotional arc map…")
        _arc_map = generate_emotional_arc_map(
            self._story_idea_str or self.p.title,
            len(self.shots),
            self.p,
        )
        # Cache on pcfg so review_and_revise_image_prompts can also access it
        self.p._arc_map_cache = _arc_map

        # Image prompts — now receive visual treatment, arc map, and continuity flag
        logger.info("[PROMPTS] Authoring structured image prompts…")
        generate_image_prompts(
            self.shots, self.characters, self.p.theme,
            cinematic=self.p.cinematic_prompts,
            expressive=getattr(self.p, "expressive_detail", True),
            visual_treatment=_visual_treatment,
            arc_map=_arc_map,
            use_continuity_context=getattr(self.p, "shot_continuity_context", True),
        )

        # 5. Shot-to-shot continuity summaries — attach prev-shot summaries
        #    AFTER image prompts are drafted so summaries describe real prompts.
        #    These summaries are then used by the prompt review pass below.
        logger.info("[Continuity] Building shot-to-shot continuity summaries…")
        build_shot_continuity_summaries(self.shots, self.p)

        # 4. Prompt review-and-revise — critic pass over the drafted prompts
        logger.info("[PromptReview] Running art-director review pass…")
        review_and_revise_image_prompts(self.shots, _visual_treatment, self.p)

        generate_motion_prompts(self.shots, self.p.theme, self.v,
                                cinematic=self.v.cinematic_motion)

        # ── Cache stats ──────────────────────────────────────────────────────
        if _HAS_NG and getattr(ng, "USE_GROK", False):
            try:
                _cs = ng.get_grok_cache_stats()
                logger.info(
                    "[Cache] plan() Grok usage — calls: %d | prompt tokens: %d | "
                    "cached tokens: %d | hit rate: %.1f%%",
                    _cs.get("calls", 0),
                    _cs.get("prompt_tokens", 0),
                    _cs.get("cached_tokens", 0),
                    _cs.get("cache_hit_rate", 0.0) * 100,
                )
            except Exception:
                pass

        logger.info("[CAST] Assigning voices (mode=%s)…", self.t.voice_mode)
        self.voices = cast_voices(self.characters, self.t, workdir=wd)

        if write:
            return write_plan(self, path)
        return ""

    # ---- PHASE 2 ────────────────────────────────────────────────────────────
    def produce(self, plan_path: Optional[str] = None, resume: bool = True,
                run_lipsync: bool = True) -> Dict[str, Any]:
        """Render from a plan. If plan_path is None, writes one from current state."""
        plan_path = plan_path or write_plan(self)
        return produce_from_plan(plan_path, resume=resume, run_lipsync=run_lipsync)

    # ---- convenience: plan + produce in one call ----------------------------
    def run(self, do_enrich: Optional[bool] = None, resume: bool = True,
            run_lipsync: bool = True) -> Dict[str, Any]:
        plan_path = self.plan(write=True, do_enrich=do_enrich)
        return produce_from_plan(plan_path, resume=resume, run_lipsync=run_lipsync)
		


# =============================================================================
# NOTEBOOK-FIRST HELPERS
# =============================================================================

def configure_notebook_local_paths(project_root: str = ".",
                                   model_cache_dir: str = "./model_cache",
                                   animation_out_dir: str = "./animation_out",
                                   latentsync_repo_dir: str = "./model_cache/LatentSync",
                                   chdir: bool = True) -> Dict[str, Any]:
    """Configure local notebook paths once; prompts/configs stay in the notebook."""
    project_root_p = Path(project_root).expanduser().resolve()
    model_cache_p = _resolve_project_path(model_cache_dir, base=project_root_p)
    animation_out_p = _resolve_project_path(animation_out_dir, base=project_root_p)
    latentsync_p = _resolve_project_path(latentsync_repo_dir, base=project_root_p)

    if chdir:
        os.chdir(project_root_p)
    os.environ["STORY_ANIMATION_PROJECT_ROOT"] = str(project_root_p)

    for q in (project_root_p, latentsync_p):
        s = str(q)
        if s not in sys.path:
            sys.path.insert(0, s)

    old_pp = os.environ.get("PYTHONPATH", "")
    pp_parts = [str(project_root_p), str(latentsync_p)]
    if old_pp:
        pp_parts.append(old_pp)
    os.environ["PYTHONPATH"] = os.pathsep.join(pp_parts)

    model_cache_p.mkdir(parents=True, exist_ok=True)
    animation_out_p.mkdir(parents=True, exist_ok=True)
    whisper_layout = ensure_latentsync_whisper_layout(str(latentsync_p)) if latentsync_p.exists() else {}

    return {
        "project_root": str(project_root_p),
        "model_cache_dir": str(model_cache_p),
        "animation_out_dir": str(animation_out_p),
        "latentsync_repo_dir": str(latentsync_p),
        "latentsync_package_exists": (latentsync_p / "latentsync").exists(),
        "latentsync_unet_exists": (latentsync_p / "latentsync" / "models" / "unet.py").exists(),
        "latentsync_inference_exists": (latentsync_p / "scripts" / "inference.py").exists(),
        "latentsync_stage2_512_exists": (latentsync_p / "configs" / "unet" / "stage2_512.yaml").exists(),
        "latentsync_tiny_expected_exists": (latentsync_p / "checkpoints" / "whisper" / "tiny.pt").exists(),
        "latentsync_tiny_source_exists": (latentsync_p / "whisper" / "tiny.pt").exists(),
        "latentsync_whisper_layout": whisper_layout,
    }


def _make_video_config_filtered(**kwargs) -> "VideoConfig":
    """Construct VideoConfig while ignoring stale kwargs from older patches.

    This keeps notebook preset helpers robust when engines/config fields are
    removed, such as the WAN+FramePack-only cleanup.
    """
    try:
        import dataclasses as _dc
        allowed = {f.name for f in _dc.fields(VideoConfig)}
        return VideoConfig(**{k: v for k, v in kwargs.items() if k in allowed})
    except Exception:
        return VideoConfig(**kwargs)



def notebook_safe_latentsync16_config_from_paths(paths: Dict[str, Any],
                                                 width: int = 960,
                                                 height: int = 544,
                                                 fps: int = 24,
                                                 fallback_engine: str = "wan_i2v",
                                                 wan_i2v_steps: int = 32,
                                                 wan_i2v_guidance_scale: float = 5.0,
                                                 lipsync_extra_args: str = "--inference_steps 25 --guidance_scale 1.5",
                                                 require_lipsync: bool = True) -> VideoConfig:
    """Create a notebook-safe 4090 video config from configure_notebook_local_paths()."""
    repo = Path(paths["latentsync_repo_dir"]).resolve()
    ensure_latentsync_whisper_layout(str(repo))
    cfg_path = repo / "configs" / "unet" / "stage2_512.yaml"
    if not cfg_path.exists():
        cfg_path = repo / "configs" / "unet" / "stage2.yaml"

    return _make_video_config_filtered(
        routing="auto",
        single_engine=fallback_engine,
        width=width,
        height=height,
        fps=fps,
        cache_dir=str(paths.get("model_cache_dir", "./model_cache")),
        wan_s2v_disable=True,
        notebook_safe_mode=True,
        notebook_safe_dialogue_engine=fallback_engine,
        wan_s2v_fallback_engine=fallback_engine,
        try_sage_attention=True,
        use_fp8=True,
        wan_i2v_steps=wan_i2v_steps,
        wan_i2v_guidance_scale=wan_i2v_guidance_scale,
        fp_steps=32,
        fp_use_teacache=False,
        fp_internal_fps=30,
        fp_force_exact_duration=True,
        framepack_min_seconds=6.0,
        dialogue_long_takes_use_framepack=True,
        lipsync_engine="latentsync",
        lipsync_repo_dir=str(repo),
        lipsync_checkpoint=str(repo / "checkpoints" / "latentsync_unet.pt"),
        lipsync_unet_config=str(cfg_path),
        lipsync_extra_args=lipsync_extra_args,
        exact_audio_video_duration=True,
        write_sync_report=True,
        require_lipsync_for_dialogue_fallback=require_lipsync,
    )


def create_animator_from_prompt(story_prompt: str,
                                project: ProjectConfig,
                                tts: Optional[TTSConfig] = None,
                                video: Optional[VideoConfig] = None) -> "StoryAnimator":
    return StoryAnimator.from_story_idea(story_prompt, project, tts=tts, video=video)


def plan_story_from_prompt(story_prompt: str,
                           project: ProjectConfig,
                           tts: Optional[TTSConfig] = None,
                           video: Optional[VideoConfig] = None,
                           write: bool = True) -> str:
    anim = create_animator_from_prompt(story_prompt, project, tts=tts, video=video)
    return anim.plan(write=write)


def run_story_from_prompt(story_prompt: str,
                          project: ProjectConfig,
                          tts: Optional[TTSConfig] = None,
                          video: Optional[VideoConfig] = None,
                          resume: bool = False,
                          run_lipsync: bool = True) -> Dict[str, Any]:
    plan_path = plan_story_from_prompt(story_prompt, project, tts=tts, video=video, write=True)
    return produce_from_plan(plan_path, video_override=video, resume=resume, run_lipsync=run_lipsync)


def render_talking_avatar_from_audio(image_path: str,
                                     audio_path: str,
                                     animation_prompt: str = "In the video, a woman is singing. Her expression is very lyrical and intoxicated with music.",
                                     out_dir: str = "./animation_out/talking_avatar_audio",
                                     character_name: str = "Speaker",
                                     project_title: str = "TalkingAvatarAudio",
                                     video: Optional[VideoConfig] = None,
                                     resume: bool = False) -> Dict[str, Any]:
    """Image + supplied audio → I2V animation → LatentSync lip-sync."""
    return produce_talking_image(
        image_path=image_path,
        audio_path=audio_path,
        out_dir=out_dir,
        character_name=character_name,
        project_title=project_title,
        animation_prompt=animation_prompt,
        video=video,
        resume=resume,
    )


def render_talking_avatar_from_text(image_path: str,
                                    text: str,
                                    voice_baseline_wav: str,
                                    animation_prompt: str = "In the video, a woman is singing. Her expression is very lyrical and intoxicated with music.",
                                    out_dir: str = "./animation_out/talking_avatar_text",
                                    character_name: str = "Speaker",
                                    project_title: str = "TalkingAvatarText",
                                    tts: Optional[TTSConfig] = None,
                                    video: Optional[VideoConfig] = None,
                                    resume: bool = False) -> Dict[str, Any]:
    """Image + text + baseline voice wav → cloned TTS → I2V → LatentSync."""
    tts = tts or TTSConfig(engine="higgs", voice_mode="clone")
    tts.character_refs[character_name] = str(_resolve_project_path(voice_baseline_wav))
    return produce_talking_image(
        image_path=image_path,
        text=text,
        out_dir=out_dir,
        character_name=character_name,
        project_title=project_title,
        animation_prompt=animation_prompt,
        tts=tts,
        video=video,
        resume=resume,
    )


# =============================================================================
# CLI / example entry point
# =============================================================================
# Importing this module should never start a render. The sample project below is
# now guarded so you can safely `import story_to_animation` from notebooks/tests.


def _demo_lighthouse(resume: bool = True) -> Dict[str, Any]:
    raise RuntimeError(
        "The hard-coded demo was removed. Pass your story prompt and configs "
        "from the notebook using plan_story_from_prompt(...) or run_story_from_prompt(...)."
    )

def _main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "Usage:\n"
            f"  python {sys.argv[0]} produce <plan.json> [--no-resume]\n"
            f"  python {sys.argv[0]} demo [--no-resume]\n"
        )
        return
    cmd = argv.pop(0).lower()
    resume = "--no-resume" not in argv
    argv = [a for a in argv if a != "--no-resume"]
    if cmd == "produce":
        if not argv:
            raise SystemExit("produce requires a plan.json path")
        manifest = produce_from_plan(argv[0], resume=resume)
        print(f"\nFilm: {manifest['film']}")
    elif cmd == "demo":
        _demo_lighthouse(resume=resume)
    else:
        raise SystemExit(f"Unknown command {cmd!r}; expected produce or demo")


if __name__ == "__main__":
    _main()

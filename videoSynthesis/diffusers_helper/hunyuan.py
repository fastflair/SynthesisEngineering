import torch

from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers_helper.utils import crop_or_pad_yield_mask


@torch.no_grad()
def encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, max_length=256):
    assert isinstance(prompt, str)
    prompt = [prompt]

    # ---- LLaMA tokenizer formatting ----
    prompt_llama = [DEFAULT_PROMPT_TEMPLATE["template"].format(p) for p in prompt]
    crop_start = DEFAULT_PROMPT_TEMPLATE["crop_start"]

    llama_inputs = tokenizer(
        prompt_llama,
        padding="max_length",
        max_length=max_length + crop_start,
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    llama_input_ids = llama_inputs.input_ids.to(text_encoder.device)
    llama_attention_mask = llama_inputs.attention_mask.to(text_encoder.device)

    # Length of the *unpadded* region (batch size is 1 here)
    llama_attention_length = int(llama_attention_mask.sum().item())

    # ---- Force hidden states + dict outputs; be robust to wrappers ----
    try:
        if hasattr(text_encoder, "config"):
            text_encoder.config.output_hidden_states = True  # ensure model wants to return them
        llama_outputs = text_encoder(
            input_ids=llama_input_ids,
            attention_mask=llama_attention_mask,
            output_hidden_states=True,
            return_dict=True,                     # <— critical
        )
    except TypeError:
        # Some wrappers don’t accept return_dict; retry without but still fetch as attribute/tuple
        llama_outputs = text_encoder(
            input_ids=llama_input_ids,
            attention_mask=llama_attention_mask,
            output_hidden_states=True,
        )

    # Prefer penultimate(-3) layer (as before); fall back to last_hidden_state if needed
    hidden_states = getattr(llama_outputs, "hidden_states", None)
    if hidden_states is None:
        # tuple-style output fallback: (last_hidden_state, ..., hidden_states, ...)
        if isinstance(llama_outputs, (tuple, list)) and len(llama_outputs) >= 3:
            hidden_states = llama_outputs[2]
    if hidden_states is None:
        # Last resort: use last_hidden_state (slightly different semantics but unblocks)
        base_hidden = getattr(llama_outputs, "last_hidden_state", None)
        if base_hidden is None and isinstance(llama_outputs, (tuple, list)) and len(llama_outputs) >= 1:
            base_hidden = llama_outputs[0]
        if base_hidden is None:
            raise RuntimeError("LLaMA encoder did not return hidden states or last_hidden_state.")
        chosen = base_hidden
    else:
        chosen = hidden_states[-3]  # what your code originally used

    # Apply crop consistent with the template
    llama_vec = chosen[:, crop_start:llama_attention_length]
    llama_attn_mask_cropped = llama_attention_mask[:, crop_start:llama_attention_length]
    assert torch.all(llama_attn_mask_cropped.bool()), "Unexpected padding in cropped region."

    # ---- CLIP-L encoder (pooler fallback) ----
    clip_l_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    clip_out = text_encoder_2(
        clip_l_inputs.to(text_encoder_2.device),
        output_hidden_states=False,
        return_dict=True,
    )
    clip_l_pooler = getattr(clip_out, "pooler_output", None)
    if clip_l_pooler is None:
        # Use [CLS] token as a robust fallback
        clip_l_pooler = clip_out.last_hidden_state[:, 0]

    return llama_vec, clip_l_pooler


@torch.no_grad()
def vae_decode_fake(latents):
    latent_rgb_factors = [
        [-0.0395, -0.0331, 0.0445],
        [0.0696, 0.0795, 0.0518],
        [0.0135, -0.0945, -0.0282],
        [0.0108, -0.0250, -0.0765],
        [-0.0209, 0.0032, 0.0224],
        [-0.0804, -0.0254, -0.0639],
        [-0.0991, 0.0271, -0.0669],
        [-0.0646, -0.0422, -0.0400],
        [-0.0696, -0.0595, -0.0894],
        [-0.0799, -0.0208, -0.0375],
        [0.1166, 0.1627, 0.0962],
        [0.1165, 0.0432, 0.0407],
        [-0.2315, -0.1920, -0.1355],
        [-0.0270, 0.0401, -0.0821],
        [-0.0616, -0.0997, -0.0727],
        [0.0249, -0.0469, -0.1703]
    ]  # From comfyui

    latent_rgb_factors_bias = [0.0259, -0.0192, -0.0761]

    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.clamp(0.0, 1.0)

    return images


@torch.no_grad()
def vae_decode(latents, vae, image_mode=False):
    latents = latents / vae.config.scaling_factor

    if not image_mode:
        image = vae.decode(latents.to(device=vae.device, dtype=vae.dtype)).sample
    else:
        latents = latents.to(device=vae.device, dtype=vae.dtype).unbind(2)
        image = [vae.decode(l.unsqueeze(2)).sample for l in latents]
        image = torch.cat(image, dim=2)

    return image


@torch.no_grad()
def vae_encode(image, vae):
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents

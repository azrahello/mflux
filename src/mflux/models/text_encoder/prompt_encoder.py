import mlx.core as mx

from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5


class PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompts: dict,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: TokenizerT5,
        clip_tokenizer: TokenizerCLIP,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
    ) -> tuple[mx.array, mx.array]:
        prompt_t5 = prompts.get("prompt_t5")
        if not prompt_t5:
            raise ValueError("'prompt_t5' is required.")

        prompt_clip = prompts.get("prompt_clip", None)
        if prompt_clip is not None and not isinstance(prompt_clip, str):
            raise TypeError("'prompt_clip' must be a string or None.")

        if not isinstance(prompt_t5, str):
            prompt_t5 = str(prompt_t5)

        if prompt_clip is not None and not isinstance(prompt_clip, str):
            raise TypeError("'prompt_clip' must be a string or None.")

        if prompt_t5:
            t5_tokens = t5_tokenizer.tokenize(prompt_t5)
            prompt_embeds = t5_text_encoder(t5_tokens)
        else:
            prompt_embeds = None

        if prompt_clip is not None:
            clip_tokens = clip_tokenizer.tokenize(prompt_clip)
            pooled_prompt_embeds = clip_text_encoder(clip_tokens)
        else:
            if prompt_t5 is not None:
                # Use the same prompt for CLIP if 'prompt_clip' is not provided
                clip_tokens = clip_tokenizer.tokenize(prompt_t5)
                pooled_prompt_embeds = clip_text_encoder(clip_tokens)
            else:
                pooled_prompt_embeds = None

        # store results in cache
        prompt_cache[str(prompts)] = (prompt_embeds, pooled_prompt_embeds)
        return prompt_embeds, pooled_prompt_embeds

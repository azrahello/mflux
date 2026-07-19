import mlx.core as mx
from PIL import Image

from mflux.models.common.tokenizer import Tokenizer, VisionLanguageTokenizer
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import KREA2_TEMPLATE, Krea2TextEncoder


class Krea2PromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: Krea2TextEncoder,
    ) -> mx.array:
        tokens = tokenizer.tokenize(prompt)
        return text_encoder.get_prompt_embeds(tokens.input_ids, tokens.attention_mask)

    @staticmethod
    def encode_prompt_pair(
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        tokenizer: Tokenizer,
        text_encoder: Krea2TextEncoder,
        prompt_cache: dict[tuple[str, str | None, float], tuple[mx.array, mx.array | None]],
    ) -> tuple[mx.array, mx.array | None]:
        cache_key = (prompt, negative_prompt, guidance)
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        embeds = Krea2PromptEncoder.encode_prompt(prompt, tokenizer, text_encoder)
        neg_embeds = None
        if guidance != 1.0:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            neg_embeds = Krea2PromptEncoder.encode_prompt(neg, tokenizer, text_encoder)

        mx.eval(embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)
        result = (embeds, neg_embeds)
        prompt_cache[cache_key] = result
        return result

    @staticmethod
    def encode_prompt_pair_with_images(
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        images: list[Image.Image],
        tokenizer: Tokenizer,
        vision_tokenizer: VisionLanguageTokenizer,
        text_encoder: Krea2TextEncoder,
    ) -> tuple[mx.array, mx.array | None]:
        # Reference images make the conditioning too expensive/awkward to key a
        # cache on (unhashable PIL images); always re-encode when images are set.
        tokens = vision_tokenizer.tokenize(prompt, images=images)
        embeds = text_encoder.get_prompt_embeds(
            tokens.input_ids,
            tokens.attention_mask,
            pixel_values=tokens.pixel_values,
            image_grid_thw=tokens.image_grid_thw,
        )

        neg_embeds = None
        if guidance != 1.0:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            # Negative conditioning stays text-only: CFG should push away from a
            # generic unconditioned prompt, not away from "the reference image".
            neg_embeds = Krea2PromptEncoder.encode_prompt(neg, tokenizer, text_encoder)

        mx.eval(embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)
        return embeds, neg_embeds

    @staticmethod
    def encode_edit_prompt_pair(
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        images: list[Image.Image],
        tokenizer: Tokenizer,
        vision_tokenizer: VisionLanguageTokenizer,
        text_encoder: Krea2TextEncoder,
    ) -> tuple[mx.array, mx.array | None]:
        # In-context edit (ai-toolkit / Ostris): the references enter under the BASE
        # t2i template with "Picture N:" vision markers — the layout the edit LoRAs
        # saw in training — NOT the image-edit descriptor template used by --img-ref.
        edit_tokenizer = VisionLanguageTokenizer(
            tokenizer=vision_tokenizer.tokenizer,
            processor=vision_tokenizer.processor,
            max_length=vision_tokenizer.max_length,
            template=KREA2_TEMPLATE,
            image_token=vision_tokenizer.image_token,
        )
        # No "Picture N:" label: this LoRA was trained on bare vision markers
        # (matching the reference ComfyUI node's KREA2_EDIT_TEMPLATE), not the
        # multi-image-labeled convention --img-ref uses.
        tokens = edit_tokenizer.tokenize(prompt, images=images, label_images=False)
        embeds = text_encoder.get_prompt_embeds(
            tokens.input_ids,
            tokens.attention_mask,
            pixel_values=tokens.pixel_values,
            image_grid_thw=tokens.image_grid_thw,
        )

        neg_embeds = None
        if guidance != 1.0:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            neg_embeds = Krea2PromptEncoder.encode_prompt(neg, tokenizer, text_encoder)

        mx.eval(embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)
        return embeds, neg_embeds

    @staticmethod
    def encode_edit_rebalance_prompts(
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        images: list[Image.Image],
        tokenizer: Tokenizer,
        vision_tokenizer: VisionLanguageTokenizer,
        text_encoder: Krea2TextEncoder,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array | None]:
        # The three encodings the edit-rebalance recipe combines: the main prompt
        # without images (still under the edit system template), the main prompt
        # with the reference images, and the negative prompt (or empty) with the
        # same images. All three go through the vision tokenizer so they share
        # the edit template, matching the reference node's compile_edit calls.
        def encode(text: str, imgs: list[Image.Image] | None) -> mx.array:
            tokens = vision_tokenizer.tokenize(text, images=imgs)
            return text_encoder.get_prompt_embeds(
                tokens.input_ids,
                tokens.attention_mask,
                pixel_values=tokens.pixel_values,
                image_grid_thw=tokens.image_grid_thw,
            )

        ref_prompt = negative_prompt if negative_prompt and negative_prompt.strip() else ""
        cond_raw = encode(prompt, None)
        cond_image_main = encode(prompt, images)
        cond_image_ref = encode(ref_prompt, images)

        neg_embeds = None
        if guidance != 1.0:
            neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "
            neg_embeds = Krea2PromptEncoder.encode_prompt(neg, tokenizer, text_encoder)

        mx.eval(cond_raw, cond_image_main, cond_image_ref)
        if neg_embeds is not None:
            mx.eval(neg_embeds)
        return cond_raw, cond_image_main, cond_image_ref, neg_embeds

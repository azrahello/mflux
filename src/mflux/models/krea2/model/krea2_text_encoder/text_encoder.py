import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding import Qwen3TextRotaryEmbedding

# tap k == HF hidden_states[k] (index 0 = embeddings), 12 layers.
KREA2_TAP_LAYERS: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Qwen2 tokenizer ids used to locate the chat-template prefix to strip.
_IM_START, _USER, _NEWLINE = 151644, 872, 198

KREA2_IMAGE_TOKEN_ID = 151655


# Same system template as Qwen-Image; the system+user-opening prefix is stripped
# from the tapped hidden states in get_prompt_embeds (see _template_end).
KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Vision-conditioned variant: same system prompt, adds "Picture N: <vision tokens>"
# markers ahead of the user text (see Krea2PromptEncoder for how images are formatted in).
KREA2_IMAGE_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Krea2TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_token_id = KREA2_IMAGE_TOKEN_ID
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                mrope_section=None,
                attention_bias=False,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        # Qwen3-VL-4B vision tower (matches krea/Krea-2-Turbo's text_encoder/config.json
        # vision_config). Weights ship inside the same checkpoint as `visual.*` keys.
        self.visual = Qwen3VLVisionModel(
            patch_size=16,
            temporal_patch_size=2,
            hidden_size=1024,
            num_heads=16,
            intermediate_size=4096,
            depth=24,
            spatial_merge_size=2,
            num_position_embeddings=2304,
            out_hidden_size=2560,
            deepstack_visual_indexes=[5, 11, 17],
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> list[mx.array]:
        # Returns the HF-style hidden-state list: [embeddings, layer0_out, ..., layer35_out].
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            hidden_states = self._merge_image_features(hidden_states, input_ids, pixel_values, image_grid_thw)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )[:, None, None, :]
        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = idx[None, :] > idx[:, None]
        causal_mask = mx.where(
            causal,
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_mask + padding_mask

        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states_list = [hidden_states]  # index 0 = embeddings (HF convention)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask_4d, position_embeddings, past_key_value=None)
            hidden_states_list.append(hidden_states)
        return hidden_states_list

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        tap_layers: tuple[int, ...] = KREA2_TAP_LAYERS,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        # Layer-major flatten: stack taps -> (B, seq, n, h) -> (B, seq, n*h)
        hidden_states_list = self(input_ids, attention_mask, pixel_values, image_grid_thw)
        stacked = mx.stack([hidden_states_list[i] for i in tap_layers], axis=2)
        b, s, n, h = stacked.shape
        embeds = stacked.reshape(b, s, n * h)
        # Strip the system + user-opening chat-template prefix from the conditioning
        # (matches ComfyUI), so only the actual prompt tokens onward condition the DiT.
        end = Krea2TextEncoder._template_end(input_ids)
        return embeds[:, end:, :]

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> list[mx.array]:
        image_embeds, _ = self.visual(pixel_values, image_grid_thw, return_deepstack=False)
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        split_sizes = (original_split_sizes // (self.visual.spatial_merge_size**2)).astype(mx.int32)
        split_sizes = [s for s in split_sizes.tolist() if s > 0]

        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        return image_embeds_split

    def _merge_image_features(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
    ) -> mx.array:
        image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = mx.concatenate(image_embeds_split, axis=0)

        image_positions = input_ids == self.image_token_id
        n_image_tokens = mx.sum(image_positions).item()
        if n_image_tokens == 0 or image_embeds.shape[0] < n_image_tokens:
            return inputs_embeds

        image_positions_flat = image_positions.flatten()
        inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])

        new_embeds_list = []
        image_idx = 0
        for i in range(len(image_positions_flat)):
            if image_positions_flat[i] and image_idx < image_embeds.shape[0]:
                new_embeds_list.append(image_embeds[image_idx])
                image_idx += 1
            else:
                new_embeds_list.append(inputs_embeds_flat[i])

        new_embeds = mx.stack(new_embeds_list, axis=0)
        return new_embeds.reshape(inputs_embeds.shape)

    @staticmethod
    def _template_end(input_ids: mx.array) -> int:
        ids = input_ids[0].tolist()
        count = 0
        end = 0
        for i, tok in enumerate(ids):
            if tok == _IM_START and count < 2:
                end = i
                count += 1
        if len(ids) > end + 3 and ids[end + 1] == _USER and ids[end + 2] == _NEWLINE:
            end += 3
        return end

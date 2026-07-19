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
        # mrope_section [24, 20, 20] (temporal/height/width channel counts, matching
        # krea/Krea-2-Turbo's text_encoder/config.json rope_parameters): image patch
        # tokens get 2D spatial rotary positions instead of a flat sequential index.
        self.rotary_emb = Qwen3TextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            mrope_section=[24, 20, 20],
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

        visual_pos_masks = None
        deepstack_features = None
        if pixel_values is not None and image_grid_thw is not None:
            hidden_states, visual_pos_masks, deepstack_features = self._merge_image_features(
                hidden_states, input_ids, pixel_values, image_grid_thw
            )

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

        if image_grid_thw is not None and pixel_values is not None:
            position_ids = Krea2TextEncoder._build_mrope_position_ids(
                input_ids, self.image_token_id, image_grid_thw, self.visual.spatial_merge_size
            )
        else:
            position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states_list = [hidden_states]  # index 0 = embeddings (HF convention)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, _ = layer(hidden_states, attention_mask_4d, position_embeddings, past_key_value=None)
            # HF's hidden_states[layer_idx + 1] snapshots the state right after the
            # layer's own transform but BEFORE that iteration's deepstack add -- append
            # here first so KREA2_TAP_LAYERS lines up with the taps the model was
            # trained against. The deepstack add still has to feed the NEXT layer's
            # input (it's a real architectural step, just not the reported snapshot).
            hidden_states_list.append(hidden_states)
            # Deepstack: the vision tower's intermediate (not just final) features get
            # re-added at the visual token positions after each of the first few text
            # layers -- required by Qwen3-VL's architecture (config: deepstack_visual_
            # _indexes), not an optional extra. Skipping it starves exactly the early
            # KREA2_TAP_LAYERS (tap 0 = layer 2) of visual signal they're meant to carry.
            if deepstack_features is not None and layer_idx < len(deepstack_features):
                hidden_states = Krea2TextEncoder._scatter_at_visual_positions(
                    hidden_states, visual_pos_masks, deepstack_features[layer_idx], add=True
                )
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

    def get_image_features(
        self, pixel_values: mx.array, image_grid_thw: mx.array
    ) -> tuple[list[mx.array], list[mx.array]]:
        image_embeds, deepstack_features = self.visual(pixel_values, image_grid_thw, return_deepstack=True)
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        split_sizes = (original_split_sizes // (self.visual.spatial_merge_size**2)).astype(mx.int32)
        split_sizes = [s for s in split_sizes.tolist() if s > 0]

        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        # deepstack_features stay unsplit (one tensor per layer, across all images) --
        # they're consumed in that same flattened order by _scatter_at_visual_positions,
        # matching how the reference implementation uses them.
        return image_embeds_split, deepstack_features

    def _merge_image_features(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        pixel_values: mx.array,
        image_grid_thw: mx.array,
    ) -> tuple[mx.array, mx.array | None, list[mx.array] | None]:
        image_embeds_split, deepstack_features = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = mx.concatenate(image_embeds_split, axis=0)

        image_positions = input_ids == self.image_token_id
        n_image_tokens = mx.sum(image_positions).item()
        if n_image_tokens == 0 or image_embeds.shape[0] < n_image_tokens:
            return inputs_embeds, None, None

        new_embeds = Krea2TextEncoder._scatter_at_visual_positions(inputs_embeds, image_positions, image_embeds)
        return new_embeds, image_positions, deepstack_features

    @staticmethod
    def _scatter_at_visual_positions(base: mx.array, mask: mx.array, values: mx.array, add: bool = False) -> mx.array:
        # Places (or adds, for deepstack) `values` (N, H) into `base` (B, S, H) at the
        # True positions of `mask` (B, S), in order.
        mask_flat = mask.flatten()
        base_flat = base.reshape(-1, base.shape[-1])

        new_rows = []
        v_idx = 0
        for i in range(mask_flat.shape[0]):
            if mask_flat[i] and v_idx < values.shape[0]:
                new_rows.append(base_flat[i] + values[v_idx] if add else values[v_idx])
                v_idx += 1
            else:
                new_rows.append(base_flat[i])

        return mx.stack(new_rows, axis=0).reshape(base.shape)

    @staticmethod
    def _build_mrope_position_ids(
        input_ids: mx.array, image_token_id: int, image_grid_thw: mx.array, spatial_merge_size: int
    ) -> mx.array:
        # Port of Qwen3VLModel.get_rope_index / get_vision_position_ids (transformers):
        # text tokens get a flat sequential index on all 3 axes; each image's patch
        # tokens get 2D (height, width) positions offset by the running "time" index.
        # After an image, the running index advances by max(gh, gw) -- the image's
        # spatial footprint -- not by its token count, so trailing text (the actual
        # edit instruction) resumes at a much smaller position than a naive sequential
        # scheme would give it.
        assert input_ids.shape[0] == 1, "mrope position-id builder assumes batch size 1"
        ids = input_ids[0].tolist()
        grids = image_grid_thw.tolist()
        grid_iter = iter(grids)

        pos_t: list[int] = []
        pos_h: list[int] = []
        pos_w: list[int] = []
        current_pos = 0
        i = 0
        n = len(ids)
        while i < n:
            if ids[i] == image_token_id:
                t, h, w = next(grid_iter)
                gh, gw = h // spatial_merge_size, w // spatial_merge_size
                for tt in range(t):
                    for hh in range(gh):
                        for ww in range(gw):
                            pos_t.append(current_pos + tt)
                            pos_h.append(current_pos + hh)
                            pos_w.append(current_pos + ww)
                current_pos += max(gh, gw)
                i += t * gh * gw
            else:
                pos_t.append(current_pos)
                pos_h.append(current_pos)
                pos_w.append(current_pos)
                current_pos += 1
                i += 1

        return mx.array([[pos_t], [pos_h], [pos_w]], dtype=mx.int32)

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

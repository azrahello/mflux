import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding import Qwen3TextRotaryEmbedding
from mflux.models.ideogram4.model.ideogram4_text_encoder.decoder_layer import Qwen3VLDecoderLayer

IDEOGRAM4_IMAGE_TOKEN_ID = 151655


class Qwen3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        vision_patch_size: int = 16,
        vision_temporal_patch_size: int = 2,
        vision_hidden_size: int = 1152,
        vision_num_heads: int = 16,
        vision_intermediate_size: int = 4304,
        vision_depth: int = 27,
        vision_spatial_merge_size: int = 2,
        vision_num_position_embeddings: int = 2304,
        vision_out_hidden_size: int = 4096,
        vision_deepstack_visual_indexes: list[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = IDEOGRAM4_IMAGE_TOKEN_ID
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
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
        # Qwen3-VL-8B vision tower (matches ideogram-ai/ideogram-4-fp8's
        # text_encoder/config.json vision_config). Blocks/merger Linear weights
        # ship fp8 (visual.blocks.*, visual.merger.*, visual.deepstack_merger_list.*);
        # patch_embed/pos_embed/norms ship bf16. See Ideogram4WeightDefinition for
        # the dequantization applied at load time.
        self.visual = Qwen3VLVisionModel(
            patch_size=vision_patch_size,
            temporal_patch_size=vision_temporal_patch_size,
            hidden_size=vision_hidden_size,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            depth=vision_depth,
            spatial_merge_size=vision_spatial_merge_size,
            num_position_embeddings=vision_num_position_embeddings,
            out_hidden_size=vision_out_hidden_size,
            deepstack_visual_indexes=vision_deepstack_visual_indexes or [8, 16, 24],
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        tap_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35),
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> list[mx.array]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            hidden_states = self._merge_image_features(hidden_states, input_ids, pixel_values, image_grid_thw)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(seq_len, dtype=mx.int32)[None, :],
                (batch_size, seq_len),
            )

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = padding_mask[:, None, None, :]

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = idx[None, :] > idx[:, None]
        causal_mask = mx.where(
            causal,
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_mask + padding_mask
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        tap_set = set(tap_layers)
        captured: dict[int, mx.array] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states
        return [captured[i] for i in tap_layers]

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        position_ids: mx.array,
        tap_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35),
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        selected = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            tap_layers=tap_layers,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        stacked = mx.stack(selected, axis=0)
        stacked = mx.transpose(stacked, (1, 2, 3, 0))
        batch_size, seq_len, hidden_dim, num_layers = stacked.shape
        return stacked.reshape(batch_size, seq_len, hidden_dim * num_layers)

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

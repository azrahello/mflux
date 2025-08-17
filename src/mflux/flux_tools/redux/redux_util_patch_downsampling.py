from pathlib import Path
from typing import Literal

import mlx.core as mx

from mflux.models.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.post_processing.image_util import ImageUtil


class ReduxUtilDownsampling:
    """
    True downsampling implementation that matches the ComfyUI approach:
    - Takes 27x27 patches and groups them into blocks
    - Averages each block to reduce token count
    - Uses proper MLX operations for efficiency
    - Supports multiple interpolation modes like ComfyUI
    """

    @staticmethod
    def embed_images(
        image_paths: list[str] | list[Path],
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
        image_strengths: list[float] | None = None,
        redux_mode: Literal["lowest", "low", "medium", "high", "highest"] = "medium",
        interpolation_mode: Literal["area", "bicubic", "nearest"] = "area",
    ) -> list[mx.array]:
        """
        Embed images using true patch downsampling like ComfyUI implementation.

        Args:
            image_paths: List of image paths
            image_encoder: SigLIP vision transformer
            image_embedder: Redux encoder
            image_strengths: Optional per-image strength multipliers
            redux_mode: ComfyUI compatible downsampling strength
            interpolation_mode: Interpolation method for downsampling (ComfyUI compatible)
                - "area": Weighted average (default, smooth results)
                - "bicubic": Cubic interpolation (very smooth, may blur)
                - "nearest": Nearest neighbor (sharp, preserves details)
        """
        image_embeds_list = []
        for idx, image_path in enumerate(image_paths):
            strength = 1.0
            if image_strengths is not None and idx < len(image_strengths):
                strength = image_strengths[idx]

            image_embeds = ReduxUtilDownsampling._embed_single_image(
                image_path=image_path,
                image_encoder=image_encoder,
                image_embedder=image_embedder,
                strength=strength,
                redux_mode=redux_mode,
                interpolation_mode=interpolation_mode,
            )
            image_embeds_list.append(image_embeds)
        return image_embeds_list

    @staticmethod
    def _embed_single_image(
        image_path: str | Path,
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
        strength: float = 1.0,
        redux_mode: Literal["lowest", "low", "medium", "high", "highest"] = "medium",
        interpolation_mode: Literal["area", "bicubic", "nearest"] = "area",
    ) -> mx.array:
        """
        Embed single image with true patch downsampling and interpolation modes.
        """
        image = ImageUtil.load_image(image_path).convert("RGB")
        image = ImageUtil.preprocess_for_model(image=image)

        # Get image latents from SigLIP encoder
        image_latents, pooler_output = image_encoder(image)

        # Process through Redux encoder FIRST (to get the tokens)
        image_embeds = image_embedder(image_latents)

        # Apply downsampling to the REDUX TOKENS (not the latents)
        processed_embeds = ReduxUtilDownsampling._apply_token_downsampling(image_embeds, redux_mode, interpolation_mode)

        # Apply individual strength factor
        if strength != 1.0:
            processed_embeds = processed_embeds * strength

        return processed_embeds

    @staticmethod
    def _apply_token_downsampling(
        image_embeds: mx.array,
        redux_mode: Literal["lowest", "low", "medium", "high", "highest"] = "medium",
        interpolation_mode: Literal["area", "bicubic", "nearest"] = "area",
    ) -> mx.array:
        """
        Apply true token downsampling following ComfyUI approach with interpolation modes.

        Args:
            image_embeds: Redux tokens [batch, num_tokens, hidden_size]
            redux_mode: Downsampling strength
            interpolation_mode: How to perform the downsampling

        Returns:
            Downsampled tokens
        """
        batch_size, num_tokens, hidden_size = image_embeds.shape

        # Determine downsampling factor based on mode (ComfyUI values)
        downsampling_factors = {
            "lowest": 5,  # Minimal image influence (maximum downsampling)
            "low": 4,  # Low image influence
            "medium": 3,  # Balanced influence (ComfyUI default)
            "high": 2,  # High image influence
            "highest": 1,  # Maximum image influence (minimum downsampling)
        }

        factor = downsampling_factors.get(redux_mode, 3)

        # No downsampling needed for factor 1
        if factor <= 1:
            return image_embeds

        # Assume tokens are arranged in spatial grid
        # For 729 tokens, this should be 27x27
        spatial_size = int(num_tokens**0.5)

        if spatial_size * spatial_size != num_tokens:
            # Can't determine spatial structure, apply simple reduction
            return ReduxUtilDownsampling._simple_token_reduction(image_embeds, factor)

        # Perfect square case - treat all as spatial
        return ReduxUtilDownsampling._downsample_spatial_tokens(
            image_embeds, spatial_size, factor, hidden_size, batch_size, interpolation_mode
        )

    @staticmethod
    def _downsample_spatial_tokens(
        spatial_tokens: mx.array,
        spatial_size: int,
        factor: float,
        hidden_size: int,
        batch_size: int,
        interpolation_mode: str = "area",
    ) -> mx.array:
        """
        Downsample spatial tokens using different interpolation methods.
        """
        # Reshape to spatial grid
        tokens_2d = spatial_tokens.reshape(batch_size, spatial_size, spatial_size, hidden_size)

        # Calculate output size (handle fractional factors)
        output_size = max(1, int(spatial_size / factor))

        # Choose interpolation method
        if interpolation_mode == "area":
            return ReduxUtilDownsampling._area_interpolation(tokens_2d, output_size, batch_size, hidden_size)
        elif interpolation_mode == "bicubic":
            return ReduxUtilDownsampling._bicubic_interpolation(tokens_2d, output_size, batch_size, hidden_size)
        elif interpolation_mode == "nearest":
            return ReduxUtilDownsampling._nearest_interpolation(tokens_2d, output_size, batch_size, hidden_size)
        else:
            # Default to area
            return ReduxUtilDownsampling._area_interpolation(tokens_2d, output_size, batch_size, hidden_size)

    @staticmethod
    def _area_interpolation(tokens_2d: mx.array, output_size: int, batch_size: int, hidden_size: int) -> mx.array:
        """
        Area interpolation: weighted average of all pixels in the source region.
        This is the default ComfyUI method - smooth and preserves information.
        """
        input_h, input_w = tokens_2d.shape[1], tokens_2d.shape[2]

        # Calculate step sizes
        step_h = input_h / output_size
        step_w = input_w / output_size

        output_tokens = []

        for i in range(output_size):
            for j in range(output_size):
                # Calculate source region
                start_h = int(i * step_h)
                end_h = int((i + 1) * step_h)
                start_w = int(j * step_w)
                end_w = int((j + 1) * step_w)

                # Ensure we don't go out of bounds
                end_h = min(end_h, input_h)
                end_w = min(end_w, input_w)

                # Extract and average the region
                region = tokens_2d[:, start_h:end_h, start_w:end_w, :]
                averaged = mx.mean(region, axis=(1, 2))
                output_tokens.append(averaged)

        # Stack all tokens
        result = mx.stack(output_tokens, axis=1)
        return result

    @staticmethod
    def _bicubic_interpolation(tokens_2d: mx.array, output_size: int, batch_size: int, hidden_size: int) -> mx.array:
        """
        Bicubic interpolation: smooth interpolation using cubic polynomials.
        Results in very smooth outputs but may introduce blur.
        """
        input_h, input_w = tokens_2d.shape[1], tokens_2d.shape[2]

        # For simplicity, we'll use a weighted average with broader kernel
        # True bicubic would require complex coefficient calculations
        step_h = input_h / output_size
        step_w = input_w / output_size

        output_tokens = []

        for i in range(output_size):
            for j in range(output_size):
                # Center of the output pixel in input coordinates
                center_h = (i + 0.5) * step_h - 0.5
                center_w = (j + 0.5) * step_w - 0.5

                # Expanded region for bicubic (larger kernel)
                start_h = max(0, int(center_h - step_h))
                end_h = min(input_h, int(center_h + step_h) + 1)
                start_w = max(0, int(center_w - step_w))
                end_w = min(input_w, int(center_w + step_w) + 1)

                # Extract region and apply gaussian-like weighting
                region = tokens_2d[:, start_h:end_h, start_w:end_w, :]

                # Simple weighted average (approximation of bicubic)
                # Create distance-based weights
                h_coords = mx.arange(start_h, end_h, dtype=mx.float32)
                w_coords = mx.arange(start_w, end_w, dtype=mx.float32)

                h_weights = 1.0 / (1.0 + mx.abs(h_coords - center_h))
                w_weights = 1.0 / (1.0 + mx.abs(w_coords - center_w))

                # Broadcast weights
                weights_2d = h_weights[:, None] * w_weights[None, :]
                weights_2d = weights_2d / mx.sum(weights_2d)

                # Apply weights
                weighted_sum = mx.sum(region * weights_2d[None, :, :, None], axis=(1, 2))
                output_tokens.append(weighted_sum)

        result = mx.stack(output_tokens, axis=1)
        return result

    @staticmethod
    def _nearest_interpolation(tokens_2d: mx.array, output_size: int, batch_size: int, hidden_size: int) -> mx.array:
        """
        Nearest neighbor interpolation: takes the closest pixel value.
        Preserves sharp details but may lose information.
        """
        input_h, input_w = tokens_2d.shape[1], tokens_2d.shape[2]

        step_h = input_h / output_size
        step_w = input_w / output_size

        output_tokens = []

        for i in range(output_size):
            for j in range(output_size):
                # Find nearest pixel
                nearest_h = int((i + 0.5) * step_h)
                nearest_w = int((j + 0.5) * step_w)

                # Clamp to bounds
                nearest_h = min(nearest_h, input_h - 1)
                nearest_w = min(nearest_w, input_w - 1)

                # Take the nearest token
                nearest_token = tokens_2d[:, nearest_h, nearest_w, :]
                output_tokens.append(nearest_token)

        result = mx.stack(output_tokens, axis=1)
        return result

    @staticmethod
    def _simple_token_reduction(tokens: mx.array, factor: float) -> mx.array:
        """
        Fallback: simple token reduction by taking every Nth token.
        """
        if factor <= 1:
            return tokens

        # Take every factor-th token
        step = max(1, int(factor))
        return tokens[:, ::step, :]

    @staticmethod
    def get_mode_description(mode: Literal["lowest", "low", "medium", "high", "highest"]) -> str:
        """Get description of downsampling modes (ComfyUI compatible)."""
        descriptions = {
            "lowest": "Minimal image influence (5x downsampling) - prompt-driven",
            "low": "Low image influence (4x downsampling)",
            "medium": "Balanced influence (3x downsampling) - ComfyUI default",
            "high": "High image influence (2x downsampling)",
            "highest": "Maximum image influence (1x downsampling) - image-focused",
        }
        return descriptions.get(mode, "Unknown mode")

    @staticmethod
    def get_interpolation_description(mode: Literal["area", "bicubic", "nearest"]) -> str:
        """Get description of interpolation modes (ComfyUI compatible)."""
        descriptions = {
            "area": "Area interpolation - weighted average (smooth, default)",
            "bicubic": "Bicubic interpolation - very smooth (may blur details)",
            "nearest": "Nearest neighbor - sharp edges (preserves details)",
        }
        return descriptions.get(mode, "Unknown interpolation mode")

from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.flux_tools.redux.redux_advanced import Flux1ReduxAdvanced
from mflux.ui import defaults as ui_defaults
from mflux.ui.cli.parsers import CommandLineParser
from mflux.ui.prompt_utils import get_effective_prompt


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using the advanced Redux tool with enhanced control.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_redux_arguments()
    parser.add_output_arguments()

    # Add advanced redux mode argument
    parser.add_argument(
        "--redux-mode",
        type=str,
        choices=["lowest", "low", "medium", "high", "highest"],
        default="medium",
        help="Redux downsampling mode (ComfyUI compatible): 'lowest' for minimal image influence (5x downsampling), 'low' for low influence (4x), 'medium' for balanced (3x), 'high' for high influence (2x), 'highest' for maximum influence (1x)",
    )

    # Add interpolation mode argument (optional)
    parser.add_argument(
        "--interpolation-mode",
        type=str,
        choices=["area", "bicubic", "nearest"],
        default="area",
        help="Interpolation method for downsampling (ComfyUI compatible, optional): 'area' for weighted average (smooth, default), 'bicubic' for cubic interpolation (very smooth, may blur), 'nearest' for nearest neighbor (sharp, preserves details)",
    )

    args = parser.parse_args()

    # 0. Set default guidance value if not provided by user
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE

    # Validate and normalize redux image strengths
    redux_image_strengths = _validate_redux_image_strengths(
        redux_image_paths=args.redux_image_paths,
        redux_image_strengths=args.redux_image_strengths,
    )

    # 1. Load the model
    flux = Flux1ReduxAdvanced(
        model_config=ModelConfig.dev_redux(),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(args=args, flux=flux)

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value with advanced redux control
            image = flux.generate_image(
                seed=seed,
                prompt=get_effective_prompt(args),
                config=Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    redux_image_paths=args.redux_image_paths,
                    redux_image_strengths=redux_image_strengths,
                ),
                redux_mode=args.redux_mode,
                interpolation_mode=args.interpolation_mode,
            )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _validate_redux_image_strengths(
    redux_image_paths: list[Path],
    redux_image_strengths: list[float] | None,
) -> list[float] | None:
    if not redux_image_strengths or len(redux_image_strengths) == 0:
        return redux_image_strengths

    # If strengths provided but not enough for all images, fill with default (1.0)
    if len(redux_image_strengths) < len(redux_image_paths):
        redux_image_strengths.extend([1.0] * (len(redux_image_paths) - len(redux_image_strengths)))

    # If too many strengths provided, raise error
    elif len(redux_image_strengths) > len(redux_image_paths):
        raise ValueError(
            f"Too many strengths provided ({len(redux_image_strengths)}), expected at most {len(redux_image_paths)}."
        )

    return redux_image_strengths


if __name__ == "__main__":
    main()

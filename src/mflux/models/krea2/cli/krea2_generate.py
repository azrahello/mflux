from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.conditioning import ConditioningBands
from mflux.models.common.config import ModelConfig
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# Krea-2 turbo defaults (reference: 8 steps, CFG 1.0, er_sde, shift 1.15).
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 1.0


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Krea-2 based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_conditioning_arguments()
    parser.add_img_ref_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.img_ref:
        missing = [p for p in args.img_ref if not Path(p).exists()]
        if missing:
            parser.error(f"--img-ref file(s) not found: {', '.join(missing)}")

    conditioning_weights = (
        ConditioningBands.parse_weights(args.conditioning_weights)
        if args.conditioning_weights
        else ui_defaults.CONDITIONING_WEIGHTS_DEFAULT["krea2"]
    )
    conditioning_multiplier = (
        args.conditioning_multiplier
        if args.conditioning_multiplier is not None
        else ui_defaults.CONDITIONING_MULTIPLIER_DEFAULT["krea2"]
    )
    guidance_schedule = args.guidance_schedule or ui_defaults.GUIDANCE_SCHEDULE_DEFAULT["krea2"]

    # 1. Load the model
    model = Krea2(
        model_config=ModelConfig.krea2(),
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks (stepwise image output, memory stats, battery saver)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Krea2LatentCreator,
    )

    try:
        steps = args.steps if args.steps is not None else DEFAULT_STEPS
        guidance = args.guidance if args.guidance is not None else DEFAULT_GUIDANCE
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                image_path=args.image_path,
                image_strength=args.image_strength,
                conditioning_weights=conditioning_weights,
                conditioning_renormalize=args.conditioning_renormalize,
                conditioning_multiplier=conditioning_multiplier,
                guidance_schedule=guidance_schedule,
                img_ref_paths=args.img_ref,
                img_ref_details=args.img_ref_detail,
                img_ref_rebalance=args.img_ref_rebalance,
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()

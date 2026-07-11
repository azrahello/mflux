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

# Krea-2 turbo defaults (reference: 8 steps, CFG 1.0, er_sde; sigmas use the official
# dynamic exponential shift, base/max 0.5/1.15 over image seq len 256..6400).
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE = 1.0


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Krea-2 based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="krea-2")
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_conditioning_arguments()
    parser.add_conditioning_crossover_arguments()
    parser.add_projector_rebalance_arguments()
    parser.add_img_ref_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.img_ref:
        missing = [p for p in args.img_ref if not Path(p).exists()]
        if missing:
            parser.error(f"--img-ref file(s) not found: {', '.join(missing)}")

    if args.edit_ref:
        if len(args.edit_ref) > 3:
            parser.error("--edit-ref supports at most 3 reference images.")
        missing = [p for p in args.edit_ref if not Path(p).exists()]
        if missing:
            parser.error(f"--edit-ref file(s) not found: {', '.join(missing)}")
        if args.img_ref or args.img_ref_rebalance:
            parser.error("--edit-ref and --img-ref/--img-ref-rebalance are mutually exclusive.")

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
    conditioning_clamp = (
        args.conditioning_clamp
        if args.conditioning_clamp is not None
        else ui_defaults.CONDITIONING_CLAMP_DEFAULT["krea2"]
    )
    guidance_schedule = args.guidance_schedule or ui_defaults.GUIDANCE_SCHEDULE_DEFAULT["krea2"]
    conditioning_crossover = (
        args.conditioning_crossover
        if args.conditioning_crossover is not None
        else ui_defaults.CONDITIONING_CROSSOVER_DEFAULT["krea2"]
    )
    conditioning_overlap = (
        args.conditioning_overlap
        if args.conditioning_overlap is not None
        else ui_defaults.CONDITIONING_OVERLAP_DEFAULT["krea2"]
    )
    projector_rebalance_weights = (
        args.projector_rebalance_weights
        if args.projector_rebalance_weights is not None
        else ui_defaults.PROJECTOR_REBALANCE_WEIGHTS_DEFAULT["krea2"]
    )
    if projector_rebalance_weights.strip().lower() in ("", "none", "off"):
        projector_rebalance_weights = None

    # 1. Load the model
    model = Krea2(
        model_config=ModelConfig.krea2(),
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        projector_rebalance_weights=projector_rebalance_weights,
        projector_rebalance_strength=args.projector_rebalance_strength,
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
                conditioning_clamp=conditioning_clamp,
                conditioning_crossover=conditioning_crossover,
                conditioning_overlap=conditioning_overlap,
                guidance_schedule=guidance_schedule,
                img_ref_paths=args.img_ref,
                img_ref_details=args.img_ref_detail,
                img_ref_rebalance=args.img_ref_rebalance,
                edit_ref_paths=args.edit_ref,
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

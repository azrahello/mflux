import warnings
from argparse import BooleanOptionalAction, Namespace

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator import Krea2LatentCreator
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# Krea-2 turbo defaults (reference: 8 steps, CFG 1.0, er_sde; sigmas use the official
# dynamic exponential shift, base/max 0.5/1.15 over image seq len 256..6400). The 8-step
# count lives in ui_defaults.MODEL_INFERENCE_STEPS under this model's registry key; the
# parser applies it, so main() sees an already-resolved args.steps.
DEFAULT_MODEL = "krea-2"
DEFAULT_GUIDANCE = 1.0


CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance other than 1.0",
        "reason": "the encoder builds the unconditional branch only when guidance != 1.0, so at the "
        "distilled default of 1.0 the negative prompt is never encoded.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Krea-2 based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_pid_decode_arguments()
    projector_group = parser.add_argument_group("Text-fusion projector rebalance")
    projector_group.add_argument(
        "--projector-rebalance-weights",
        type=str,
        default=None,
        help="12 comma-separated diffs added to txtfusion.projector (a model patch, not a conditioning scale). Default: no patch; --projector-rebalance-strength alone applies the ComfyUI identity-edit preset.",
    )
    projector_group.add_argument(
        "--projector-rebalance-strength",
        type=float,
        default=None,
        help="Scales the diffs before they are added (default 0.05 when the patch is active; 0 disables it).",
    )
    conditioning_group = parser.add_argument_group("Conditioning rebalance")
    conditioning_group.add_argument(
        "--conditioning-weights",
        type=str,
        default=None,
        help=f"12 comma-separated gains on the tapped text-encoder layers, or 'off'. Default: {Krea2Initializer.CONDITIONING_WEIGHTS_DEFAULT} (also used in training).",
    )
    conditioning_group.add_argument(
        "--conditioning-clamp",
        type=float,
        default=None,
        help="Optional abs clamp after the gains and multiplier, against the attention-sink NaN on raw boosts (default 0 = off).",
    )
    conditioning_group.add_argument(
        "--conditioning-multiplier",
        type=float,
        default=1.0,
        help="Global gain on all 12 taps, applied after the renormalize (default 1; artifacts past ~2-3).",
    )
    conditioning_group.add_argument(
        "--conditioning-renormalize",
        action=BooleanOptionalAction,
        default=True,
        help="Rescale the weighted conditioning back to its original RMS, as the ComfyUI node does (default: on).",
    )
    parser.add_output_arguments()
    return parser


class Krea2Command:
    latent_creator = Krea2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only. --model accepts
        # only krea-2 / krea-2-raw aliases, so a foreign name is never silently run as Krea-2-Turbo.
        return ConfigResolution.resolve_restricted(
            args.model, DEFAULT_MODEL, model_path=args.model_path, extra_keys=("krea-2-raw",)
        )

    @staticmethod
    def load(args: Namespace) -> Krea2:
        projector_kwargs = Krea2Initializer.projector_rebalance_kwargs(
            getattr(args, "projector_rebalance_weights", None), getattr(args, "projector_rebalance_strength", None)
        )
        return Krea2(
            model_config=Krea2Command.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
            **projector_kwargs,
            conditioning_weights=getattr(args, "conditioning_weights", None),
            conditioning_clamp=getattr(args, "conditioning_clamp", None),
            conditioning_multiplier=getattr(args, "conditioning_multiplier", 1.0),
            conditioning_renormalize=getattr(args, "conditioning_renormalize", True),
        )

    @staticmethod
    def generate(model: Krea2, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        # ScaleFactor dims ("2x", and the "auto" default) need the reference image; resolved
        # here so a caller of generate() gets them without going through main().
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=args.steps,
            height=height,
            width=width,
            guidance=Krea2Command._guidance(args),
            scheduler=args.scheduler,
            negative_prompt=args.negative_prompt,
            image_path=args.image_path,
            image_strength=args.image_strength,
            pid_decode=args.pid_decode,
            pid_degrade_sigma=args.pid_degrade_sigma,
        )

    @staticmethod
    def _guidance(args: Namespace) -> float:
        # Read by generate() and by main()'s warning; never written back into args.
        return args.guidance if args.guidance is not None else DEFAULT_GUIDANCE


def main():
    # 0. Parse command line arguments
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load the model
    model = Krea2Command.load(args)

    # 2. Register callbacks (stepwise image output, memory stats, battery saver)
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Krea2Command.latent_creator,
    )

    try:
        # The declared condition, checked once the default has resolved: the encoder
        # only builds the unconditional branch when guidance != 1.0. It reads sys.argv,
        # so it stays here, not in generate().
        if Krea2Command._guidance(args) == 1.0 and CommandLineParser._option_was_provided("--negative-prompt"):
            warnings.warn(
                "--negative-prompt is ignored at guidance 1.0; " + CONDITIONAL_OPTIONS["--negative-prompt"]["reason"],
                stacklevel=2,
            )
        # Sizes are resolved once per run, before the first prompt is read. generate()
        # resolves again for direct callers, which changes nothing on plain numbers.
        args.width, args.height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = Krea2Command.generate(model, args, seed, PromptUtil.read_prompt(args))
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()

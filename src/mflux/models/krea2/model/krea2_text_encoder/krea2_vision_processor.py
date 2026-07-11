from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor


class Krea2VisionLanguageProcessor(QwenVisionLanguageProcessor):
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer,
            image_processor=QwenImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        )

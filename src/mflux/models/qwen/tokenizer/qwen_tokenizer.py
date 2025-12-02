import re

import mlx.core as mx
from transformers import Qwen2Tokenizer


class TokenizerQwen:
    def __init__(self, tokenizer: Qwen2Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.template_start_idx = 34

    @staticmethod
    def _fix_blue_color_bias(prompt: str) -> str:
        """
        Fix Qwen's specific bias against the word 'blue' by replacing it with 'azure'.
        Qwen has a known issue where it ignores or misinterprets the bare word 'blue',
        but correctly handles the synonym 'azure'.

        Only replaces 'blue' when it's not already qualified (e.g., preserves 'pale blue', 'bright blue').
        """
        # Replace bare 'blue' with 'azure' only if not already qualified
        # Negative lookbehind to check for common qualifiers
        qualified_pattern = r"(?<!bright )(?<!dark )(?<!light )(?<!deep )(?<!pale )(?<!vivid )(?<!intense )(?<!soft )(?<!muted )(?<!rich )(?<!warm )(?<!cool )(?<!neon )(?<!pastel )\bblue\b"
        fixed_prompt = re.sub(qualified_pattern, "azure", prompt, flags=re.IGNORECASE)

        return fixed_prompt

    def tokenize(self, prompt: str) -> tuple[mx.array, mx.array]:
        # Fix Qwen's blue color bias
        fixed_prompt = self._fix_blue_color_bias(prompt)
        formatted_text = self.prompt_template.format(fixed_prompt)
        tokens = self.tokenizer(
            formatted_text,
            max_length=self.max_length + self.template_start_idx,
            padding=False,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        return input_ids, attention_mask

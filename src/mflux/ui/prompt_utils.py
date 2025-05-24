import typing as t
from pathlib import Path

from mflux.error.exceptions import PromptFileReadError


def read_prompt_file(prompt_file_path: Path) -> str:
    # Check if file exists
    if not prompt_file_path.exists():
        raise PromptFileReadError(f"Prompt file does not exist: {prompt_file_path}")

    try:
        with open(prompt_file_path, "rt") as f:
            content: str = f.read().strip()

        # Validate content
        if not content:
            raise PromptFileReadError(f"Prompt file is empty: {prompt_file_path}")

        print(f"Using prompt from file: {prompt_file_path}")
        return content
    except (IOError, OSError) as e:
        raise PromptFileReadError(f"Error reading prompt file '{prompt_file_path}': {e}")


def get_effective_prompt(args: t.Any) -> dict:
    prompt_t5 = None
    prompt_clip = None

    if args.prompt_file is not None:
        content = read_prompt_file(args.prompt_file)

        # Se il contenuto del file è un singolo prompt
        if isinstance(content, str):
            if "--clip:" in content:
                parts = content.split("--clip:", 1)
                prompt_t5 = parts[0].strip()
                prompt_clip = parts[1].strip()
            else:
                prompt_t5 = content.strip()

        # Se il contenuto del file è già un dict con entrambi i prompt
        elif isinstance(content, dict):
            prompt_t5 = content.get("prompt_t5", None)
            prompt_clip = content.get("prompt_clip", None)

    elif args.prompt is not None:
        prompt_t5 = args.prompt.strip()

    # Se viene fornito esplicitamente il clip
    if args.clip is not None:
        prompt_clip = args.clip.strip()

    return {"prompt_t5": prompt_t5, "prompt_clip": prompt_clip}

import time

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig

# Importa dal modo corretto
from mflux.models.qwen.qwen_initializer import QwenInitializer

# Carica il modello
print("Caricamento modello...")
model_config = ModelConfig.qwen_image()
model = QwenInitializer(
    model_config=model_config,
    quantization=4,
    local_path=None,
    lora_paths=None,
    lora_scales=None,
).load()

print("Modello caricato!")

# Genera con timing per step
print("\nGenerazione con timing dettagliato per step:")
config = Config(
    num_inference_steps=10,
    height=512,
    width=512,
)

prompt = "test"
seed = 42

# Misura tempo manualmente guardando l'output
print("\nINIZIO GENERAZIONE - Osserva i tempi per step\n")

start_total = time.time()
image = model.generate_image(seed=seed, prompt=prompt, config=config)
end_total = time.time()

print("\n=== RIEPILOGO ===")
print(f"Tempo totale: {end_total - start_total:.2f}s")
print(f"Media per step: {(end_total - start_total) / 10:.2f}s/step")

import mlx.core as mx

from mflux import QwenImage

model = QwenImage.from_alias("qwen-image")

# Codifica prompt
prompt = "blue sheets"

# Estrai text embeddings
with mx.stream(mx.cpu):
    text_emb = model.text_encoder.encode(prompt)

# Verifica che "blue" abbia encoding forte
print("Text embedding shape:", text_emb.shape)
print("Text embedding mean:", text_emb.mean())
print("Text embedding std:", text_emb.std())

# Confronta con "yellow sheets"
text_emb_yellow = model.text_encoder.encode("yellow sheets")

# La differenza dovrebbe essere significativa
diff = mx.abs(text_emb - text_emb_yellow).mean()
print(f"Difference blue vs yellow: {diff}")

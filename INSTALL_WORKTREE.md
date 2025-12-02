# Installazione del Worktree per Testare STORK

## Problema
L'errore indica che stai usando l'installazione di mflux dalla repository principale (`~/mflux`) invece del worktree con STORK.

## Soluzione

### Opzione 1: Reinstalla mflux dal worktree (Raccomandato)

```bash
cd ~/.claude-worktrees/mflux/condescending-lehmann

# Disinstalla la versione corrente
pip uninstall mflux -y

# Installa dal worktree in modalità editable
pip install -e .

# Verifica l'installazione
python -c "from mflux.schedulers import STORKScheduler; print('✓ STORK disponibile')"
```

### Opzione 2: Usa direttamente Python invece del CLI

```bash
cd ~/.claude-worktrees/mflux/condescending-lehmann

# Crea uno script di test
cat > test_stork_qwen.py << 'EOF'
import sys
sys.path.insert(0, "src")

from mflux import ModelConfig
from mflux.config.config import Config
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

# Carica il modello
qwen = QwenImage(
    model_config=ModelConfig.from_alias("qwen_image"),
    local_path="/Volumes/NewHome/test/mflux_models/qwenwhite_8step_fp16_franky"
)

# Genera con STORK-4
image = qwen.generate_image(
    seed=1010,
    prompt="A beautiful sunset over mountains",
    config=Config(
        num_inference_steps=13,
        height=1248,
        width=832,
        scheduler="stork-4",
        guidance=1.0,
    )
)

image.save("test_stork4.png")
print("✓ Immagine generata con STORK-4!")
EOF

python test_stork_qwen.py
```

### Opzione 3: Merge nel branch main e reinstalla

```bash
# Nel worktree
cd ~/.claude-worktrees/mflux/condescending-lehmann
git add -A
git commit -m "feat: Add STORK-2 and STORK-4 schedulers for faster sampling"

# Torna alla repo principale
cd ~/mflux
git checkout main
git merge condescending-lehmann

# Reinstalla
pip install -e .
```

## Verifica dell'Installazione

Dopo aver installato, verifica che STORK sia disponibile:

```bash
python -c "from mflux.schedulers import SCHEDULER_REGISTRY; print('stork-4' in SCHEDULER_REGISTRY)"
# Dovrebbe stampare: True
```

## Test Rapido

```bash
# Test con Qwen
mflux-generate-qwen \
  --model "AlessandroRizzo/whiteQWEN-alpha01" \
  --path /Volumes/NewHome/test/mflux_models/qwenwhite_8step_fp16_franky \
  --scheduler stork-4 \
  --steps 13 \
  --prompt "A beautiful sunset" \
  --output test_stork4.png
```

## Troubleshooting

### "NotImplementedError: The scheduler 'stork-4' is not implemented"
- Significa che stai usando la vecchia installazione
- Segui l'Opzione 1 per reinstallare

### "ModuleNotFoundError: No module named 'mflux'"
- Devi installare mflux: `pip install -e .`

### "ImportError: cannot import name 'STORKScheduler'"
- Assicurati di essere nel worktree corretto
- Verifica che `src/mflux/schedulers/stork_scheduler.py` esista

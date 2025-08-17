#!/bin/bash

# Test comparativo Redux Advanced con downsampling patch
echo "🧪 Testing Redux Advanced with Interpolation Modes"
echo "=============================================="

IMAGE_PATH="/Users/alessandrorizzo/image_23.png"
PROMPT="anime style, Studio Ghibli, beautiful magical artwork"
SEED=12345
STEPS=12

echo ""
echo "📸 Image: $IMAGE_PATH"
echo "✨ Prompt: $PROMPT"
echo "🎲 Seed: $SEED"
echo ""

# Test tutte e cinque le modalità ComfyUI
for mode in lowest low medium high highest; do
    echo "🔄 Testing $mode mode..."
    
    case $mode in
        "lowest")
            echo "   → Minimal image influence (5x downsampling) - prompt-driven"
            ;;
        "low")
            echo "   → Low image influence (4x downsampling)"
            ;;
        "medium") 
            echo "   → Balanced influence (3x downsampling) - ComfyUI default"
            ;;
        "high")
            echo "   → High image influence (2x downsampling)"
            ;;
        "highest")
            echo "   → Maximum image influence (1x downsampling) - image-focused"
            ;;
    esac
    
    mflux-generate-redux-advanced \
        --prompt "$PROMPT" \
        --redux-image-paths "$IMAGE_PATH" \
        --redux-mode $mode \
        --seed $SEED \
        --steps $STEPS \
        --output "redux_comparison_${mode}.png" \
        --metadata
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $mode mode completed successfully"
    else
        echo "   ❌ $mode mode failed"
    fi
    echo ""
done

echo "🎉 Test completato!"
echo ""
echo "📝 Note: All modes use 'area' interpolation (ComfyUI default)"
echo "    You can also test different interpolation modes:"
echo "    --interpolation-mode area    (smooth, default)"
echo "    --interpolation-mode bicubic (very smooth, may blur)"
echo "    --interpolation-mode nearest (sharp, preserves details)"
echo ""
echo "📋 Risultati generati:"
echo "   - redux_comparison_lowest.png (minima influenza immagine - 5x downsampling)"
echo "   - redux_comparison_low.png (bassa influenza - 4x downsampling)"
echo "   - redux_comparison_medium.png (influenza bilanciata - 3x downsampling)"
echo "   - redux_comparison_high.png (alta influenza - 2x downsampling)"
echo "   - redux_comparison_highest.png (massima influenza - 1x downsampling)"
echo ""
echo "💡 Confronta le immagini per vedere l'effetto del downsampling!"

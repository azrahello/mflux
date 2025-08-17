#!/bin/bash

# Test avanzato per valutare l'efficacia del downsampling
echo "🔬 Advanced Redux Downsampling Analysis"
echo "======================================"

IMAGE_PATH="/Users/alessandrorizzo/image_23.png"
SEED=9999

# Test diversi scenari
declare -a SCENARIOS=(
    "weak_prompt:minimal anime hint:8"
    "medium_prompt:anime style, beautiful colors:10" 
    "strong_prompt:highly detailed anime artwork, Studio Ghibli style, masterpiece, beautiful lighting, magical atmosphere:12"
)

for scenario in "${SCENARIOS[@]}"; do
    IFS=':' read -r test_name prompt steps <<< "$scenario"
    
    echo ""
    echo "📝 Scenario: $test_name"
    echo "   Prompt: $prompt"
    echo "   Steps: $steps"
    echo ""
    
    for mode in weak medium strong; do
        echo "   Testing $mode mode..."
        
        mflux-generate-redux-advanced \
            --prompt "$prompt" \
            --redux-image-paths "$IMAGE_PATH" \
            --redux-mode $mode \
            --seed $SEED \
            --steps $steps \
            --output "${test_name}_${mode}.png" > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo "   ✅ ${mode} completed"
        else
            echo "   ❌ ${mode} failed"
        fi
    done
done

echo ""
echo "🎯 Test completati! Analizza i risultati:"
echo ""
echo "📊 Weak Prompt (dovrebbe mostrare grosse differenze tra modalità):"
echo "   - weak_prompt_weak.png"
echo "   - weak_prompt_medium.png" 
echo "   - weak_prompt_strong.png"
echo ""
echo "📊 Medium Prompt (influenza bilanciata):"
echo "   - medium_prompt_weak.png"
echo "   - medium_prompt_medium.png"
echo "   - medium_prompt_strong.png"
echo ""
echo "📊 Strong Prompt (prompt potrebbe dominare in weak mode):"
echo "   - strong_prompt_weak.png"
echo "   - strong_prompt_medium.png"
echo "   - strong_prompt_strong.png"
